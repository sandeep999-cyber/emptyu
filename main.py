"""Unified Command-Line Interface for Pure Market Foundation Model Dataset Builder."""

import argparse
import asyncio
from pathlib import Path
import sys
import time
from rich.console import Console

from src.config import config
from src.data.binance_vision import downloader
from src.data.binance_rest import rest_client
from src.data.db import db_manager
from src.data.parquet_converter import converter
from src.data.resampler import resampler
from src.data.calendar_builder import calendar_builder
from src.data.lake import lake
from src.data.validator import validator
from src.data.quality_report import quality_reporter
from src.data.manifest_builder import manifest_builder
from src.data.snapshot_manager import snapshot_manager
from src.data.datacard_builder import datacard_builder
from src.data.pipeline_manifest import pipeline_manifest
from src.data.feature_builder import feature_builder
from src.data.windowing import windowing_engine
from src.data.market_dataset import MarketDataset
from src.data.reports import reports_generator
from src.data.modality_registry import modality_registry
from src.data.metadata import metadata_manager
from src.logger import get_stage_logger

console = Console()

_DEFAULT_SYMBOLS = config.dataset.get("default_symbols", ["BTCUSDT"])
_DEFAULT_MARKET = config.dataset.get("default_market", "futures")
_DEFAULT_YEAR = 2024
_DEFAULT_MONTH = 1
_DATASET_TYPES = list(config.download.get("datasets", ["klines", "funding", "open_interest"]))
_SNAPSHOT_DATE = config.dataset.get("snapshot_date", "2026-07-30")

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def _resolve_symbols(symbols_arg: str) -> list:
    if symbols_arg:
        return symbols_arg.split(",")
    return _DEFAULT_SYMBOLS


def _record_pipeline_stage(stage: str, inputs: list, outputs: list, checksums: list, duration: float, meta=None, timestamp: str = "unknown"):
    pipeline_manifest.record_stage(
        stage_name=stage, inputs=inputs, outputs=outputs, checksums=checksums,
        duration_seconds=duration, metadata=meta or {}, timestamp=timestamp,
    )
    pipeline_manifest.save()


def cmd_download(args):
    """Execute historical download."""
    log = get_stage_logger("download")
    console.print("[bold green]Starting Download...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    start_year = args.start_year or args.year or _DEFAULT_YEAR
    limit_months = args.limit_months or 1
    log.info(f"Download start: symbols={symbols} market={market} start_year={start_year} months={limit_months}")

    t0 = time.time()
    try:
        asset_info_list = rest_client.get_all_symbols(market)
        for info in asset_info_list:
            if info["symbol"] in symbols:
                db_manager.register_asset(
                    symbol=info["symbol"], market_type=info["market_type"],
                    base_asset=info["base_asset"], quote_asset=info["quote_asset"],
                    is_active=info["is_active"], listing_date=info.get("listing_date"),
                    delisting_date=info.get("delisting_date"),
                    contract_type=info.get("contract_type"),
                    tick_size=info.get("tick_size", 0.0), step_size=info.get("step_size", 0.0),
                    min_qty=info.get("min_qty", 0.0), contract_size=info.get("contract_size", 1.0),
                )
                # Archive exchange_info per year
                ei_dir = config.canonical_dir / market / info["symbol"] / "exchange_info"
                ei_dir.mkdir(parents=True, exist_ok=True)
                ei_path = ei_dir / f"{start_year}.json"
                if not ei_path.exists():
                    import json as _json
                    ei_path.write_text(_json.dumps(info, indent=2), encoding="utf-8")
    except Exception as e:
        console.print(f"[red]Error fetching exchange info: {e}[/red]")
        log.error(f"exchangeInfo fetch failed: {e}")
        sys.exit(EXIT_FAILURE)

    download_date = time.strftime("%Y-%m-%d")

    async def run_dl():
        dl_results = []
        for sym in symbols:
            for ds_type in _DATASET_TYPES:
                for month_offset in range(limit_months):
                    month = ((start_year - start_year) * 12 + (args.month or 1) + month_offset - 1) % 12 + 1
                    year = start_year + ((args.month or 1) + month_offset - 1) // 12
                    zip_path = await downloader.download_monthly_archive(market, sym, ds_type, year, month)
                    if zip_path:
                        console.print(f"[green]Downloaded:[/green] {zip_path}")
                        log.info(f"Downloaded: {zip_path}")
                        dl_results.append(("ok", sym, ds_type, str(zip_path)))
                    else:
                        console.print(f"[yellow]Download unavailable:[/yellow] {sym} {ds_type} {year}-{month:02d}")
                        dl_results.append(("missing", sym, ds_type, ""))
        return dl_results

    dl_results = asyncio.run(run_dl())
    duration = time.time() - t0
    _record_pipeline_stage(
        "download",
        inputs=[f"{sym}_{ds}" for sym, ds in set((s, d) for _, s, d, _ in dl_results)],
        outputs=[r[3] for r in dl_results if r[0] == "ok"],
        checksums=[], duration=duration,
        meta={"market": market, "year": start_year, "months": limit_months, "download_date": download_date},
    )
    log.info(f"Download complete in {duration:.1f}s")
    console.print("[bold green]Download complete.[/bold green]")


def cmd_convert(args):
    """Convert downloaded raw archives to canonical Snappy Parquets."""
    log = get_stage_logger("convert")
    console.print("[bold green]Starting Parquet Conversion...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    start_year = args.start_year or args.year or _DEFAULT_YEAR
    limit_months = args.limit_months or 1
    download_date = args.download_date or _SNAPSHOT_DATE
    t0 = time.time()
    converted = []
    for sym in symbols:
        for ds_type in _DATASET_TYPES:
            for month_offset in range(limit_months):
                month = ((args.month or 1) + month_offset - 1) % 12 + 1
                year = start_year + ((args.month or 1) + month_offset - 1) // 12
                target_dir = config.raw_dir / market / sym / ds_type / ("1m" if ds_type == "klines" else "")
                zip_filename = f"{sym}-1m-{year}-{month:02d}.zip" if ds_type == "klines" else f"{sym}-{ds_type}-{year}-{month:02d}.zip"
                zip_path = target_dir / zip_filename
                if zip_path.exists():
                    try:
                        out_path = converter.convert_zip_to_parquet(
                            zip_path=zip_path, market=market, symbol=sym,
                            dataset_type=ds_type, interval="1m" if ds_type == "klines" else "",
                            year=year, month=month, download_date=download_date,
                        )
                        console.print(f"[green]Converted:[/green] {out_path}")
                        log.info(f"Converted: {out_path}")
                        converted.append(str(out_path))
                    except Exception as e:
                        console.print(f"[red]Conversion failed for {zip_path}: {e}[/red]")
                        log.error(f"Conversion failed: {zip_path}: {e}")
                        sys.exit(EXIT_FAILURE)
    duration = time.time() - t0
    _record_pipeline_stage("convert", [], converted, [], duration)
    log.info(f"Convert complete in {duration:.1f}s")


def cmd_resample(args):
    """Resample 1m klines into 5m, 15m, 1h, 4h, 1d."""
    log = get_stage_logger("resample")
    console.print("[bold green]Starting Resampling...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    start_year = args.start_year or args.year or _DEFAULT_YEAR
    limit_months = args.limit_months or 1
    target_intervals = ["5m", "15m", "1h", "4h", "1d"]
    t0 = time.time()
    resampled = []
    for sym in symbols:
        for month_offset in range(limit_months):
            month = ((args.month or 1) + month_offset - 1) % 12 + 1
            year = start_year + ((args.month or 1) + month_offset - 1) // 12
            input_1m = config.canonical_dir / market / sym / "klines" / "1m" / f"{year}-{month:02d}.parquet"
            if input_1m.exists():
                for target_tf in target_intervals:
                    try:
                        out_path = resampler.resample_file(input_1m, market, sym, target_tf, year, month)
                        console.print(f"[green]Resampled {target_tf}:[/green] {out_path}")
                        log.info(f"Resampled {target_tf}: {out_path}")
                        resampled.append(str(out_path))
                    except Exception as e:
                        console.print(f"[red]Resample failed for {sym} {target_tf}: {e}[/red]")
                        log.error(f"Resample failed: {sym} {target_tf}: {e}")
                        sys.exit(EXIT_FAILURE)
    duration = time.time() - t0
    _record_pipeline_stage("resample", [], resampled, [], duration)
    log.info(f"Resample complete in {duration:.1f}s")


def cmd_align(args):
    """Explicit alignment step — run alignment engine and report coverage."""
    log = get_stage_logger("alignment")
    console.print("[bold green]Running Alignment...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    t0 = time.time()
    for sym in symbols:
        df_state = lake.market_state(sym, market=market)
        if df_state.empty:
            console.print(f"[yellow]No data for {sym}[/yellow]")
            continue
        total = len(df_state)
        fr_valid = int(df_state["funding_rate"].notna().sum()) if "funding_rate" in df_state.columns else 0
        oi_valid = int(df_state["open_interest"].notna().sum()) if "open_interest" in df_state.columns else 0
        console.print(
            f"[green]{sym}:[/green] {total} rows aligned | "
            f"FR coverage: {fr_valid}/{total} | OI coverage: {oi_valid}/{total}"
        )
        log.info(f"{sym}: {total} rows, FR={fr_valid}, OI={oi_valid}")
    duration = time.time() - t0
    _record_pipeline_stage("alignment", [], [], [], duration, meta={"symbols": symbols, "market": market})
    log.info(f"Alignment complete in {duration:.1f}s")
    console.print("[bold green]Alignment complete.[/bold green]")


def cmd_build_lake(args):
    """Generate calendar and query market state views."""
    log = get_stage_logger("alignment")
    console.print("[bold green]Building Data Lake Views & Calendar...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    year = args.year or _DEFAULT_YEAR
    t0 = time.time()
    for sym in symbols:
        cal_path = calendar_builder.build_calendar_for_year(market, sym, year)
        console.print(f"[green]Calendar created:[/green] {cal_path}")

        df_state = lake.market_state(sym, market=market)
        if not df_state.empty:
            # Generate per-symbol metadata
            meta_dir = config.canonical_dir / market / sym / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)

            ds_ver = metadata_manager.create_dataset_version(
                binance_snapshot=_SNAPSHOT_DATE, created=_SNAPSHOT_DATE,
            )
            metadata_manager.save_json(ds_ver, meta_dir / "dataset_version.json")

            stats = metadata_manager.compute_statistics(df_state)
            metadata_manager.save_json(stats, meta_dir / "statistics_v1.json")

            import shutil, json as _json
            mkt_state_src = config._CONFIGS_DIR / "market_state_schema_v1.json" if hasattr(config, '_CONFIGS_DIR') else Path(__file__).resolve().parent.parent / "configs" / "market_state_schema_v1.json"
            if mkt_state_src.exists():
                shutil.copy2(str(mkt_state_src), str(meta_dir / "market_state_schema_v1.json"))

            datacard = datacard_builder.build_symbol_datacard(sym, market)
            datacard_builder.save_datacard(datacard, meta_dir / "DATASET.md")

        console.print(f"[green]Market State View for {sym}:[/green] {len(df_state)} records aligned.")
        log.info(f"Built lake view for {sym}: {len(df_state)} records")

    duration = time.time() - t0
    _record_pipeline_stage("alignment", [], [], [], duration, meta={"market": market, "year": year, "symbols": symbols})
    log.info(f"Build-lake complete in {duration:.1f}s")


def cmd_validate(args):
    """Audit data integrity and modality validation rules."""
    log = get_stage_logger("validator")
    console.print("[bold green]Running Data Validation...[/bold green]")
    files = db_manager.query_files(
        symbol=args.symbol if args.symbol else None,
        market=args.market if args.market else None,
        dataset_type=args.type if args.type else None,
    )
    console.print(f"[green]Validating {len(files)} indexed files...[/green]")
    errors = 0
    for f in files:
        file_path = Path(f["file_path"])
        is_valid = validator.verify_sha256(file_path, f["sha256"])
        status_str = "[green]VALID[/green]" if is_valid else "[red]INVALID[/red]"
        if not is_valid:
            errors += 1
        console.print(f"File {file_path.name}: {status_str}")
    if errors > 0:
        console.print(f"[red]Validation failed: {errors} file(s) have mismatched checksums.[/red]")
        log.error(f"Validation failed: {errors} files")
        sys.exit(EXIT_FAILURE)
    log.info(f"Validation passed: {len(files)} files")
    console.print("[green]All files pass integrity check.[/green]")


def cmd_quality_report(args):
    """Generate quality_report.json per symbol."""
    log = get_stage_logger("alignment")
    console.print("[bold green]Generating Quality Reports...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    market = args.market or _DEFAULT_MARKET
    t0 = time.time()
    for sym in symbols:
        df_state = lake.market_state(sym, market=market)
        report = quality_reporter.generate_report(sym, market, df_state)
        out_path = config.canonical_dir / market / sym / "metadata" / "quality_report.json"
        quality_reporter.save_report(report, out_path)
        console.print(f"[green]Quality report for {sym}:[/green] score={report['quality_score']}")
        log.info(f"Quality report {sym}: score={report['quality_score']}")
    duration = time.time() - t0
    _record_pipeline_stage("quality_report", [], [], [], duration, meta={"symbols": symbols})
    console.print("[bold green]Quality reports complete.[/bold green]")


def cmd_snapshot(args):
    """Create an immutable reproducible dataset snapshot."""
    log = get_stage_logger("alignment")
    console.print("[bold green]Creating Dataset Snapshot...[/bold green]")
    date_str = args.date or _SNAPSHOT_DATE
    manifest = manifest_builder.build_manifest_from_index(
        snapshot_date=date_str,
        train_symbols=config.dataset.get("train_symbols"),
        val_symbols=config.dataset.get("val_symbols"),
        test_symbols=config.dataset.get("test_symbols"),
        train_end_date=config.dataset.get("train_end_date", "2024-11-30"),
    )
    try:
        snap_path = snapshot_manager.create_snapshot(
            snapshot_date=date_str, manifest_data=manifest,
            checksums_data=None, stats_data={"snapshot": date_str},
        )
        console.print(f"[green]Snapshot created at:[/green] {snap_path}")
        log.info(f"Snapshot created: {snap_path}")

        root_manifest_path = config.training_dir / "training_manifest_v1.json"
        manifest_builder.save_manifest(manifest, root_manifest_path)
        console.print(f"[green]Root manifest:[/green] {root_manifest_path}")

        fingerprint = manifest_builder.compute_fingerprint(manifest)
        fp_path = config.training_dir / "dataset_fingerprint.json"
        manifest_builder.save_fingerprint(fingerprint, fp_path)
        console.print(f"[green]Fingerprint:[/green] {fp_path}")
        log.info(f"Fingerprint: {fingerprint['fingerprint'][:16]}...")

    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        log.error(str(e))
        sys.exit(EXIT_FAILURE)


def cmd_report(args):
    """Generate storage and quality reports."""
    console.print("[bold green]Generating Storage Report...[/bold green]")
    summary = reports_generator.generate_storage_summary()
    out_path = config.training_dir / "storage_summary.md"
    reports_generator.save_report(summary, out_path)
    console.print(f"[green]Report saved to:[/green] {out_path}")


def cmd_benchmark(args):
    """Benchmark DataLoader performance."""
    console.print("[bold green]Running DataLoader Benchmark...[/bold green]")
    symbols = _resolve_symbols(args.symbols)
    sym = symbols[0]
    df_state = lake.market_state(sym)
    if not df_state.empty:
        feats, fm, ts = feature_builder.build_features(df_state)
        wins = windowing_engine.create_windows(feats, fm, ts, metadata={
            "symbol": sym,
            "start_ts": int(ts[0]) if len(ts) > 0 else 0,
            "end_ts": int(ts[-1]) if len(ts) > 0 else 0,
            "snapshot_id": _SNAPSHOT_DATE,
            "modality_config": "modalities_v1.yaml",
            "windowing_config": "windowing_v1.yaml",
        })
        ds = MarketDataset(wins)
        res = benchmark_suite.benchmark_dataloader(ds, batch_size=32, num_batches=50)
        console.print(f"[bold cyan]Benchmark Results:[/bold cyan] {res}")
    else:
        console.print("[yellow]No market state data available.[/yellow]")


from src.training.benchmark import benchmark_suite


def main():
    parser = argparse.ArgumentParser(description="Pure Market Foundation Model Dataset Builder CLI")
    subparsers = parser.add_subparsers(dest="command")
    default_syms = ",".join(_DEFAULT_SYMBOLS)

    dl_p = subparsers.add_parser("download", help="Download raw archives")
    dl_p.add_argument("--symbols", type=str, default=default_syms)
    dl_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)
    dl_p.add_argument("--year", type=int, default=None)
    dl_p.add_argument("--month", type=int, default=None)
    dl_p.add_argument("--start-year", type=int, default=None, help="Start year for range download")
    dl_p.add_argument("--limit-months", type=int, default=1, help="Number of months to download from start-year")
    dl_p.add_argument("--download-date", type=str, default=None, help="Explicit download date for provenance")

    cv_p = subparsers.add_parser("convert", help="Convert raw CSV/ZIP to canonical Parquet")
    cv_p.add_argument("--symbols", type=str, default=default_syms)
    cv_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)
    cv_p.add_argument("--year", type=int, default=None)
    cv_p.add_argument("--month", type=int, default=None)
    cv_p.add_argument("--start-year", type=int, default=None)
    cv_p.add_argument("--limit-months", type=int, default=1)
    cv_p.add_argument("--download-date", type=str, default=None)

    rs_p = subparsers.add_parser("resample", help="Resample 1m klines")
    rs_p.add_argument("--symbols", type=str, default=default_syms)
    rs_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)
    rs_p.add_argument("--year", type=int, default=None)
    rs_p.add_argument("--month", type=int, default=None)
    rs_p.add_argument("--start-year", type=int, default=None)
    rs_p.add_argument("--limit-months", type=int, default=1)

    al_p = subparsers.add_parser("align", help="Run causal alignment and report coverage")
    al_p.add_argument("--symbols", type=str, default=default_syms)
    al_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)

    lk_p = subparsers.add_parser("build-lake", help="Build calendar and Data Lake views")
    lk_p.add_argument("--symbols", type=str, default=default_syms)
    lk_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)
    lk_p.add_argument("--year", type=int, default=_DEFAULT_YEAR)

    vl_p = subparsers.add_parser("validate", help="Run data validation")
    vl_p.add_argument("--symbol", type=str, default=None)
    vl_p.add_argument("--market", type=str, default=None)
    vl_p.add_argument("--type", type=str, default=None)

    qr_p = subparsers.add_parser("quality-report", help="Generate quality_report.json per symbol")
    qr_p.add_argument("--symbols", type=str, default=default_syms)
    qr_p.add_argument("--market", type=str, default=_DEFAULT_MARKET)

    sn_p = subparsers.add_parser("snapshot", help="Create dataset snapshot")
    sn_p.add_argument("--date", type=str, default="2026-07-30")

    subparsers.add_parser("report", help="Generate summary reports")

    bm_p = subparsers.add_parser("benchmark", help="Benchmark DataLoader throughput")
    bm_p.add_argument("--symbols", type=str, default=default_syms)

    args = parser.parse_args()
    try:
        dispatch = {
            "download": cmd_download, "convert": cmd_convert, "resample": cmd_resample,
            "align": cmd_align, "build-lake": cmd_build_lake, "validate": cmd_validate,
            "quality-report": cmd_quality_report, "snapshot": cmd_snapshot,
            "report": cmd_report, "benchmark": cmd_benchmark,
        }
        handler = dispatch.get(args.command)
        if handler:
            handler(args)
        else:
            parser.print_help()
    except Exception as e:
        console.print(f"[red]Unhandled error: {type(e).__name__}: {e}[/red]")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
