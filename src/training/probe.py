"""Colab dry-run probe for picking training efficiency settings.

Runs a warmup pass (triggers + caches torch.compile) then a timed pass for
micro_batch_size in {32, 64} x torch_compile in {false, true}, reporting steady-
state step time, projected full-epoch time, and peak VRAM. Output is meant to
inform edits to configs/trainer_v1_scale30.yaml before the full 30-epoch run.
"""

import argparse
import json
import re
import time
from pathlib import Path
import yaml
import torch

from src.training.trainer import TeacherTrainer

FULL_TRAIN_WINDOWS = 65638


def _apply_recommendation(src_path: str, micro: int, compile_on: bool, out_path: str) -> None:
    """Write a tuned copy of the trainer yaml (micro_batch/torch_compile/compile_mode).

    The tracked config file is never modified: a tuned copy is written to
    ``out_path`` (Drive-backed on Colab via the models/ symlink), so future
    ``git pull --ff-only`` calls in the notebook never conflict and the tuned
    settings persist across Colab sessions.
    """
    text = Path(src_path).read_text()

    def _repl(pattern: str, value: str) -> None:
        nonlocal text
        updated, n = re.subn(pattern, value, text, count=1, flags=re.MULTILINE)
        if n == 0:
            raise RuntimeError(f"Could not locate {pattern!r} in {src_path}")
        text = updated

    def _set(key: str, value: str) -> None:
        nonlocal text
        if re.search(rf"^\s*{key}:", text, flags=re.MULTILINE):
            _repl(rf"^(\s*{key}:\s*)\S+", lambda m: m.group(1) + value)
        else:
            anchor = re.search(r"^\s*torch_compile:.*$", text, flags=re.MULTILINE)
            if anchor is None:
                raise RuntimeError(f"No torch_compile anchor to insert {key} after in {src_path}")
            indent = anchor.group(0)[: len(anchor.group(0)) - len(anchor.group(0).lstrip())]
            text = text[: anchor.end()] + f"\n{indent}{key}: {value}" + text[anchor.end():]

    _set("micro_batch_size", str(micro))
    _set("torch_compile", "true" if compile_on else "false")
    # CUDA graphs (reduce-overhead) cut kernel-launch overhead for the static
    # train batch; keep compile_mode consistent with torch_compile.
    _set("compile_mode", "reduce-overhead" if compile_on else "null")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[Probe] Wrote tuned config micro_batch={micro} torch_compile={compile_on} "
          f"compile_mode={'reduce-overhead' if compile_on else 'null'} -> {out}")


def _run_case(model_cfg, opt_cfg, base, micro, comp, n_train, n_val, run_dir):
    tc = dict(base)
    tc.update(
        micro_batch_size=micro,
        torch_compile=comp,
        # Probe the exact compile_mode that --apply writes, so the recommended
        # setting is the one that was actually measured (and verified not to crash).
        compile_mode="reduce-overhead" if comp else None,
        epochs=1,
        max_train_windows=n_train,
        max_val_windows=n_val,
        val_every=1,
        checkpoint_every=1,
    )
    trainer = TeacherTrainer(model_cfg, opt_cfg, tc, Path(run_dir))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    del trainer
    torch.cuda.empty_cache()
    return elapsed, vram


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", default="configs/model_v1.yaml")
    ap.add_argument("--optimizer-config", default="configs/optimizer_v1.yaml")
    ap.add_argument("--trainer-config", default="configs/trainer_v1_scale30.yaml")
    ap.add_argument("--warmup-windows", type=int, default=256)
    ap.add_argument("--timed-windows", type=int, default=2048)
    ap.add_argument("--apply", action="store_true",
                    help="Write the recommended micro_batch_size/torch_compile to --apply-to")
    ap.add_argument("--apply-to", default="models/foundation/teacher_v1/probe_config.yaml",
                    help="Where to write the tuned config (Drive-backed on Colab via models/ symlink)")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.trainer_config).read_text())["trainer"]
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    opt_cfg = yaml.safe_load(Path(args.optimizer_config).read_text())
    eff_batch = int(base.get("batch_size", 64))

    cases = [(32, False), (32, True), (64, False)]
    rows = []
    for micro, comp in cases:
        accum = max(1, eff_batch // max(1, micro))
        try:
            _run_case(model_cfg, opt_cfg, base, micro, comp,
                      args.warmup_windows, 64, f"/tmp/probe_warm_m{micro}_c{comp}")
            elapsed, vram = _run_case(model_cfg, opt_cfg, base, micro, comp,
                                      args.timed_windows, 128, f"/tmp/probe_timed_m{micro}_c{comp}")
            n_steps = (args.timed_windows // micro) // accum
            rows.append({
                "micro_batch_size": micro,
                "torch_compile": comp,
                "step_ms": round(elapsed / max(1, n_steps) * 1000, 1),
                "proj_full_epoch_min": round(elapsed * (FULL_TRAIN_WINDOWS / args.timed_windows) / 60, 1),
                "peak_vram_gb": round(vram, 2),
            })
        except Exception as e:  # e.g. CUDA OOM for micro 64
            rows.append({
                "micro_batch_size": micro,
                "torch_compile": comp,
                "step_ms": None,
                "proj_full_epoch_min": None,
                "peak_vram_gb": None,
                "error": str(e)[:200],
            })
        print(json.dumps(rows[-1]))

    viable = [r for r in rows if r.get("proj_full_epoch_min") is not None]
    if viable:
        rec = min(viable, key=lambda r: r["proj_full_epoch_min"])
        print("RECOMMEND: micro_batch_size={} torch_compile={} "
              "(proj ~{:.1f} min/epoch, peak {:.2f} GB)".format(
                  rec["micro_batch_size"], rec["torch_compile"],
                  rec["proj_full_epoch_min"], rec["peak_vram_gb"]))
        if rec["torch_compile"]:
            print("HINT: compile_mode: reduce-overhead is auto-set on apply for CUDA-graph "
                  "launch overhead (safe: drop_last=True keeps the train batch static).")
        print("NOTE: the inductor compile cache is now warm, so the training subprocess "
              "in this same session reuses it (no recompile).")
        if args.apply:
            _apply_recommendation(args.trainer_config, rec["micro_batch_size"],
                                  rec["torch_compile"], args.apply_to)
            print("TUNED_CONFIG=" + str(Path(args.apply_to).resolve()))
    else:
        print("RECOMMEND: all probed cases failed; keep micro_batch_size=32, torch_compile=false")


if __name__ == "__main__":
    main()
