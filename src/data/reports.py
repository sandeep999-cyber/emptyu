"""Summary Reporter for dataset statistics, storage size, and validation outputs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import config
from src.data.db import db_manager


class ReportsGenerator:
    """Generates markdown reports for storage, validation, missing data, and dataset statistics."""

    def generate_storage_summary(self) -> str:
        """Generate storage size summary report."""
        files = db_manager.query_files()
        total_size = sum(f["file_size"] for f in files) if files else 0
        total_rows = sum(f["row_count"] for f in files) if files else 0

        report = "# Storage Size Summary Report\n\n"
        report += f"- **Total Parquet Files**: {len(files)}\n"
        report += f"- **Total Dataset Rows**: {total_rows:,}\n"
        report += f"- **Total Storage Size**: {total_size / (1024 * 1024):.2f} MB\n\n"
        report += "## File Details\n"
        report += "| File ID | Symbol | Market | Type | Interval | Year/Month | Rows | Size (MB) |\n"
        report += "|---|---|---|---|---|---|---|---|\n"

        for f in files:
            size_mb = f["file_size"] / (1024 * 1024)
            ym = f"{f['year']}-{f['month']:02d}" if f["month"] is not None else str(f["year"])
            report += f"| {f['file_id']} | {f['symbol']} | {f['market']} | {f['dataset_type']} | {f['interval']} | {ym} | {f['row_count']:,} | {size_mb:.2f} |\n"

        return report

    def save_report(self, content: str, path: Path) -> None:
        """Save report content to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


reports_generator = ReportsGenerator()
