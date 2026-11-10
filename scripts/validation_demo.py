"""Show the validation stage catching defects on a deliberately corrupted sample.

MovieLens 1M is clean, so the pipeline's own validation report shows only
warnings. This script runs the *same* validation code on the small corrupted
sample used by the unit tests (src/validation/samples.py) and writes a separate report,
without touching the pipeline's data.

    python -m scripts.validation_demo
"""
from __future__ import annotations

from src.common.config import get_paths, load_config
from src.common.io import df_to_markdown
from src.validation.validate import render_report, validate_tables
from src.validation.samples import NOW, dirty_sample


def main() -> None:
    cfg = load_config()
    paths = get_paths(cfg)
    out = validate_tables(dirty_sample(), cfg, now_ts=NOW)
    md = render_report(out, cfg, "src/validation/samples.py dirty_sample() - demo only, not pipeline data")
    md = md.replace("# Data Validation Report", "# Data Validation Report - corrupted-sample demo", 1)
    md += "\n## Quarantined rows (what would be written to data/errors/)\n"
    for t, df in out["errors"].items():
        if len(df):
            md += f"\n### {t}\n\n" + df_to_markdown(df) + "\n"
    dest = paths.reports / "validation" / "validation_demo_corrupted_sample.md"
    dest.write_text(md, encoding="utf-8")
    print(f"wrote {dest.relative_to(paths.root)}")
    for t, s in out["summary"].items():
        print(f"{t:8s} in={s['rows_in']:3d} quarantined={s['rows_quarantined']} warnings={s['rows_with_warnings']}")


if __name__ == "__main__":
    main()
