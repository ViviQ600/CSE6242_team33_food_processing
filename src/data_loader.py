"""Chunked Open Food Facts loading with a source-verified all-column cache."""

import gzip
import json
import time
from functools import partial
import pandas as pd
from IPython.display import display, Markdown
from .utils import PipelineConfig, WorkflowState, save_table
from .visualization import show_figure


def load_data(config: PipelineConfig) -> WorkflowState:
    """Extract the exact U.S. market/date cohort while retaining all raw columns.

    Dates are UTC last-modification timestamps, not launch dates or historical
    product versions. Bad TSV lines raise errors. Cache provenance is checked
    against source byte size, nanosecond modification time, and filter settings.
    """
    raw = config.raw_path
    if not raw.is_file():
        raise FileNotFoundError(
            f"Raw OFF archive not found: {raw}. Set TEAM33_RAW_PATH or edit PipelineConfig."
        )
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    output = config.output_dir
    directories = [output, output / "tables", output / "figures", output / "models"]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    cache = config.cache_dir / "us_window_all_211_columns.csv.gz"
    metadata_path = config.cache_dir / "all_columns_manifest.json"
    default_window = (config.start_date, config.end_date_exclusive) == ("2024-07-01", "2026-07-01")
    filter_key = (
        "exact_country_tag_20240701_20260701_exclusive_v1"
        if default_window
        else f"exact_country_tag_{config.start_date}_{config.end_date_exclusive}_exclusive_v1"
    )
    stat = raw.stat()
    fingerprint = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "filter": filter_key}
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    verified = cache.exists() and all(metadata.get(k) == v for k, v in fingerprint.items())
    start = pd.Timestamp(config.start_date, tz="UTC")
    end = pd.Timestamp(config.end_date_exclusive, tz="UTC")
    if not verified:
        retained = scanned = us_rows = 0
        first = True
        temporary = cache.with_suffix(".tmp")
        columns = []
        started = time.monotonic()
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as writer:
            for number, chunk in enumerate(
                pd.read_csv(
                    raw,
                    sep="\t",
                    dtype="string",
                    compression="gzip",
                    chunksize=50000,
                    on_bad_lines="error",
                ),
                1,
            ):
                columns = list(chunk.columns)
                us = (
                    chunk["countries_tags"]
                    .fillna("")
                    .str.lower()
                    .str.contains(r"(?:^|,)\s*en:united-states\s*(?:,|$)", regex=True)
                )
                timestamp = pd.to_numeric(chunk["last_modified_t"], errors="coerce")
                selected = chunk.loc[
                    us & timestamp.ge(start.timestamp()) & timestamp.lt(end.timestamp())
                ]
                if len(selected):
                    selected.to_csv(writer, index=False, header=first)
                    first = False
                scanned += len(chunk)
                us_rows += int(us.sum())
                retained += len(selected)
                if number % 10 == 0:
                    print(
                        f"Scanned {scanned:,}; retained {retained:,}; elapsed {time.monotonic()-started:.0f}s",
                        flush=True,
                    )
            if first:
                pd.DataFrame(columns=columns).to_csv(writer, index=False)
        temporary.replace(cache)
        metadata = {
            **fingerprint,
            "source": str(raw),
            "raw_rows": scanned,
            "filtered_rows": retained,
            "columns": len(columns),
            "us_market_rows": us_rows,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
    else:
        print("All-column cache verified against the source archive.")
    df_raw = pd.read_csv(cache, dtype="string", low_memory=False)
    if len(df_raw) != metadata["filtered_rows"] or len(df_raw.columns) != metadata["columns"]:
        raise ValueError("Cache dimensions do not match its manifest; rebuild the cache.")
    manifest = {
        "source_path": str(raw),
        "source_bytes": metadata["bytes"],
        "raw_rows_scanned": metadata["raw_rows"],
        "us_window_rows": len(df_raw),
        "columns_loaded": len(df_raw.columns),
        "filter_version": filter_key,
        "us_market_rows": metadata.get("us_market_rows"),
        "market_count_source": "cache manifest; unavailable in older cache manifests",
    }
    actual_latest = pd.to_datetime(
        pd.to_numeric(df_raw["last_modified_t"], errors="coerce"), unit="s", utc=True
    ).max()
    state = WorkflowState()
    state.update(
        {
            "df_raw": df_raw,
            "manifest": manifest,
            "actual_latest": actual_latest,
            "OUTPUT_DIR": output,
            "TABLE_DIR": output / "tables",
            "FIG_DIR": output / "figures",
            "MODEL_DIR": output / "models",
            "SEED": config.seed,
            "RUN_FINAL_TEST": config.run_final_test,
            "RUN_TEXT_BASELINE": config.run_text_baseline,
            "RUN_SHAP": config.run_shap,
            "REFIT_FOR_DEPLOYMENT": config.refit_for_deployment,
            "save_table": partial(save_table, table_dir=output / "tables"),
            "show_figure": partial(show_figure, figure_dir=output / "figures"),
        }
    )
    display(pd.DataFrame([manifest]))
    display(
        Markdown(
            f"**Window records: {len(df_raw):,}. Latest observed modification: {actual_latest}.** "
            "This is a current-snapshot cohort. Products updated after the window may be absent; "
            "the extract does not reconstruct historical inventories or recipes."
        )
    )
    display(df_raw.head(3)[["code", "product_name", "nova_group", "nutriscore_grade"]])
    return state
