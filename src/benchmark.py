"""Time feature engineering, model fitting, and Tableau export by sample size."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .analysis import (
    INGREDIENT_FEATURES,
    engineer_features,
    load_processed_data,
)
from .data_processing import clean_data
from .modeling import make_pipeline
from .utils import CORE_NUTRIENTS, save_table
from .visualization import export_tableau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results"
SAMPLE_SIZES = (10_000, 50_000, 100_000)


def benchmark_scalability(raw, output_dir, sizes=SAMPLE_SIZES, seed=33):
    """Measure the three primary pipeline stages on deterministic samples."""
    records = []
    feature_columns = INGREDIENT_FEATURES + CORE_NUTRIENTS + ["category"]
    for requested_size in (*sizes, "full"):
        if requested_size == "full":
            sample = raw.copy()
            label = "full"
        else:
            sample_size = int(requested_size)
            if sample_size > len(raw):
                raise ValueError(
                    f"Requested {sample_size:,} rows but only {len(raw):,} are available."
                )
            sample = raw.sample(n=sample_size, random_state=seed)
            label = sample_size

        started = perf_counter()
        frame, _ = engineer_features(sample)
        feature_seconds = perf_counter() - started

        eligible = frame.loc[
            frame.nova_target.notna() & ~frame.needs_review
        ]
        if eligible.nova_target.nunique() != 4:
            raise ValueError(
                f"Benchmark sample {label} does not contain all four NOVA classes."
            )
        training = eligible.sample(frac=0.8, random_state=seed)
        nutrients = [
            column for column in CORE_NUTRIENTS if column in training.columns
        ]
        model = make_pipeline("logistic", nutrients)
        started = perf_counter()
        model.fit(training[feature_columns], training.nova_target.astype(int))
        training_seconds = perf_counter() - started

        frame["nutri_split"] = "inference"
        grades = pd.DataFrame(
            {
                "product_id": frame.product_id,
                "grade_display": frame.nutri_target.str.upper(),
                "grade_source": np.where(
                    frame.nutri_target.notna(), "OFF_observed", "unknown"
                ),
                "formula_score_2023": np.nan,
            }
        )
        started = perf_counter()
        with TemporaryDirectory(prefix="tableau_export_benchmark_") as temporary:
            export_tableau(
                frame,
                model,
                "logistic",
                grades,
                pd.DataFrame(),
                pd.DataFrame(),
                temporary,
                run_final_test=False,
            )
        export_seconds = perf_counter() - started
        records.append(
            {
                "sample_size": label,
                "rows_engineered": len(frame),
                "labeled_training_rows": len(training),
                "feature_engineering_seconds": feature_seconds,
                "model_training_seconds": training_seconds,
                "tableau_export_seconds": export_seconds,
                "total_seconds": feature_seconds
                + training_seconds
                + export_seconds,
            }
        )
        print(
            label,
            "feature/train/export seconds:",
            round(feature_seconds, 2),
            round(training_seconds, 2),
            round(export_seconds, 2),
        )
    result = pd.DataFrame(records)
    save_table(result, output_dir, "e5_scalability_benchmark.csv")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args(argv)
    raw, _ = load_processed_data(DATA_DIR, clean_function=clean_data)
    with threadpool_limits(limits=2):
        benchmark_scalability(raw, OUTPUT_DIR, seed=args.seed)


if __name__ == "__main__":
    main()
