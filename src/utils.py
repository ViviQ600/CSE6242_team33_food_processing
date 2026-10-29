"""Configuration, explicit workflow state, and reproducible output utilities.

Importing this module never reads datasets, trains models, or writes files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and switches used by both the notebook and command-line workflow."""

    project_root: Path
    raw_path: Path
    cache_dir: Path
    output_dir: Path
    seed: int = 33
    run_final_test: bool = False
    run_text_baseline: bool = True
    run_shap: bool = True
    refit_for_deployment: bool = False
    start_date: str = "2024-07-01"
    end_date_exclusive: str = "2026-07-01"

    @classmethod
    def from_project(cls, project_root):
        """Use portable project paths, with optional local environment overrides."""
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            raw_path=Path(
                os.environ.get(
                    "TEAM33_RAW_PATH", root / "data/raw/en.openfoodfacts.org.products.csv.gz"
                )
            ),
            cache_dir=Path(os.environ.get("TEAM33_CACHE_DIR", root / "data/cache")),
            output_dir=Path(os.environ.get("TEAM33_OUTPUT_DIR", root / "results")),
        )


@dataclass
class WorkflowState:
    """Explicit intermediate artifacts shared between named workflow steps.

    Each step declares the names it reads using require() and publishes only
    artifacts needed by later steps. Nothing is held in hidden module globals.
    """

    artifacts: dict[str, Any] = field(default_factory=dict)

    def require(self, name):
        if name not in self.artifacts:
            raise KeyError(
                f"Missing workflow artifact {name!r}; run the preceding notebook steps first."
            )
        return self.artifacts[name]

    def update(self, artifacts):
        self.artifacts.update(artifacts)
        return self

    def __getitem__(self, name):
        return self.require(name)


def save_table(table, name, table_dir):
    """Write an actual result table and return it for notebook display."""
    table.to_csv(Path(table_dir) / name, index=False)
    return table
