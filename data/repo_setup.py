from pathlib import Path
from textwrap import dedent

root = Path.cwd()

directories = [
    "docs",
    "data/raw",
    "data/processed",
    "results/figures",
    "results/tables",
    "results/models",
    "notebooks",
    "src",
    "tests",
    "config",
]

for directory in directories:
    (root / directory).mkdir(parents=True, exist_ok=True)

files = {
    ".gitignore": dedent("""
        # Byte-compiled / optimized / DLL files
        __pycache__/
        *.py[cod]
        *$py.class

        # C extensions
        *.so

        # Distribution / packaging
        .Python
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib/
        lib64/
        parts/
        sdist/
        var/
        wheels/
        pip-wheel-metadata/
        share/python-wheels/
        *.egg-info/
        .installed.cfg
        *.egg
        PIPFILE.lock

        # PyInstaller
        *.manifest
        *.spec

        # Jupyter Notebook
        .ipynb_checkpoints

        # IPython
        profile_default/
        ipython_config.py

        # pyenv
        .python-version

        # Environments
        .env
        .venv
        env/
        venv/
        ENV/
        env.bak/
        venv.bak/

        # IDE
        .vscode/
        .idea/
        *.swp
        *.swo
        *~
        .DS_Store

        # Data and results
        data/raw/*
        !data/raw/.gitkeep
        data/processed/*
        !data/processed/.gitkeep
        results/figures/*
        !results/figures/.gitkeep
        results/tables/*
        !results/tables/.gitkeep
        results/models/*
        !results/models/.gitkeep

        # Logs
        *.log

        # OS
        .DS_Store
        Thumbs.db
    """).strip() + "\n",

    "requirements.txt": dedent("""
        # Data Processing and Analysis
        pandas>=1.3.0
        numpy>=1.21.0
        scipy>=1.7.0
        scikit-learn>=1.0.0

        # Visualization
        matplotlib>=3.4.0
        seaborn>=0.11.0
        plotly>=5.0.0

        # Jupyter
        jupyter>=1.0.0
        ipython>=7.0.0
        notebook>=6.0.0

        # Utilities
        pyyaml>=21.0.0
        python-dotenv>=0.19.0
        requests>=2.26.0

        # Testing
        pytest>=6.2.0
        pytest-cov>=2.12.0

        # Code Quality
        black>=21.0
        flake8>=3.9.0
        isort>=5.9.0
    """).strip() + "\n",

    "README.md": dedent("""
        # CSE 6242 Team 33 - Food Processing Analysis

        ## Project Overview

        This repository contains analysis and processing of food processing data for CSE 6242 Data & Visual Analytics course at Georgia Tech. Team 33 focuses on [add specific research question/objective here].

        ## Project Structure

        ```text
        CSE6242_team33_food_processing/
        ├── docs/                   # Project documentation
        ├── data/                   # Data files (raw and processed)
        ├── notebooks/              # Jupyter notebooks for analysis
        ├── src/                    # Python source code modules
        ├── results/                # Output results and visualizations
        ├── tests/                  # Unit tests
        ├── config/                 # Configuration files
        └── README.md
        ```

        ## Getting Started

        ### Prerequisites
        - Python 3.8+
        - pip or conda

        ### Installation

        1. Clone this repository:
        ```bash
        git clone https://github.com/ViviQ600/CSE6242_team33_food_processing.git
        cd CSE6242_team33_food_processing
        ```

        2. Create a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

        3. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

        ## Usage

        1. Place raw data in `data/raw/`
        2. Run notebooks in sequence:
           - `01_data_exploration.ipynb`
           - `02_data_cleaning.ipynb`
           - `03_analysis.ipynb`
        3. Results will be saved in `results/`

        ### Running Tests
        ```bash
        pytest tests/ -v
        ```

        ## Team Members
        - [Team member names]

        ## Documentation
        - [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
        - [INSTALLATION.md](docs/INSTALLATION.md)
        - [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
        - [METHODOLOGY.md](docs/METHODOLOGY.md)

        ## License
        This project is part of CSE 6242 coursework.
    """).strip() + "\n",

    "setup.py": dedent("""
        from setuptools import setup, find_packages

        with open("README.md", "r", encoding="utf-8") as fh:
            long_description = fh.read()

        with open("requirements.txt", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

        setup(
            name="cse6242_team33_food_processing",
            version="0.1.0",
            author="Team 33",
            description="Food Processing Analysis for CSE 6242",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/ViviQ600/CSE6242_team33_food_processing",
            packages=find_packages(),
            classifiers=[
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "License :: OSI Approved :: MIT License",
            ],
            python_requires=">=3.8",
            install_requires=requirements,
        )
    """).strip() + "\n",

    "docs/PROJECT_OVERVIEW.md": dedent("""
        # Project Overview

        ## Title
        Food Processing Analysis - CSE 6242 Team 33

        ## Objective
        [Add specific research question and objectives]

        ## Key Questions
        1. [Research question 1]
        2. [Research question 2]
        3. [Research question 3]

        ## Data Description
        [Describe the dataset]
        - Source: [data source]
        - Size: [number of records]
        - Time period: [date range]
        - Variables: [number of features]

        ## Methodology
        [Brief overview of analysis approach]

        ## Expected Outcomes
        [Describe expected results and deliverables]

        ## Timeline
        - Week 1: Data exploration and cleaning
        - Week 2: Analysis and visualization
        - Week 3: Documentation and presentation

        ## References
        [Add relevant references and citations]
    """).strip() + "\n",

    "docs/INSTALLATION.md": dedent("""
        # Installation Guide

        ## System Requirements
        - Python 3.8 or higher
        - pip or conda
        - Git

        ## Step-by-Step Installation

        ### 1. Clone the Repository
        ```bash
        git clone https://github.com/ViviQ600/CSE6242_team33_food_processing.git
        cd CSE6242_team33_food_processing
        ```

        ### 2. Create Virtual Environment
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

        ### 3. Install Dependencies
        ```bash
        pip install -r requirements.txt
        ```

        ### 4. Verify Installation
        ```bash
        python -c "import pandas; import numpy; print('Installation successful!')"
        ```

        ## Troubleshooting
        - If modules are missing, run `pip install -r requirements.txt`
        - If Jupyter kernel is missing, run:
          ```bash
          pip install ipykernel
          python -m ipykernel install --user --name team33_env --display-name "Team33"
          ```

        ## Next Steps
        - Read [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
        - Review [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
        - Start with notebooks in `notebooks/`
    """).strip() + "\n",

    "docs/DATA_DICTIONARY.md": dedent("""
        # Data Dictionary

        ## Dataset Information
        ### Source
        [Add data source information]

        ### File Format
        [CSV, Excel, JSON, etc.]

        ### Time Period
        [Add date range]

        ## Variables

        | Variable Name | Data Type | Description | Units | Range/Values |
        |---|---|---|---|---|
        | [Variable 1] | [type] | [description] | [units] | [range] |
        | [Variable 2] | [type] | [description] | [units] | [range] |
        | [Variable 3] | [type] | [description] | [units] | [range] |

        ## Data Quality Notes
        - Missing values: [Describe missing data handling]
        - Outliers: [Note any outliers]
        - Data validation: [Validation rules applied]

        ## Data Preparation
        Raw data should be placed in `data/raw/`.
        Processed data will be saved to `data/processed/`.
    """).strip() + "\n",

    "docs/METHODOLOGY.md": dedent("""
        # Methodology

        ## Analysis Approach

        ### Phase 1: Data Exploration
        - Descriptive statistics
        - Data distribution analysis
        - Missing value analysis

        ### Phase 2: Data Preparation
        - Handling missing values
        - Outlier detection
        - Data normalization

        ### Phase 3: Analysis
        - [Specify analytical techniques]
        - [Statistical tests]
        - [Machine learning models]

        ### Phase 4: Visualization
        - Summary charts
        - Key findings dashboard
        - Final report visuals

        ## Tools and Technologies
        - Python 3.8+
        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Seaborn
        - Plotly
        - Jupyter

        ## Validation
        - Unit tests in `tests/`
        - Data validation checks
        - Reproducibility checks

        ## Reproducibility
        All scripts and workflows are version-controlled and documented for analysis repeatability.
    """).strip() + "\n",

    "data/README.md": dedent("""
        # Data Directory

        ## Structure

        ### raw/
        Contains the original, unmodified raw data.

        ### processed/
        Contains cleaned and transformed data ready for analysis.

        ## Data Management Guidelines
        1. Keep raw data unchanged
        2. Write processed data to `data/processed/`
        3. Document all transformation steps
        4. Record source and metadata clearly

        ## Example
        ```python
        import pandas as pd

        raw_data = pd.read_csv("data/raw/your_file.csv")
        processed_data = pd.read_csv("data/processed/your_file_processed.csv")
        ```
    """).strip() + "\n",

    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "results/figures/.gitkeep": "",
    "results/tables/.gitkeep": "",
    "results/models/.gitkeep": "",
    "notebooks/.gitkeep": "",

    "src/__init__.py": dedent("""
        \"\"\"CSE 6242 Team 33 Food Processing Analysis Package\"\"\"

        __version__ = "0.1.0"
        __author__ = "Team 33"

        from . import data_processing
        from . import analysis
        from . import visualization
        from . import utils

        __all__ = ["data_processing", "analysis", "visualization", "utils"]
    """).strip() + "\n",

    "src/data_processing.py": dedent("""
        \"\"\"Data Processing Module

        Handles loading, cleaning, and transforming data.
        \"\"\"

        from pathlib import Path
        import pandas as pd

        def load_raw_data(filepath):
            \"\"\"Load raw data from a CSV or Excel file.\"\"\"
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        def clean_data(df):
            \"\"\"Remove duplicates and basic invalid entries.\"\"\"
            df = df.drop_duplicates()
            return df

        def save_processed_data(df, output_path):
            \"\"\"Save processed data to a CSV file.\"\"\"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
    """).strip() + "\n",

    "src/analysis.py": dedent("""
        \"\"\"Analysis Module

        Contains functions for descriptive statistics and basic modeling.
        \"\"\"

        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        def compute_descriptive_statistics(df):
            \"\"\"Compute summary statistics for a dataframe.\"\"\"
            return df.describe()

        def perform_pca(df, n_components=2):
            \"\"\"Apply PCA to numeric features.\"\"\"
            numeric_df = df.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df)
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(scaled)
            return transformed, pca, pca.explained_variance_ratio_

        def correlation_analysis(df):
            \"\"\"Compute correlation matrix for numeric columns.\"\"\"
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df.corr()
    """).strip() + "\n",

    "src/visualization.py": dedent("""
        \"\"\"Visualization Module

        Contains functions for chart generation.
        \"\"\"

        from pathlib import Path
        import matplotlib.pyplot as plt
        import seaborn as sns

        def set_style():
            \"\"\"Set plot style.\"\"\"
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            plt.rcParams["figure.figsize"] = (12, 6)

        def plot_distributions(df, output_dir="results/figures/"):
            \"\"\"Create distribution plots for numeric columns.\"\"\"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            numeric_cols = df.select_dtypes(include=["number"]).columns

            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].hist(bins=50, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(output_dir / f"{col}_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()

        def plot_correlation_heatmap(correlation_matrix, output_path="results/figures/correlation_heatmap.png"):
            \"\"\"Create a correlation heatmap.\"\"\"
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Matrix Heatmap")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
    """).strip() + "\n",

    "src/utils.py": dedent("""
        \"\"\"Utility Functions

        General helper utilities for the project.
        \"\"\"

        import json
        from pathlib import Path
        import yaml

        def load_config(config_path="config/config.yaml"):
            \"\"\"Load configuration from a YAML file.\"\"\"
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        def save_config(config, output_path="config/config.yaml"):
            \"\"\"Save config to YAML.\"\"\"
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False)

        def create_directory_structure(base_path="."):
            \"\"\"Ensure project folders exist.\"\"\"
            directories = [
                "data/raw",
                "data/processed",
                "results/figures",
                "results/tables",
                "results/models",
                "notebooks",
            ]
            for directory in directories:
                path = Path(base_path) / directory
                path.mkdir(parents=True, exist_ok=True)

        def print_summary(data):
            \"\"\"Print a summary of data or dictionary content.\"\"\"
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data.to_string())
    """).strip() + "\n",

    "tests/__init__.py": '"""Test package for CSE 6242 Team 33"""\n',

    "tests/test_data_processing.py": dedent("""
        \"\"\"Unit tests for data_processing module\"\"\"

        import pandas as pd
        import pytest

        from src.data_processing import clean_data, load_raw_data

        class TestDataProcessing:
            @pytest.fixture
            def sample_data(self):
                return pd.DataFrame({
                    "A": [1, 2, 2, 4],
                    "B": [5, 6, 6, 8],
                    "C": [9, 10, 11, 12],
                })

            def test_clean_data_removes_duplicates(self, sample_data):
                cleaned = clean_data(sample_data)
                assert len(cleaned) == len(sample_data.drop_duplicates())

            def test_load_raw_data_nonexistent_file(self):
                with pytest.raises(FileNotFoundError):
                    load_raw_data("nonexistent_file.csv")
    """).strip() + "\n",

    "tests/test_analysis.py": dedent("""
        \"\"\"Unit tests for analysis module\"\"\"

        import numpy as np
        import pandas as pd
        import pytest

        from src.analysis import compute_descriptive_statistics, correlation_analysis

        class TestAnalysis:
            @pytest.fixture
            def sample_data(self):
                np.random.seed(42)
                return pd.DataFrame({
                    "X": np.random.randn(100),
                    "Y": np.random.randn(100),
                    "Z": np.random.randn(100),
                })

            def test_descriptive_statistics(self, sample_data):
                stats = compute_descriptive_statistics(sample_data)
                assert not stats.empty

            def test_correlation_analysis(self, sample_data):
                corr = correlation_analysis(sample_data)
                assert corr.shape == (3, 3)
                assert np.allclose(np.diag(corr), np.ones(3))
    """).strip() + "\n",

    "config/config.yaml": dedent("""
        # Project Configuration File
        # CSE 6242 Team 33 - Food Processing Analysis

        data:
          raw_dir: data/raw
          processed_dir: data/processed
          file_format: csv

        analysis:
          pca_components: 2
          test_size: 0.2
          random_state: 42

        visualization:
          dpi: 300
          style: whitegrid
          palette: husl
          figure_size: [12, 6]

        output:
          figures_dir: results/figures
          tables_dir: results/tables
          models_dir: results/models

        logging:
          level: INFO
          format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """).strip() + "\n",
}

for relative_path, content in files.items():
    file_path = root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

print("Repository structure and templates created successfully.")
print("Next steps:")
print("  git add .")
print("  git commit -m \"Initial project structure setup\"")
print("  git push origin main")
