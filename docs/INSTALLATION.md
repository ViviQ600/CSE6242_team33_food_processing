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
