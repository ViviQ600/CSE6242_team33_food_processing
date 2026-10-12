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
