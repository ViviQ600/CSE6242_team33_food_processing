import gzip
import os
import re
import time
import urllib.request
import pandas as pd

# Save to repo directories
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# File paths
DATA_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
RAW_FILE_PATH = os.path.join(RAW_DIR, "en.openfoodfacts.org.products.csv.gz")

# Filter: Last 3 years
data_years = 3
current_timestamp = int(time.time())
years_sec = 365 * 24 * 3600
cutoff_timestamp = current_timestamp - (data_years * years_sec)

PROCESSED_FILE_PATH = os.path.join(
    PROCESSED_DIR, f"us_openfoodfacts_nutri_nova_last_{data_years}years.csv"
)
UNLABELED_FILE_PATH = os.path.join(
    PROCESSED_DIR, f"us_openfoodfacts_unlabeled_last_{data_years}years.csv"
)

# Target sample size for data processing (Set to None for full dataset extraction)
sample_limit = None

# Required features for multi-target classification & reformulation sandbox
ESSENTIAL_COLS = [
    # Metadata & Identifiers
    "code",
    "product_name",
    "url",
    "countries_en",
    "created_t",
    "last_modified_t",
    # Target Labels for Multi-target Modeling
    "nutriscore_grade",
    "nutriscore_score",
    "nova_group",
    # Text & Additives Features (NLP extraction)
    "ingredients_text",
    "additives_n",
    "additives_tags",
    # Continuous Nutritional Attributes (per 100g)
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "sodium_100g",
]

us_label = re.compile(r"(?i)\b(?:united states|usa|us|america)\b")

# ==============================================================================
# 2. Raw data
# ==============================================================================
# Check alternative local path in root if not in data/raw/
if not os.path.exists(RAW_FILE_PATH) and os.path.exists("openfoodfacts_dump.csv.gz"):
    RAW_FILE_PATH = "openfoodfacts_dump.csv.gz"

if not os.path.exists(RAW_FILE_PATH):
    print(f"Downloading raw dataset archive to: {RAW_FILE_PATH} ...")
    headers = {"User-Agent": "OMSA_CSE6242_Project - Python/3.10"}
    req = urllib.request.Request(DATA_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(
        RAW_FILE_PATH, "wb"
    ) as out_f:
        buf_size = 16 * 1024 * 1024  # 16 MB chunk size
        downloaded = 0
        while True:
            chunk = resp.read(buf_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            print(f"Downloaded: {downloaded / (1024 * 1024):.1f} MB...", end="\r")
    print(f"\nRaw archive saved successfully at '{RAW_FILE_PATH}'.")
else:
    print(f"Found existing raw dataset archive at '{RAW_FILE_PATH}'.")

# ==============================================================================
# 3. Processing data
# ==============================================================================
print(
    f"\nFiltering US products modified within the last {data_years} years..."
)

numeric_fields = [
    "nutriscore_score",
    "nova_group",
    "additives_n",
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "sodium_100g",
]

first_chunk_proc = True
first_chunk_unlab = True
total_records = 0
unlabeled_records = 0

for chunk in pd.read_csv(
    RAW_FILE_PATH,
    sep="\t",
    usecols=ESSENTIAL_COLS,
    compression="gzip",
    chunksize=50000,
    low_memory=False,
    on_bad_lines="skip",
    dtype=str,
):
    # Filter 1: US marketed products
    is_us = chunk["countries_en"].fillna("").str.contains(us_label)

    # Filter 2: Modified within the last 3 years
    mod_ts = pd.to_numeric(chunk["last_modified_t"], errors="coerce").fillna(0)
    is_recent = mod_ts >= cutoff_timestamp

    base_filter = is_us & is_recent
    if not base_filter.any():
        continue

    chunk_base = chunk[base_filter].copy()

    # Filter 3: Contains at least Nutri-Score or NOVA label (excluding unknown)
    valid_nutri = chunk_base["nutriscore_grade"].notna() & (
        ~chunk_base["nutriscore_grade"].str.lower().str.strip().isin(["unknown", "not-applicable", ""])
    )
    valid_nova = chunk_base["nova_group"].notna()
    has_labels = valid_nutri | valid_nova

    subset = chunk_base[has_labels]
    # Unlabeled pool: Neither label is present, but has ingredients or energy for feature extraction
    subset_unlabeled = chunk_base[
        (~has_labels)
        & (chunk_base["ingredients_text"].notna() | chunk_base["energy_100g"].notna())
    ]

    # Convert continuous metrics and group labels to numeric types
    for col in numeric_fields:
        if col in subset.columns:
            subset.loc[:, col] = pd.to_numeric(subset[col], errors="coerce")
        if col in subset_unlabeled.columns:
            subset_unlabeled.loc[:, col] = pd.to_numeric(subset_unlabeled[col], errors="coerce")

    # Stream write processed labeled data directly to disk (Append mode)
    if not subset.empty:
        subset.to_csv(
            PROCESSED_FILE_PATH,
            mode="w" if first_chunk_proc else "a",
            header=first_chunk_proc,
            index=False,
        )
        first_chunk_proc = False
        total_records += len(subset)

    # Stream write unlabeled data for future inference
    if not subset_unlabeled.empty:
        subset_unlabeled.to_csv(
            UNLABELED_FILE_PATH,
            mode="w" if first_chunk_unlab else "a",
            header=first_chunk_unlab,
            index=False,
        )
        first_chunk_unlab = False
        unlabeled_records += len(subset_unlabeled)

    print(f"Extracted labeled records: {total_records:,} | Unlabeled: {unlabeled_records:,} ...", end="\r")

    if sample_limit and total_records >= sample_limit:
        break

print(f"\nProcessed data saved to: '{PROCESSED_FILE_PATH}'")
print(f"Unlabeled data saved to: '{UNLABELED_FILE_PATH}'")

# Load saved processed data for summary generation
df_processed = pd.read_csv(PROCESSED_FILE_PATH, low_memory=False)

# ==============================================================================
# 4. Print data summary
# ==============================================================================
print("\n" + "=" * 70)
print(f"DATASET SUMMARY REPORT (US - LAST {data_years} YEARS)")
print("=" * 70)
print(f"Raw archive:              {RAW_FILE_PATH}")
print(f"Processed target:         {PROCESSED_FILE_PATH}")
print(f"Unlabeled target:         {UNLABELED_FILE_PATH}")
print(f"Total Labeled Records:    {len(df_processed):,}")
print(f"Total Unlabeled Records:  {unlabeled_records:,}")
print(
    f"Cutoff Date:              >= {time.strftime('%Y-%m-%d', time.gmtime(cutoff_timestamp))} (UTC)"
)

print("\n--- 1. Nutri-Score Distribution (nutriscore_grade) ---")
if "nutriscore_grade" in df_processed.columns:
    nutri_counts = (
        df_processed["nutriscore_grade"]
        .str.lower()
        .str.strip()
        .value_counts(dropna=False)
    )
    nutri_pct = (
        df_processed["nutriscore_grade"]
        .str.lower()
        .str.strip()
        .value_counts(dropna=False, normalize=True)
        * 100
    )
    print(
        pd.DataFrame({"Count": nutri_counts, "Percentage (%)": nutri_pct.round(2)})
    )

print("\n--- 2. NOVA Group Distribution (nova_group: 1-4) ---")
if "nova_group" in df_processed.columns:
    nova_counts = df_processed["nova_group"].value_counts(dropna=False).sort_index()
    nova_pct = (
        df_processed["nova_group"].value_counts(dropna=False, normalize=True)
        * 100
    ).sort_index()
    print(
        pd.DataFrame({"Count": nova_counts, "Percentage (%)": nova_pct.round(2)})
    )

valid_nutri_mask = df_processed["nutriscore_grade"].notna() & (
    ~df_processed["nutriscore_grade"].str.lower().str.strip().isin(["unknown", "not-applicable", ""])
)
valid_nova_mask = df_processed["nova_group"].notna()
dual_label_count = (valid_nutri_mask & valid_nova_mask).sum()

print(
    f"\n--- 3. Overlap: Products with BOTH Labels ---: {dual_label_count:,} ({dual_label_count / len(df_processed):.1%})"
)

print("\n--- 4. Feature Coverage (%) ---")
coverage = (
    (df_processed.notna().mean() * 100).round(2).to_frame(name="Coverage (%)")
)
key_cols = [
    "product_name",
    "ingredients_text",
    "additives_n",
    "sugars_100g",
    "fat_100g",
    "sodium_100g",
    "energy_100g",
]
print(coverage.loc[[c for c in key_cols if c in coverage.index]])

print("\n--- 5. Key Nutritional Attributes Stats (per 100g) ---")
nutri_cols = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "proteins_100g",
    "sodium_100g",
]
print(
    df_processed[nutri_cols]
    .describe()
    .T[["count", "mean", "std", "50%", "max"]]
)
