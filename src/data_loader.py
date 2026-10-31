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

# Filter: Fixed window from July 2024 to June 2026
# 2024-07-01 00:00:00 UTC = 1719792000
# 2026-06-30 23:59:59 UTC = 1782863999
START_TIMESTAMP = 1719792000
END_TIMESTAMP = 1782863999

PROCESSED_FILE_PATH = os.path.join(
    PROCESSED_DIR, "us_openfoodfacts_nutri_nova_202407_202606.csv"
)
UNLABELED_FILE_PATH = os.path.join(
    PROCESSED_DIR, "us_openfoodfacts_unlabeled_202407_202606.csv"
)
SAMPLE_FILE_PATH = os.path.join(
    PROCESSED_DIR, "sample_5000_us_food.csv"
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
    # Food Categories for Sandbox & Visual Filtering
    "categories_en",
    "pnns_groups_1",
    "pnns_groups_2",
    # Target Labels for Multi-target Modeling (Primary: NOVA; Secondary: Nutri-Score)
    "nova_group",
    "nutriscore_grade",
    "nutriscore_score",
    # Text & Additives Features (NLP extraction for UPF classification)
    "ingredients_text",
    "additives_n",
    "additives_tags",
    "additives_en",
    # Continuous Nutritional Attributes (per 100g for ML & What-If sliders)
    "energy_100g",
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "trans-fat_100g",
    "sugars_100g",
    "added-sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "sodium_100g",
]

us_label = re.compile(r"(?i)\b(?:united states|usa|us|america)\b")

# ==============================================================================
# 2. Raw data
# ==============================================================================
# Detect potential local copies to avoid re-downloading
if not os.path.exists(RAW_FILE_PATH):
    fallback_locations = [
        "openfoodfacts_dump.csv.gz",
        os.path.join("..", "CSE6242", "openfoodfacts_dump.csv.gz"),
    ]
    for loc in fallback_locations:
        if os.path.exists(loc):
            RAW_FILE_PATH = loc
            break

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
    f"\nFiltering US products modified between 2024-07 and 2026-06..."
)

numeric_fields = [
    "nutriscore_score",
    "nova_group",
    "additives_n",
    "energy_100g",
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "trans-fat_100g",
    "sugars_100g",
    "added-sugars_100g",
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

    # Filter 2: Modified between July 2024 and June 2026
    mod_ts = pd.to_numeric(chunk["last_modified_t"], errors="coerce").fillna(0)
    in_time_window = (mod_ts >= START_TIMESTAMP) & (mod_ts <= END_TIMESTAMP)

    base_filter = is_us & in_time_window
    if not base_filter.any():
        continue

    chunk_base = chunk[base_filter].copy()

    # Filter 3: Contains at least Nutri-Score or NOVA label
    valid_nutri = chunk_base["nutriscore_grade"].notna() & (
        ~chunk_base["nutriscore_grade"].str.lower().str.strip().isin(["unknown", "not-applicable", ""])
    )
    valid_nova = chunk_base["nova_group"].notna()
    has_labels = valid_nutri | valid_nova

    subset = chunk_base[has_labels]
    # Unlabeled pool: No target labels, but retains ingredients or nutrients for inference
    subset_unlabeled = chunk_base[
        (~has_labels)
        & (chunk_base["ingredients_text"].notna() | chunk_base["energy_100g"].notna())
    ]

    # Convert numeric fields
    for col in numeric_fields:
        if col in subset.columns:
            subset.loc[:, col] = pd.to_numeric(subset[col], errors="coerce")
        if col in subset_unlabeled.columns:
            subset_unlabeled.loc[:, col] = pd.to_numeric(subset_unlabeled[col], errors="coerce")

    # Stream write labeled processed dataset
    if not subset.empty:
        subset.to_csv(
            PROCESSED_FILE_PATH,
            mode="w" if first_chunk_proc else "a",
            header=first_chunk_proc,
            index=False,
        )
        first_chunk_proc = False
        total_records += len(subset)

    # Stream write unlabeled dataset
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

print(f"\nProcessed labeled data saved to: '{PROCESSED_FILE_PATH}'")
print(f"Unlabeled inference data saved to: '{UNLABELED_FILE_PATH}'")

# Generate small test sample for GitHub version control (first 5,000 records)
df_sample = pd.read_csv(PROCESSED_FILE_PATH, nrows=5000, low_memory=False)
df_sample.to_csv(SAMPLE_FILE_PATH, index=False)
print(f"Small repo sample saved to: '{SAMPLE_FILE_PATH}'")

# Load processed dataset for summary generation
df_processed = pd.read_csv(PROCESSED_FILE_PATH, low_memory=False)

# ==============================================================================
# 4. Print data summary
# ==============================================================================
print("\n" + "=" * 70)
print(f"DATASET SUMMARY REPORT (US: 2024-07 to 2026-06)")
print("=" * 70)
print(f"Raw archive:              {RAW_FILE_PATH}")
print(f"Processed target:         {PROCESSED_FILE_PATH}")
print(f"Unlabeled target:         {UNLABELED_FILE_PATH}")
print(f"Total Labeled Records:    {len(df_processed):,}")
print(f"Total Unlabeled Records:  {unlabeled_records:,}")
print(
    f"Time Window:              {time.strftime('%Y-%m-%d', time.gmtime(START_TIMESTAMP))} to {time.strftime('%Y-%m-%d', time.gmtime(END_TIMESTAMP))} (UTC)"
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
    c for c in [
        "energy_100g",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "trans-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "sodium_100g",
    ] if c in df_processed.columns
]


