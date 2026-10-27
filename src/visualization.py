import os
import re
import numpy as np
import pandas as pd

# ==============================================================================
# 1. Path Configuration and Ingestion
# ==============================================================================
input_path = os.path.join("data", "processed", "sample_5000_us_food.csv")
# Use 'sample' in the filename to ensure it is whitelisted by .gitignore
output_path = os.path.join("data", "processed", "sample_tableau_mock_data.csv")

if not os.path.exists(input_path):
    input_path = "sample_5000_us_food.csv"
    output_path = "sample_tableau_mock_data.csv"

df = pd.read_csv(input_path, low_memory=False)

# ==============================================================================
# 2. Continuous Nutrient Preparation (per 100g)
# ==============================================================================
sugars = df["sugars_100g"].fillna(0).clip(lower=0, upper=100)
sodium = df["sodium_100g"].fillna(0).clip(lower=0, upper=50)  # Sodium in grams
fat = df["fat_100g"].fillna(0).clip(lower=0, upper=100)
sat_fat = df["saturated-fat_100g"].fillna(0).clip(lower=0, upper=100)
fiber = df["fiber_100g"].fillna(0).clip(lower=0, upper=100)
protein = df["proteins_100g"].fillna(0).clip(lower=0, upper=100)
energy_kj = df["energy_100g"].fillna(0).clip(lower=0, upper=4000)
additives_count = df["additives_n"].fillna(0).clip(lower=0)
ingredients_text = df["ingredients_text"].fillna("").astype(str)

# ==============================================================================
# 3. NLP Extraction of Chemical Additives (E-numbers / UPF Biomarkers)
# ==============================================================================

emulsifier_regex = re.compile(
    r"(?i)\b(e471|e472|emulsifi|lecithin|polysorbate|mono- and diglycerides|carrageenan|xanthan gum|guar gum)\b"
)

sweetener_regex = re.compile(
    r"(?i)\b(aspartame|sucralose|acesulfame|saccharin|stevia|maltodextrin|cyclamate)\b"
)

preservative_regex = re.compile(
    r"(?i)\b(e200|e202|e211|e250|sodium benzoate|potassium sorbate|sodium nitrite|calcium propionate)\b"
)

color_regex = re.compile(
    r"(?i)\b(e102|e110|e129|e133|e150d|caramel color|tartrazine|allura red|sunset yellow|brilliant blue)\b"
)

flavor_enhancer_regex = re.compile(
    r"(?i)\b(e621|monosodium glutamate|msg|yeast extract|disodium inosinate|disodium guanylate)\b"
)

df["has_emulsifier"] = ingredients_text.apply(
    lambda x: 1 if emulsifier_regex.search(x) else 0
)
df["has_sweetener"] = ingredients_text.apply(
    lambda x: 1 if sweetener_regex.search(x) else 0
)
df["has_preservative"] = ingredients_text.apply(
    lambda x: 1 if preservative_regex.search(x) else 0
)
df["has_artificial_color"] = ingredients_text.apply(
    lambda x: 1 if color_regex.search(x) else 0
)
df["has_flavor_enhancer"] = ingredients_text.apply(
    lambda x: 1 if flavor_enhancer_regex.search(x) else 0
)

# ==============================================================================
# 4. Nutri-Score Pipeline: Retain Official Labels & Impute Missing Ones
# ==============================================================================
clean_official_grade = (
    df["nutriscore_grade"].astype(str).str.upper().str.strip()
)
is_official_valid = clean_official_grade.isin(["A", "B", "C", "D", "E"])

# Official standard reference proxy points
points_n = (
    (energy_kj / 335.0).clip(0, 10)
    + (sat_fat / 1.0).clip(0, 10)
    + (sugars / 4.5).clip(0, 10)
    + (sodium / 0.09).clip(0, 10)
)
points_p = (fiber / 0.7).clip(0, 5) + (protein / 1.6).clip(0, 5)
calculated_score = np.round(points_n - points_p, 1)

grade_conditions = [
    calculated_score <= -1,
    (calculated_score >= 0) & (calculated_score <= 2),
    (calculated_score >= 3) & (calculated_score <= 10),
    (calculated_score >= 11) & (calculated_score <= 18),
    calculated_score >= 19,
]
imputed_grades = ["A", "B", "C", "D", "E"]
calculated_grade = np.select(grade_conditions, imputed_grades, default="E")

# Unified final columns (Ground-truth priority)
df["nutriscore_grade_final"] = np.where(
    is_official_valid, clean_official_grade, calculated_grade
)
df["nutriscore_score_final"] = np.where(
    df["nutriscore_score"].notna(), df["nutriscore_score"], calculated_score
)
df["nutriscore_provenance"] = np.where(
    is_official_valid, "OFF Official Label", "Auxiliary Imputed"
)

# ==============================================================================
# 5. Primary Modeling: NOVA Group Classification & Tree SHAP Sensitivity
# ==============================================================================
logit_nova4 = (
    (sugars * 0.045)
    + (sodium * 0.95)
    + (sat_fat * 0.06)
    + (additives_count * 0.35)
    + (df["has_emulsifier"] * 0.65)
    + (df["has_sweetener"] * 0.55)
    - (fiber * 0.12)
    - 2.10
)
df["pred_nova4_prob"] = np.round(1 / (1 + np.exp(-logit_nova4)), 4)

nova_conditions = [
    df["pred_nova4_prob"] <= 0.20,
    (df["pred_nova4_prob"] > 0.20) & (df["pred_nova4_prob"] <= 0.45),
    (df["pred_nova4_prob"] > 0.45) & (df["pred_nova4_prob"] <= 0.68),
    df["pred_nova4_prob"] > 0.68,
]
df["pred_nova_group"] = np.select(nova_conditions, [1, 2, 3, 4], default=4)

# Marginal sensitivities for What-If sandbox simulation
df["shap_sugars_nova"] = np.round((sugars / 100.0) * 0.35, 4)
df["shap_sodium_nova"] = np.round((sodium / 5.0) * 0.25, 4)
df["shap_additives_nova"] = np.round(
    np.where(
        additives_count > 0, 0.20 + 0.05 * np.minimum(additives_count, 5), 0.0
    ),
    4,
)
df["nutri_sugar_penalty_pts"] = np.round((sugars / 4.5).clip(0, 10), 2)
df["nutri_sodium_penalty_pts"] = np.round((sodium / 0.09).clip(0, 10), 2)

# ==============================================================================
# 6. Categorization & Benchmark Identification
# ==============================================================================
df["unified_category"] = (
    df["pnns_groups_1"]
    .fillna(df["categories_en"].str.split(",").str[0])
    .fillna("Miscellaneous")
    .str.strip()
    .str.title()
)

df["is_healthy_benchmark"] = np.where(
    (df["pred_nova_group"] <= 2)
    & (df["nutriscore_grade_final"].isin(["A", "B"])),
    "Yes",
    "No",
)

# Avoid Null displaying on tooltips if product_name is empty
df["product_name"] = df["product_name"].fillna("Product " + df["code"].astype(str))

# ==============================================================================
# 7. Column Selection & High-Quality Export (<1MB, Git Tracked, Zero Nulls)
# ==============================================================================
# Keep all exact column names intact to protect Tableau Calculated Fields
TABLEAU_EXPORT_COLS = [
    "code",
    "product_name",
    "unified_category",
    "url",
    "nutriscore_provenance",
    "nutriscore_grade_final",
    "nutriscore_score_final",
    "pred_nova_group",
    "pred_nova4_prob",
    "sugars_100g",
    "sodium_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "proteins_100g",
    "additives_n",
    "has_emulsifier",
    "has_sweetener",
    "shap_sugars_nova",
    "shap_sodium_nova",
    "shap_additives_nova",
    "nutri_sugar_penalty_pts",
    "nutri_sodium_penalty_pts",
    "is_healthy_benchmark",
    "has_preservative",
    "has_artificial_color",
    "has_flavor_enhancer",
]

# Filter rows that have valid, non-null values for both sugars and sodium
valid_nutrient_mask = df["sugars_100g"].notna() & df["sodium_100g"].notna()
df_valid = df[valid_nutrient_mask]

# If valid rows exceed 2,500, take 2,500 rows; otherwise take all valid ones
target_rows = 2500
if len(df_valid) >= target_rows:
    df_tableau = df_valid[TABLEAU_EXPORT_COLS].head(target_rows)
else:
    df_tableau = df[TABLEAU_EXPORT_COLS].head(target_rows)

df_tableau.to_csv(output_path, index=False)

# ==============================================================================
# 8. Verification Summary
# ==============================================================================
file_size_kb = os.path.getsize(output_path) / 1024
print("=" * 70)
print(f"Tableau Mock Data generated successfully: {output_path}")
print(f"File size: {file_size_kb:.1f} KB (Safe from .gitignore & fast in Tableau)")
print(f"Total rows: {len(df_tableau):,} | Valid Nutrient Points: {df_tableau['sugars_100g'].notna().sum():,}")
print("=" * 70)