# Project and Repository Overview

## Project

The project uses the Open Food Facts product database to explore how product
nutrients, ingredient descriptions, and additive evidence relate to NOVA
processing groups. It also produces cautious Nutri-Score completions where
official grades are missing and prepares an interactive, product-level Tableau
dataset.

The loader selects products marketed in the United States and modified between
2024-07-01 00:00:00 UTC and 2026-06-30 23:59:59 UTC. The latest local processed
snapshot contains 208,036 labeled-pool rows and 55,861 unlabeled-pool rows;
after exact deduplication and latest-record-per-code selection, the model
workflow contains 263,897 unique products. These counts describe the current
local data, not a permanent guarantee: the Open Food Facts dump can change.

## Repository map (file by file)

| File or path | Purpose |
|---|---|
| `README.md` | Entry point, Windows quick start, and links to detailed docs. |
| `requirements.txt` | Python runtime, ML, plotting, and test dependencies. scikit-learn 1.2+ is required by the pipeline API. |
| `setup.py` | Setuptools package metadata and dependencies loaded from `requirements.txt`. |
| `repo_setup.py` | Legacy repository scaffolding/template generator; it is not part of the current data or model run. |
| `.gitignore` | Excludes raw/full-size datasets, generated results, models, and Python caches; small `*sample*` files are whitelisted. |
| `config/config.yaml` | General project configuration template. The current loader/model scripts use constants in Python and do not read this YAML file. |
| `data/raw/.gitkeep` | Keeps the raw-data folder in Git. The compressed Open Food Facts dump is local and ignored by Git. |
| `data/raw/en.openfoodfacts.org.products.csv.gz` | Local source dump used by the loader (about 1.28 GB in the current snapshot); not tracked. |
| `data/processed/.gitkeep` | Keeps the processed-data folder in Git when local datasets are absent. |
| `data/processed/us_openfoodfacts_nutri_nova_202407_202606.csv` | Full processed pool with at least one valid NOVA or Nutri-Score label; not tracked. |
| `data/processed/us_openfoodfacts_unlabeled_202407_202606.csv` | Products with neither valid target label but with ingredient or nutrient evidence; not tracked. |
| `data/processed/sample_5000_us_food.csv` | Small, tracked sample for inspection and tests. Not used by the full model workflow. |
| `data/processed/sample_tableau_mock_data.csv` | Legacy Tableau mock export from the former heuristic visualization script. Do not use for current analysis/dashboard refresh. |
| `notebooks/.gitkeep` | Placeholder only; there are no current notebook workflow steps. |
| `src/__init__.py` | Package metadata; importing `src` does not run the pipeline. |
| `src/data_loader.py` | Downloads/reads the raw dump, filters the date/country window, and writes the two processed pools and sample. |
| `src/data_processing.py` | Small reusable CSV/Excel reader, exact-row deduplicator, and CSV writer. The main pipeline uses `clean_data` for exact deduplication. |
| `src/analysis.py` | Data-source merge, product-level deduplication, normalization, feature engineering, grouped splits, nutrient selection, PCA, and additive evidence extraction. |
| `src/modeling.py` | NOVA model comparison, Nutri-Score rule/model completion, ingredient sensitivity scenarios, metrics, and orchestration. |
| `src/visualization.py` | EDA/model diagnostic figures and the Tableau-compatible CSV export. |
| `src/utils.py` | Shared nutrient names and `results/tables` CSV writer. |
| `src/Project Dashboard.twb` | Tableau workbook. It is an XML workbook (`.twb`, not `.twd`) connected to `results/tables/tableau_products.csv`. |
| `tests/__init__.py` | Test package marker. |
| `tests/test_data_processing.py` | Tests basic duplicate removal and missing input errors. |
| `tests/test_analysis.py` | Tests merging/auditing the two sources and the sparse nutrient coverage threshold. |
| `tests/test_visualization.py` | Tests additive flags, Tableau output schema, long product codes, and test-label hiding. |
| `docs/PROJECT_OVERVIEW.md` | This file: repository map, data dictionary, methodology, run sequence, and output inventory. |
| `results/tables/`, `results/figures/`, `results/models/` | Generated pipeline outputs. They are local/ignored artifacts except for `.gitkeep`. |
| `results/tables/.gitkeep`, `results/figures/.gitkeep`, `results/models/.gitkeep` | Keep the three output directories in Git; generated results are ignored. |

## Run order

Run commands from the repository root in PowerShell.

1. **Set up Python** — see [README — Quick start](../README.md#quick-start-windows-powershell).
2. **Refresh source data, if needed**:

   ```powershell
   python -m src.data_loader
   ```

   This downloads the current public dump only when the expected raw archive is
   absent. It writes the labeled pool, unlabeled pool, and first-5,000-row sample
   into `data/processed/`. It can take substantial disk space and time.
3. **Run the complete modeling and export workflow**:

   ```powershell
   python -m src.modeling
   ```

   If both full processed CSVs already exist, this is the only required
   pipeline command. The sample files are not substitutes.
4. **Open the Tableau workbook**:

   ```text
   src/Project Dashboard.twb
   ```

   Open it in Tableau Desktop after step 3 has created
   `results/tables/tableau_products.csv`.
5. **Run regression tests**:

   ```powershell
   python -m pytest -q
   ```

## Tableau workbook status and portability

The workbook was updated to read the current 27-column Tableau export from
`results/tables/tableau_products.csv`; both workbook data sources use the same
schema. `code` is a string in the workbook so long Open Food Facts identifiers
are preserved. The local CSV has been generated and validated, so the workbook
can be opened directly in Tableau on this machine.

The `.twb` contains an absolute `directory` value for the current repository
location. If the repository is copied to a different folder or machine, Tableau
may prompt for a missing file. In Tableau, use **Data > [data source] > Edit
Connection** (or reconnect the data source) and select the new
`results/tables/tableau_products.csv`.

## Generated outputs

### Tables (`results/tables/`)

| Output | Contents |
|---|---|
| `tableau_products.csv` | Full combined product population and the 27 workbook fields. Main Tableau input. |
| `input_counts.csv` | Per-file loaded rows, exact-deduped rows, and column counts. |
| `nutrient_quality.csv` | Numeric parse, invalid-value, and usable-coverage audit. |
| `product_quality_flags.csv` | Product ID and nutrient inconsistency/review flags. |
| `split_assignments.csv` | Product ID, duplicate-proxy family, and target-specific split. |
| `family_group_sizes.csv` | Number of products per linked family group. |
| `nova_split_counts.csv`, `nutriscore_split_counts.csv` | Target class counts in inference/train/validation/test/quality-excluded groups. |
| `nova_feature_selection.csv`, `nutriscore_feature_selection.csv` | Training coverage, distinct values, and nutrient inclusion decisions. |
| `nova_validation_metrics.csv` | NOVA candidate-model validation metrics. |
| `nova_feature_comparison.csv` | Feature-set ablation comparison using a fixed model/split. |
| `feature_importance.csv` | Permutation macro-F1 drop and standard deviation on validation data. |
| `nutrient_correlations.csv` | Training-set pairwise nutrient correlations. |
| `pca_validation.csv`, `pca_variance.csv` | Validation PCA coordinates and explained variance. |
| `nutriscore_metrics.csv` | Nutri-Score classifier validation metrics by nutrient coverage. |
| `nutriscore_completion.csv`, `nutriscore_completion_counts.csv` | Product-level grade/source/rule result and source counts. |
| `scenario_eligibility.csv`, `ingredient_scenarios.csv` | Supported ingredient marker scenarios and probability changes. |
| `nova_final_test.csv` | Created only when `RUN_FINAL_TEST = True`; otherwise removed. |

### Figures (`results/figures/`)

- `nutrient_correlations.png`
- `nutrient_pca_validation.png`
- `nova_validation_confusion_matrix.png`
- `nova_validation_model_comparison.png`

### Models (`results/models/`)

- `nova_model.joblib` — selected NOVA pipeline.
- `nutriscore_model.joblib` — nutrient-based Nutri-Score completion pipeline.

Generated files are ignored by Git. Run the model workflow again to refresh
them; do not use `sample_tableau_mock_data.csv` for the current dashboard.

## Source data and row structure

The public source is the Open Food Facts tab-separated gzip dump:
`https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz`.
The loader selects products whose `countries_en` contains a US label and whose
`last_modified_t` is in the UTC interval 2024-07-01 through 2026-06-30.

| Dataset | Row meaning | Current rows |
|---|---|---:|
| `us_openfoodfacts_nutri_nova_202407_202606.csv` | Products with a valid NOVA (1–4) or Nutri-Score (A–E) label | 208,036 |
| `us_openfoodfacts_unlabeled_202407_202606.csv` | No valid target label, but ingredient or nutrient evidence exists | 55,861 |
| Combined model population | Exact-deduplicated inputs, then latest row per nonblank code | 263,897 |

These are counts in the current local snapshot, not guaranteed future counts.
Raw and full processed CSVs are ignored by Git. The tracked
`sample_5000_us_food.csv` is only a development sample.

Data grain and keys:

- Raw dump rows are source product records and may repeat product codes.
- Each processed CSV is a wide comma-separated table retaining source names.
- Modeling concatenates the two processed tables, adds `input_file`, removes
  exact duplicate rows within each source, and keeps the most recently modified
  record for each nonblank `code`.
- `product_id` is the unique product key. Missing source codes receive generated
  `__missing_code_<row>` identifiers.
- `nova_split` and `nutri_split` are separate target-specific assignments.
  Model matrices are imputed copies; source measurements remain unchanged.
- Tableau output has one row per deduplicated `product_id`, 27 ordered fields,
  and excludes modeling-only intermediate fields.
- Audit and score tables have row grain appropriate to their names (per input,
  feature, class/split, model, or product).

## Processed source fields (31 columns)

Nutrient measurements are per 100 g unless stated otherwise.

| Field | Type / units | Meaning |
|---|---|---|
| `code` | Text ID | Open Food Facts product code. Preserve as text; identifiers may exceed integer precision. |
| `product_name` | Text | Submitted product name. |
| `url` | URL text | Product page. |
| `created_t` | Unix seconds | Product creation timestamp. |
| `last_modified_t` | Unix seconds | Modification timestamp used for filtering and latest-row selection. |
| `countries_en` | Comma-separated text | Country labels searched for US membership. |
| `brands` | Text | Brand names; used to form duplicate-proxy groups. |
| `categories_en` | Comma-separated text | OFF categories; category fallback and scoring context. |
| `pnns_groups_1` | Category text | Broad PNNS category; preferred model/display category. |
| `pnns_groups_2` | Category text | Detailed PNNS category; helps resolve formula branch. |
| `ingredients_text` | Text | Ingredient list for normalization and marker detection. |
| `additives_n` | Count | Reported additive count; validated and used as a model feature. |
| `additives_tags` | Comma-separated tags | Additive identifiers such as `en:e955`. |
| `additives_en` | Text | Human-readable additive names; supports dashboard marker flags. |
| `nova_group` | Integer 1–4 | Observed NOVA target; other values are not valid targets. |
| `nutriscore_grade` | A–E/status | Observed grade; only A–E are valid classifier targets. |
| `nutriscore_score` | Numeric points | Source score; version is not assumed verified. |
| `energy_100g` | Numeric, normally kJ | Legacy energy field and formula fallback. |
| `energy-kj_100g` | kJ | Preferred energy input to formula completion. |
| `energy-kcal_100g` | kcal | Core model nutrient; can be converted from kJ. |
| `fat_100g` | g | Total fat. |
| `saturated-fat_100g` | g | Saturated fat. |
| `trans-fat_100g` | g | Optional predictor. |
| `carbohydrates_100g` | g | Used to check sugars consistency. |
| `sugars_100g` | g | Core predictor and formula input. |
| `added-sugars_100g` | g | Optional predictor. |
| `fiber_100g` | g | Core predictor and formula input. |
| `proteins_100g` | g | Core predictor and formula input. |
| `sodium_100g` | g | Core predictor; salt fallback is sodium × 2.5. |
| `salt_100g` | g | Preferred salt formula input. |
| `fruits-vegetables-legumes_100g` | Percent | Required FVL formula input. |

The loader also requests `fruits-vegetables-legumes-estimate-from-ingredients_100g`
and `fruits-vegetables-nuts-estimate-from-ingredients_100g` if available. They
are not substituted for the explicit FVL formula input.

## Engineered analysis fields

These fields are created in memory by `engineer_features`, not written back to
the processed input CSVs.

| Field | Meaning |
|---|---|
| `input_file` | Processed source filename for provenance. |
| `_modified` | Numeric `last_modified_t`, temporary sorting field. |
| `code_generated` | Source product code is missing or blank. |
| `product_id` | Unique source code or generated missing-code identifier. |
| `nova_target` | Valid NOVA class 1–4, otherwise missing. |
| `nutri_target` | Lowercase observed Nutri-Score A–E, otherwise missing. |
| `nutri_not_applicable` | Source grade explicitly indicates not applicable. |
| `ingredients_text_norm` | Unicode NFKC, lowercase and whitespace-normalized ingredient text. |
| `ingredient_present` | Ingredient text is present and not a placeholder. |
| `additives_n_feature` | Plausible additive count (integer 0–500), otherwise missing. |
| `ingredient_characters`, `ingredient_words` | Length and whitespace word count of ingredient text. |
| `marker_<name>` | Evidence feature for 14 additives/markers: sucralose, aspartame, acesulfame, saccharin, carrageenan, lecithin, polysorbate80, xanthan, guar, benzoate, sorbate, nitrite, MSG, and maltodextrin. Text and tags are combined. |
| `energy-kcal_100g` | Original kcal value, or kJ divided by 4.184 if absent. |
| `energy_kcal_converted_from_kj` | Whether kcal was derived from kJ. |
| `flag_satfat_gt_fat` | Saturated fat exceeds fat by more than 0.1 g/100 g. |
| `flag_sugars_gt_carbs` | Sugars exceed carbohydrates by more than 0.1 g/100 g. |
| `needs_review` | Either nutrient inconsistency flag; retained, but excluded from supervised splits. |
| `category` | Normalized PNNS group, else first OFF category, else `unknown`. Model category input. |
| `category_source` | `observed_pnns`, `observed_OFF_category`, or `unknown`. |
| `observed_core_nutrients` | Number of nonmissing core nutrient values, 0–7. |
| `family_group` | Transitive duplicate-proxy group from detailed ingredients and/or brand/product name. |
| `nova_split`, `nutri_split` | Target-specific `train`, `validation`, `test`, `inference`, or `excluded_quality`. |

Numeric values are parsed and invalid values are set missing, not winsorized:
negative/nonfinite measurements, energy above 4,200 kJ-like units, kcal above
1,000, or other nutrient/FVL percentages above 100.

## NOVA and Nutri-Score modeling

### Grouped splits and selected predictors

Duplicate proxies are linked using either the same normalized ingredient text
(at least 40 characters and six words) or matching known brand plus a product
name at least 12 characters and two words. Transitive groups stay together.
NOVA and Nutri-Score use independent `StratifiedGroupKFold` splits: five folds,
shuffle enabled, seed 33; fold 0 is test, fold 1 validation, folds 2–4 train.
Unlabeled products are `inference`; quality-flagged rows are
`excluded_quality`. Grouping reduces, but cannot eliminate, near-duplicate
leakage.

Core predictors in `src/utils.py`:

```text
energy-kcal_100g, fat_100g, saturated-fat_100g, sugars_100g,
fiber_100g, proteins_100g, sodium_100g
```

Candidate nutrient predictors also include `trans-fat_100g` and
`added-sugars_100g`. Selection uses training rows only: at least 25% nonmissing
coverage and five distinct values. The current snapshot has roughly 29–33%
coverage for core nutrients. Numeric columns are median-imputed with missing
indicators and standardized; categories use one-hot encoding
(`min_frequency=50`, `max_categories=40`, unknown values ignored).

### NOVA classifier candidates and metrics

| Candidate | Configuration |
|---|---|
| Majority baseline | Most frequent class. |
| Logistic regression | Balanced classes; up to 1,000 iterations. |
| Random forest | 120 trees, max depth 18, minimum leaf 5, balanced subsampling. |
| Histogram gradient boosting | 120 iterations, 15 leaves, learning rate 0.08, balanced classes. |

The selected non-baseline candidate maximizes validation macro-F1. Metrics
include accuracy, macro-F1 over fixed classes 1–4, NOVA-4 precision/recall/F1,
average precision, and Brier score. Further analyses use fixed-split feature
ablations, permutation importance (five validation repeats), and nutrient PCA.
PCA preprocessing is fitted on training nutrients and then applied to
validation rows.

### Nutri-Score completion

For each product, resolution order is:

1. retain an observed A–E grade (`OFF_observed`);
2. calculate a supported 2023-style formula grade (`formula_calculated`);
3. if still unknown and at least four of seven core nutrients are present, use
   the five-class logistic regression (`model_estimated`);
4. otherwise keep `unknown`, `not_applicable`, or `needs_review`.

Observed grade versions are recorded as `OFF_version_unverified`. The model's
maximum class probability is uncalibrated, not a calibrated confidence score.
The classifier trains on labeled supported products with at least four
observed core nutrients and evaluates on validation; metrics include accuracy,
macro-F1, and ordinal MAE.

#### Formula branch and inputs

Formula scoring supports explicitly allowlisted general foods, cheese, and
plain natural/spring water. Water is assigned A. The general-food allowlist is
bread, breakfast cereals, cereals, legumes, pasta, rice, potatoes, vegetables,
fruits, soups, biscuits and cakes, pastries, chocolate products, sweets, ice
cream, sandwiches, pizza pies and quiche, one-dish meals, salty and fatty
products, fish and seafood, eggs, and dairy desserts. Baby foods, baby milks,
infant formula, and alcoholic beverages are excluded when marked not
applicable. Unknown categories are not guessed.

Required formula measurements are energy in kJ, sugars, saturated fat, salt,
fiber, protein, and fruit/vegetable/legume proportion. Energy fallback:
`energy-kj_100g`, then `energy_100g`, then `energy-kcal_100g × 4.184`. Salt
fallback: `salt_100g`, then `sodium_100g × 2.5`. Ingredient-estimated FVL is
not substituted.

For supported general food and cheese, negative points are:

```text
N = energy_points + sugar_points + saturated_fat_points + salt_points
```

Each negative point is one strictly crossed threshold:

- energy: 335, 670, ..., 3,350 kJ/100 g;
- sugars: 3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51 g/100 g;
- saturated fat: 1, 2, ..., 10 g/100 g;
- salt: 0.2, 0.4, ..., 4.0 g/100 g.

Positive points are:

- fiber: thresholds 3, 4.1, 5.2, 6.3, 7.4 g/100 g;
- FVL: thresholds 40%, 60%, 80%, 80%, 80%;
- protein: thresholds 2.4, 4.8, 7.2, 9.6, 12, 14, 17 g/100 g.

Protein points count for cheese, or for general food if `N < 11`.

```text
score = N - fiber_points - FVL_points - applicable_protein_points
```

Strict grade boundaries: score ≤0 → A, >0 → B, >2 → C, >10 → D, >18 → E.
This is a partial supported-branch implementation, not a claim that every
product is graded under all official rules.

### Ingredient sensitivity scenarios

For validation NOVA-4 products, removal scenarios are evaluated for sucralose,
carrageenan, and lecithin only when both ingredient text and tags support the
marker and reported additive count matches parsed tags. Nutrients are held
constant. Results are model-input sensitivity analyses, not confirmed recipe
changes or causal health effects.

## Tableau export data dictionary (27 ordered columns)

The workbook/export contract is `results/tables/tableau_products.csv`. Keep the
column names and order in sync with `TABLEAU_EXPORT_COLUMNS` in
`src/visualization.py` and the workbook.

| Field | Type | Meaning |
|---|---|---|
| `code` | Text | Exact source product code, never converted to number. |
| `product_name` | Text | Source name, or `Product <product_id>` if blank. |
| `unified_category` | Text | Title-cased PNNS/category fallback, or `Miscellaneous`. |
| `url` | Text | Product page URL. |
| `nutriscore_provenance` | Text | Grade source; `hidden_final_test` marks suppressed test grade. |
| `nutriscore_grade_final` | A–E/null | Selected grade display. |
| `nutriscore_score_final` | Numeric/null | Source score for observed grade; formula score for formula grade; otherwise null. |
| `pred_nova_group` | Integer 1–4 | Selected classifier prediction for every product. |
| `pred_nova4_prob` | Numeric 0–1 | Predicted probability of NOVA 4. |
| `sugars_100g`, `sodium_100g`, `saturated-fat_100g`, `fiber_100g`, `proteins_100g` | Numeric g/100 g | Original measurement; missing values remain missing. |
| `additives_n` | Numeric/null | Validated additive count used by model. |
| `has_emulsifier`, `has_sweetener`, `has_preservative`, `has_artificial_color`, `has_flavor_enhancer` | 0/1/null | Evidence-based ingredient/tag/additive-name flags. Null means evidence is insufficient. |
| `shap_sugars_nova`, `shap_sodium_nova`, `shap_additives_nova` | Numeric/null | Legacy-compatible names for local NOVA-4 probability changes when a feature signal is set to zero. Not SHAP values. |
| `nutri_sugar_penalty_pts` | Numeric/null | Dashboard proxy: `clip(sugars_100g / 4.5, 0, 10)`. |
| `nutri_sodium_penalty_pts` | Numeric/null | Dashboard proxy: `clip(sodium_100g / 0.09, 0, 10)`. |
| `is_healthy_benchmark` | `Yes`/`No` | `Yes` if predicted NOVA ≤2 and displayed Nutri-Score is A or B. |
| `has_other_additive` | 0/1/null | Positive count with none of the named categories detected; null if count missing. |

The Tableau data contains the full labeled + unlabeled deduplicated population,
not the old 2,500-row mock. `RUN_FINAL_TEST = False` suppresses test-split
Nutri-Score grades and scores, but does not suppress model predictions.
`shap_*_nova` are probability deltas and `nutri_*_penalty_pts` are dashboard
proxies; neither should be described as official SHAP/official full-formula
scores.

## Installation, execution, and Tableau

Install from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 is recommended; the model code uses scikit-learn 1.2+.

Run in this order:

```powershell
python -m src.data_loader   # optional when both full processed CSVs exist
python -m src.modeling
python -m pytest -q
```

The loader downloads the public gzip dump only if absent, then writes the
labeled/unlabeled files and 5,000-row sample. The model command requires both
full input files; it never substitutes the sample. Open
`src/Project Dashboard.twb` in Tableau after the model command has generated
`results/tables/tableau_products.csv`.

The workbook is a `.twb` XML workbook, not `.twd`. It was updated so both
connections use the 27-column full Tableau export and `code` is text to
preserve long identifiers. Its connection currently stores an absolute path
for this workspace. If moved, use Tableau's **Data > [data source] > Edit
Connection** and point to the new `results/tables/tableau_products.csv`.

If processed input files are missing, rerun the loader. If Python dependencies
are missing, activate the project virtual environment and reinstall
`requirements.txt`. Generated CSVs/models/figures are ignored by Git; rerun the
model command to regenerate them.
