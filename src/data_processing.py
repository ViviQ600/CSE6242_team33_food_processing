"""Target validation, nutrient cleaning, ingredient features, and grouped splits.

All preparation runs inside explicit functions. Missing nutrients remain missing;
feature availability is screened on the NOVA training cohort only.
"""

from .visualization import plot_target_distributions, plot_nutrient_coverage
import json
import unicodedata
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import StratifiedGroupKFold
from .utils import WorkflowState

SEED = 33
unknown_tokens = {"", "unknown", "not-applicable", "not applicable", "none", "null", "nan"}
MARKERS = {
    "sucralose": ("\\b(?:sucralose|e\\s*-?\\s*955)\\b", {"e955"}),
    "aspartame": ("\\b(?:aspartame|e\\s*-?\\s*951)\\b", {"e951"}),
    "acesulfame": ("\\b(?:acesulfame(?:[-\\s]+(?:k|potassium))?|e\\s*-?\\s*950)\\b", {"e950"}),
    "saccharin": ("\\b(?:saccharin|e\\s*-?\\s*954)\\b", {"e954"}),
    "carrageenan": ("\\b(?:carrageenan|e\\s*-?\\s*407)\\b", {"e407"}),
    "lecithin": (
        "\\b(?:(?:soy|soya|sunflower)[-\\s]+)?lecithins?\\b|\\be\\s*-?\\s*322\\b",
        {"e322"},
    ),
    "polysorbate80": ("\\b(?:polysorbate[-\\s]*80|e\\s*-?\\s*433)\\b", {"e433"}),
    "xanthan": ("\\b(?:xanthan(?:[-\\s]+gum)?|e\\s*-?\\s*415)\\b", {"e415"}),
    "guar": ("\\b(?:guar[-\\s]+gum|e\\s*-?\\s*412)\\b", {"e412"}),
    "benzoate": ("\\b(?:sodium[-\\s]+benzoate|e\\s*-?\\s*211)\\b", {"e211"}),
    "sorbate": ("\\b(?:potassium[-\\s]+sorbate|e\\s*-?\\s*202)\\b", {"e202"}),
    "nitrite": ("\\b(?:sodium[-\\s]+nitrite|e\\s*-?\\s*250)\\b", {"e250"}),
    "msg": ("\\b(?:monosodium[-\\s]+glutamate|e\\s*-?\\s*621)\\b", {"e621"}),
    "maltodextrin": ("\\bmaltodextrin\\b", set()),
}


def norm_text(series):
    return (
        series.fillna("")
        .astype(str)
        .map(lambda s: unicodedata.normalize("NFKC", s))
        .str.lower()
        .str.strip()
        .str.replace("\\s+", " ", regex=True)
    )


def col(frame, name):
    return frame[name] if name in frame else pd.Series(pd.NA, index=frame.index, dtype="string")


def parse_tags(value):
    if pd.isna(value) or str(value).strip().lower() in unknown_tokens:
        return set()
    return {x.strip().lower().removeprefix("en:") for x in str(value).split(",") if x.strip()}


def ingredient_features(frame):
    text = norm_text(col(frame, "ingredients_text_norm"))
    tags = col(frame, "additives_tags").map(parse_tags)
    present = ~text.isin(unknown_tokens)
    counts = pd.to_numeric(col(frame, "additives_n"), errors="coerce").astype(float)
    counts = counts.where(np.isfinite(counts) & counts.ge(0) & counts.le(500))
    # With no ingredient evidence, a missing marker remains unknown rather than zero.
    known = present | counts.notna() | tags.map(bool)
    out = pd.DataFrame(index=frame.index)
    out["additives_n_feature"] = counts
    out["ingredient_characters"] = text.str.len().astype(float).where(present)
    out["ingredient_words"] = text.str.split().str.len().astype(float).where(present)
    for name, (pattern, codes) in MARKERS.items():
        flag = text.str.contains(pattern, regex=True, na=False) | tags.map(
            lambda t: bool(t & codes)
        )
        out["marker_" + name] = flag.astype(float).where(known)
    return out


def labeled_group_split(frame, target, seed=33):
    labels = frame[target].notna()
    result = pd.Series("inference", index=frame.index, dtype="string")
    labeled = frame.loc[labels]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, positions) in enumerate(
        cv.split(np.zeros(len(labeled)), labeled[target].astype(str), labeled.family_group)
    ):
        result.loc[labeled.index[positions]] = (
            "test" if fold == 0 else "validation" if fold == 1 else "train"
        )
    assert (
        frame.loc[labels]
        .assign(_split=result[labels])
        .groupby("family_group")
        ._split.nunique()
        .max()
        == 1
    )
    return result


def audit_targets(state: WorkflowState) -> WorkflowState:
    """Validate observed labels and report target availability.

    Required artifacts: df_raw, save_table, show_figure.
    Published artifacts: flow, grade_dist, nova_dist, unknown_tokens.
    """
    df_raw = state.require("df_raw")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    raw_nova = pd.to_numeric(col(df_raw, "nova_group"), errors="coerce")
    raw_grade = norm_text(col(df_raw, "nutriscore_grade"))
    valid_nova = raw_nova.isin([1, 2, 3, 4])
    valid_grade = raw_grade.isin(list("abcde"))
    raw_ingredients = norm_text(col(df_raw, "ingredients_text"))
    unknown_tokens = {"", "unknown", "not-applicable", "not applicable", "none", "null", "nan"}
    ingredients_present = ~raw_ingredients.isin(unknown_tokens)
    old_inference_rule = (
        ~valid_nova
        & ~valid_grade
        & (col(df_raw, "ingredients_text").notna() | col(df_raw, "energy_100g").notna())
    )
    flow = pd.DataFrame(
        {
            "cohort": [
                "All US/window records",
                "Valid NOVA",
                "Valid Nutri-Score A-E",
                "Both valid",
                "At least one valid",
                "Neither valid",
                "Neither + original inference-input filter",
            ],
            "rows": [
                len(df_raw),
                int(valid_nova.sum()),
                int(valid_grade.sum()),
                int((valid_nova & valid_grade).sum()),
                int((valid_nova | valid_grade).sum()),
                int((~valid_nova & ~valid_grade).sum()),
                int(old_inference_rule.sum()),
            ],
        }
    )
    display(save_table(flow, "01_label_flow.csv"))
    nova_dist = (
        raw_nova[valid_nova]
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("NOVA")
        .reset_index(name="rows")
    )
    grade_dist = (
        raw_grade[valid_grade]
        .value_counts()
        .reindex(list("abcde"), fill_value=0)
        .rename_axis("grade")
        .reset_index(name="rows")
    )
    display(save_table(nova_dist, "02_nova_distribution.csv"))
    display(save_table(grade_dist, "03_nutriscore_distribution.csv"))
    plot_target_distributions(nova_dist, grade_dist, show_figure)
    state.update(
        {
            "flow": flow,
            "grade_dist": grade_dist,
            "nova_dist": nova_dist,
            "unknown_tokens": unknown_tokens,
        }
    )
    return state


def clean_data(state: WorkflowState) -> WorkflowState:
    """Deduplicate barcodes, clean nutrients, and record quality flags.

    Required artifacts: df_raw, save_table, show_figure.
    Published artifacts: NUTRIENTS, coverage, df.
    """
    df_raw = state.require("df_raw")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    df = df_raw.copy()
    df["code"] = col(df, "code").astype("string").str.strip()
    rows_before = len(df)
    df["_modified_numeric"] = pd.to_numeric(df["last_modified_t"], errors="coerce")
    known_code = df["code"].notna() & df["code"].ne("")
    kept_known = (
        df.loc[known_code].sort_values("_modified_numeric").drop_duplicates("code", keep="last")
    )
    df = pd.concat([kept_known, df.loc[~known_code]], ignore_index=True)
    print("Duplicate-barcode rows removed:", rows_before - len(df))
    # Accept only observed NOVA groups 1-4 as training or evaluation targets.
    df["nova_target"] = pd.to_numeric(df["nova_group"], errors="coerce").where(
        lambda s: s.isin([1, 2, 3, 4])
    )
    df["nutri_target"] = norm_text(df["nutriscore_grade"]).where(lambda s: s.isin(list("abcde")))
    df["nutri_status_original"] = norm_text(df["nutriscore_grade"])
    df["ingredients_text_norm"] = norm_text(df["ingredients_text"])
    df["ingredient_present"] = ~df["ingredients_text_norm"].isin(unknown_tokens)
    NUTRIENTS = [
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "sodium_100g",
    ]
    quality_rows = []
    for field in NUTRIENTS + ["energy_100g", "carbohydrates_100g", "salt_100g"]:
        original = pd.to_numeric(col(df, field), errors="coerce").astype(float)
        upper = 1000 if field == "energy-kcal_100g" else 4200 if field == "energy_100g" else 100
        invalid = ~np.isfinite(original) | original.lt(0) | original.gt(upper)
        quality_rows.append(
            {
                "field": field,
                "original_numeric_count": int(original.notna().sum()),
                "out_of_range_count": int((invalid & original.notna()).sum()),
                "negative_count": int(original.lt(0).sum()),
                "upper_bound": upper,
            }
        )
        df[field] = original.mask(invalid)
    # Convert units only to fill missing kcal; never replace an observed kcal value.
    df["energy_from_kj"] = df["energy-kcal_100g"].isna() & df["energy_100g"].notna()
    df["energy-kcal_100g"] = df["energy-kcal_100g"].fillna(df["energy_100g"] / 4.184)
    df["flag_satfat_gt_fat"] = (df["saturated-fat_100g"] > df["fat_100g"] + 0.1).fillna(False)
    df["flag_sugars_gt_carbs"] = (df["sugars_100g"] > df["carbohydrates_100g"] + 0.1).fillna(False)
    df["flag_salt_sodium_mismatch"] = (
        df["salt_100g"].notna()
        & df["sodium_100g"].notna()
        & (
            (df["salt_100g"] - 2.5 * df["sodium_100g"]).abs()
            > np.maximum(0.1, 0.2 * df["salt_100g"])
        )
    )
    coverage = pd.DataFrame(
        {
            "field": NUTRIENTS,
            "usable_count": [int(df[c].notna().sum()) for c in NUTRIENTS],
            "usable_pct": [round(df[c].notna().mean() * 100, 2) for c in NUTRIENTS],
        }
    )
    df["observed_nutrients"] = df[NUTRIENTS].notna().sum(axis=1)
    display(save_table(pd.DataFrame(quality_rows), "04_nutrient_range_audit.csv"))
    display(save_table(coverage, "05_nutrient_coverage.csv"))
    display(
        df[["flag_satfat_gt_fat", "flag_sugars_gt_carbs", "flag_salt_sodium_mismatch"]]
        .sum()
        .to_frame("flagged_rows")
    )
    plot_nutrient_coverage(coverage, show_figure)
    state.update({"NUTRIENTS": NUTRIENTS, "coverage": coverage, "df": df})
    return state


def build_ingredient_features(state: WorkflowState) -> WorkflowState:
    """Attach observed categories and interpretable ingredient indicators.

    Required artifacts: df, save_table.
    Published artifacts: INGREDIENT_FEATURES.
    """
    df = state.require("df")
    save_table = state.require("save_table")
    pnns = norm_text(col(df, "pnns_groups_1"))
    category_fallback = norm_text(col(df, "categories_en")).str.split(",").str[0].str.strip()
    pnns_known = ~pnns.isin(unknown_tokens)
    fallback_known = ~category_fallback.isin(unknown_tokens)
    df["category"] = pnns.where(pnns_known, category_fallback.where(fallback_known, "unknown"))
    df["category_source"] = np.where(
        pnns_known, "observed_pnns", np.where(fallback_known, "observed_off_category", "unknown")
    )
    features = ingredient_features(df)
    INGREDIENT_FEATURES = list(features.columns)
    for c in INGREDIENT_FEATURES:
        df[c] = features[c]
    display(
        save_table(
            df.category_source.value_counts().rename_axis("source").reset_index(name="rows"),
            "06_category_sources.csv",
        )
    )
    marker_counts = pd.DataFrame(
        {
            "feature": INGREDIENT_FEATURES[3:],
            "products_flagged": [int(df[c].eq(1).sum()) for c in INGREDIENT_FEATURES[3:]],
        }
    )
    display(save_table(marker_counts, "07_marker_counts.csv"))
    state.update({"INGREDIENT_FEATURES": INGREDIENT_FEATURES})
    return state


def create_grouped_splits(state: WorkflowState) -> WorkflowState:
    """Build target-free product families and split observed NOVA labels only.

    Required artifacts: SEED, df, save_table.
    Published artifacts: main, train, val.
    """
    SEED = state.require("SEED")
    df = state.require("df")
    save_table = state.require("save_table")
    parent = np.arange(len(df))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join_keys(keys):
        seen = {}
        for i, key in enumerate(keys):
            if not key:
                continue
            if key in seen:
                a, b = (find(i), find(seen[key]))
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i

    brand = norm_text(col(df, "brands"))
    name = (
        norm_text(col(df, "product_name"))
        .str.replace("[^\\w\\s]", " ", regex=True)
        .str.replace("\\s+", " ", regex=True)
        .str.strip()
    )
    recipe = (
        df.ingredients_text_norm.str.replace("[^\\w\\s]", " ", regex=True)
        .str.replace("\\s+", " ", regex=True)
        .str.strip()
    )
    join_keys([f"{b}|recipe|{t}" if len(t) >= 12 else "" for b, t in zip(brand, recipe)])
    join_keys(
        [
            f"{b}|name|{t}" if len(t) >= 5 and t not in unknown_tokens else ""
            for b, t in zip(brand, name)
        ]
    )
    df["family_group"] = [f"g{find(i)}" for i in range(len(df))]
    # Unlabeled products remain in the inference pool, outside all labeled splits.
    df["split"] = labeled_group_split(df, "nova_target", seed=SEED)
    main = df.loc[df.nova_target.notna()].copy()
    train = main.loc[main.split.eq("train")].copy()
    val = main.loc[main.split.eq("validation")].copy()
    display(
        save_table(
            pd.crosstab(main.split, main.nova_target).reset_index(),
            "08_nova_labeled_split_counts.csv",
        )
    )
    save_table(df[["code", "family_group", "split"]], "09_split_assignments.csv")
    print(
        "Labeled NOVA cohort:",
        len(main),
        "train:",
        len(train),
        "validation:",
        len(val),
        "test:",
        int(main.split.eq("test").sum()),
    )
    print("Separate missing-NOVA application pool:", int(df.nova_target.isna().sum()))
    assert set(train.nova_target.astype(int)) == {1, 2, 3, 4}
    state.update({"main": main, "train": train, "val": val})
    return state


def select_features(state: WorkflowState) -> WorkflowState:
    """Audit every raw field and select nutrients from training coverage.

    Required artifacts: INGREDIENT_FEATURES, NUTRIENTS, OUTPUT_DIR, df, df_raw, main, save_table, train, val.
    Published artifacts: CORE_NUTRIENTS, NUTRIENTS, feature_profile, main, train, val, valid_numeric.
    """
    INGREDIENT_FEATURES = state.require("INGREDIENT_FEATURES")
    NUTRIENTS = state.require("NUTRIENTS")
    OUTPUT_DIR = state.require("OUTPUT_DIR")
    df = state.require("df")
    df_raw = state.require("df_raw")
    main = state.require("main")
    save_table = state.require("save_table")
    train = state.require("train")
    val = state.require("val")
    CORE_NUTRIENTS = list(NUTRIENTS)
    NUTRIENT_CANDIDATES = CORE_NUTRIENTS + [
        "carbohydrates_100g",
        "trans-fat_100g",
        "added-sugars_100g",
        "cholesterol_100g",
        "calcium_100g",
        "iron_100g",
        "potassium_100g",
        "vitamin-a_100g",
        "vitamin-c_100g",
        "vitamin-d_100g",
    ]
    # Availability screening uses only NOVA training rows, before model selection.
    MIN_TRAIN_COVERAGE = 0.6
    valid_numeric = {}
    for field in [c for c in df_raw.columns if c.endswith("_100g")]:
        x = pd.to_numeric(col(df, field), errors="coerce").astype(float)
        upper = (
            4200
            if field in ["energy_100g", "energy-kj_100g", "energy-from-fat_100g"]
            else 1000 if field == "energy-kcal_100g" else 100
        )
        valid_numeric[field] = x.where(np.isfinite(x) & x.ge(0) & x.le(upper))
    valid_numeric["energy-kcal_100g"] = df["energy-kcal_100g"]
    selected_nutrients = []
    profile = []
    for field in df_raw.columns:
        raw = col(df, field)
        meaningful = ~norm_text(raw).isin(unknown_tokens)
        training_raw = raw.loc[train.index]
        training_meaningful = meaningful.loc[train.index]
        role = "exclude_other"
        decision = "exclude"
        reason = "Outside ingredient/nutrition research inputs"
        numeric_count = None
        numeric_coverage = None
        unique_train = None
        if field in valid_numeric:
            x = valid_numeric[field].loc[train.index]
            numeric_count = int(x.notna().sum())
            numeric_coverage = float(x.notna().mean())
            unique_train = int(x.nunique(dropna=True))
        if field in ["nova_group", "nutriscore_grade"]:
            role = "target"
            reason = "Task target; never an input"
        elif field == "nutriscore_score" or field.startswith(
            ("environmental_score", "nutrient_levels")
        ):
            role = "exclude_score"
            reason = "Derived rating; excluded from independent feature comparisons"
        elif field in [
            "code",
            "url",
            "product_name",
            "generic_name",
            "abbreviated_product_name",
            "brands",
            "brands_tags",
            "brands_en",
            "brand_owner",
        ]:
            role = "identifier_grouping_display"
            reason = "Identity/grouping/display; not a main predictor"
        elif field in [
            "created_t",
            "last_modified_t",
            "last_updated_t",
            "countries_tags",
            "countries_en",
            "countries",
        ]:
            role = "filter_provenance"
            reason = "Cohort filtering or provenance only"
        elif field in ["ingredients_text", "additives_tags", "additives_n"]:
            role = "ingredient_input"
            decision = "retain_for_features"
            reason = "Ingredient representation and interpretable markers"
        elif field in ["ingredients_tags", "additives_en", "additives"]:
            role = "ingredient_reference"
            reason = (
                "Retained for inspection; not duplicated as another encoding in initial main model"
            )
        elif field in ["pnns_groups_1", "pnns_groups_2", "categories_tags", "categories_en"]:
            role = "category_or_rule_context"
            decision = "retain_for_context"
            reason = "Category covariate / scoring branch resolution; no unsupported USDA crosswalk"
        elif field in [
            "fruits-vegetables-legumes_100g",
            "salt_100g",
            "energy_100g",
            "energy-kj_100g",
        ]:
            role = "rule_or_unit_conversion"
            decision = "retain_for_rules"
            reason = "Nutri-Score input or redundant unit conversion, not duplicate main feature"
        elif field in NUTRIENT_CANDIDATES:
            role = "nutrient_candidate"
            if (
                numeric_coverage is not None
                and numeric_coverage >= MIN_TRAIN_COVERAGE
                and (unique_train >= 5)
            ):
                decision = "select"
                reason = "Domain-relevant, training numeric coverage >=60%, >=5 values"
                selected_nutrients.append(field)
            else:
                reason = "Training coverage <60% or insufficient variation; reserve for sensitivity analysis"
        elif field.endswith("_100g"):
            role = "reserve_nutrient_or_measure"
            reason = (
                "Not in initial domain candidate set; inspect coverage and units before expansion"
            )
        profile.append(
            {
                "field": field,
                "role": role,
                "decision": decision,
                "reason": reason,
                "overall_meaningful_pct": round(meaningful.mean() * 100, 2),
                "training_meaningful_pct": round(training_meaningful.mean() * 100, 2),
                "training_valid_numeric_count": numeric_count,
                "training_valid_numeric_pct": (
                    round(numeric_coverage * 100, 2) if numeric_coverage is not None else None
                ),
                "training_unique_numeric_values": unique_train,
            }
        )
    feature_profile = pd.DataFrame(profile)
    save_table(feature_profile, "33_all_211_feature_decisions.csv")
    for c in selected_nutrients:
        df[c] = valid_numeric[c]
    NUTRIENTS = selected_nutrients
    main = df.loc[df.nova_target.notna()].copy()
    train = main.loc[main.split.eq("train")].copy()
    val = main.loc[main.split.eq("validation")].copy()
    print("Selected nutritional inputs:", NUTRIENTS)
    display(
        feature_profile.loc[
            feature_profile.role.eq("nutrient_candidate"),
            ["field", "training_valid_numeric_pct", "decision", "reason"],
        ]
    )
    assert len(feature_profile) == 211
    assert len(NUTRIENTS) >= 4
    (OUTPUT_DIR / "feature_selection.json").write_text(
        json.dumps(
            {
                "criterion": "training coverage >=60% and >=5 distinct valid values among domain candidates",
                "nutrients": NUTRIENTS,
                "ingredient_features": INGREDIENT_FEATURES,
                "category": "category",
                "train_rows": len(train),
                "selection_scope": "initial evidence-based screen, not an optimal subset search",
            },
            indent=2,
        )
    )
    state.update(
        {
            "CORE_NUTRIENTS": CORE_NUTRIENTS,
            "NUTRIENTS": NUTRIENTS,
            "feature_profile": feature_profile,
            "main": main,
            "train": train,
            "val": val,
            "valid_numeric": valid_numeric,
        }
    )
    return state
