"""Data checks, explicit ingredient features, grouped splits, and nutrient EDA."""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .utils import CORE_NUTRIENTS, save_table
from .visualization import plot_eda

NUTRIENT_CANDIDATES = CORE_NUTRIENTS + ["trans-fat_100g", "added-sugars_100g"]
MARKERS = {
    "sucralose": (r"\b(?:sucralose|e\s*-?\s*955)\b", {"e955"}),
    "aspartame": (r"\b(?:aspartame|e\s*-?\s*951)\b", {"e951"}),
    "acesulfame": (
        r"\b(?:acesulfame(?:[-\s]+(?:k|potassium))?|e\s*-?\s*950)\b",
        {"e950"},
    ),
    "saccharin": (r"\b(?:saccharin|e\s*-?\s*954)\b", {"e954"}),
    "carrageenan": (r"\b(?:carrageenan|e\s*-?\s*407)\b", {"e407"}),
    "lecithin": (
        r"\b(?:(?:soy|soya|sunflower)[-\s]+)?lecithins?\b|\be\s*-?\s*322\b",
        {"e322"},
    ),
    "polysorbate80": (r"\b(?:polysorbate[-\s]*80|e\s*-?\s*433)\b", {"e433"}),
    "xanthan": (r"\b(?:xanthan(?:[-\s]+gum)?|e\s*-?\s*415)\b", {"e415"}),
    "guar": (r"\b(?:guar[-\s]+gum|e\s*-?\s*412)\b", {"e412"}),
    "benzoate": (r"\b(?:sodium[-\s]+benzoate|e\s*-?\s*211)\b", {"e211"}),
    "sorbate": (r"\b(?:potassium[-\s]+sorbate|e\s*-?\s*202)\b", {"e202"}),
    "nitrite": (r"\b(?:sodium[-\s]+nitrite|e\s*-?\s*250)\b", {"e250"}),
    "msg": (r"\b(?:monosodium[-\s]+glutamate|e\s*-?\s*621)\b", {"e621"}),
    "maltodextrin": (r"\bmaltodextrin\b", set()),
}
TABLEAU_ADDITIVE_MARKERS = {
    "has_emulsifier": (
        r"\b(?:e\s*-?\s*(?:471|472|322|433|407|415|412)|emulsifi\w*|lecithins?|"
        r"polysorbates?|mono[-\s]+and[-\s]+diglycerides|carrageenan|"
        r"xanthan(?:[-\s]+gum)?|guar[-\s]+gum)\b",
        {"e471", "e472", "e322", "e433", "e407", "e415", "e412"},
    ),
    "has_sweetener": (
        r"\b(?:e\s*-?\s*(?:950|951|954|955|960|961|962|952)|aspartame|"
        r"sucralose|acesulfame|saccharin|stevia|steviol glycosides|"
        r"cyclamate|maltodextrin)\b",
        {"e950", "e951", "e954", "e955", "e960", "e961", "e962", "e952"},
    ),
    "has_preservative": (
        r"\b(?:e\s*-?\s*(?:200|202|211|250|282)|sodium benzoate|"
        r"potassium sorbate|sodium nitrite|calcium propionate)\b",
        {"e200", "e202", "e211", "e250", "e282"},
    ),
    "has_artificial_color": (
        r"\b(?:e\s*-?\s*(?:102|110|129|133|150d)|caramel color|"
        r"tartrazine|allura red|sunset yellow|brilliant blue)\b",
        {"e102", "e110", "e129", "e133", "e150d"},
    ),
    "has_flavor_enhancer": (
        r"\b(?:e\s*-?\s*621|monosodium glutamate|msg|yeast extract|"
        r"disodium inosinate|disodium guanylate)\b",
        {"e621", "e631", "e627"},
    ),
}
FVL = "fruits-vegetables-legumes_100g"
INGREDIENT_FEATURES = [
    "additives_n_feature",
    "ingredient_characters",
    "ingredient_words",
] + ["marker_" + name for name in MARKERS]
UNKNOWN = {"", "unknown", "none", "null", "nan", "not-applicable", "not applicable"}
CATEGORY_LABELS = {
    "alcoholic beverages": "Alcoholic Beverages",
    "baby foods": "Baby & Infant Foods",
    "beverages": "Beverages",
    "cereals and potatoes": "Grains & Bakery",
    "composite foods": "Prepared Meals",
    "fat and sauces": "Condiments & Cooking",
    "fish meat eggs": "Meat & Seafood",
    "fruits and vegetables": "Fruits & Vegetables",
    "milk and dairy products": "Dairy & Eggs",
    "salty snacks": "Snacks & Sweets",
    "sugary snacks": "Snacks & Sweets",
}
CATEGORY_RULES = [
    (r"\b(?:alcohol|beer|wine|spirits?|liquors?|hard seltzer|cider)\b", "Alcoholic Beverages"),
    (r"\b(?:baby|infant|formula|baby milks?)\b", "Baby & Infant Foods"),
    (r"\b(?:supplements?|vitamins?|protein powders?|protein shakes?|sports nutrition|bodybuilding supplements?)\b", "Supplements & Nutrition"),
    (r"\b(?:non food products?|sunscreen|deodorant|soap)\b", "Other / Unclassified"),
    (r"\b(?:beverages?|drinks?|water|juice|soda|tea|coffee|smoothies?|kombucha)\b", "Beverages"),
    (r"\b(?:snacks?|sweets?|cand(?:y|ies)|chocolate|desserts?|cookies?|biscuits?|chips?|crisps?|popcorn|ice cream|pastries|cakes?)\b", "Snacks & Sweets"),
    (r"\b(?:dairy|milk|yogurt|cheese|butter|cream|eggs?|whey)\b", "Dairy & Eggs"),
    (r"\b(?:meat|beef|pork|chicken|turkey|poultry|fish|seafood|shrimp|salmon|sausage|bacon|ham|offals?)\b", "Meat & Seafood"),
    (r"\b(?:fruits?|vegetables?|produce|salads?|legumes?|beans?|lentils?|peas?)\b", "Fruits & Vegetables"),
    (r"\b(?:cereals?|grains?|bread|bakery|baked goods|flour|pasta|rice|noodles?|oats?|crackers?|tortillas?|potatoes)\b", "Grains & Bakery"),
    (r"\b(?:meals?|entrees?|prepared foods?|frozen foods?|pizza|soups?|broths?|appetizers?|sides?|dumplings?)\b", "Prepared Meals"),
    (r"\b(?:condiments?|sauces?|dressings?|oils?|fats?|spices?|herbs?|seasonings?|spreads?|sweeteners?|syrups?|vinegars?|toppings?|cooking helpers?)\b", "Condiments & Cooking"),
]
PNNS_SUBCATEGORIES = {
    "unsweetened beverages": ("Unsweetened Beverages", "Beverages"),
    "bread": ("Bread", "Grains & Bakery"),
    "milk and yogurt": ("Milk & Yogurt", "Dairy & Eggs"),
    "biscuits and cakes": ("Biscuits & Cakes", "Grains & Bakery"),
    "dressings and sauces": ("Dressings & Sauces", "Condiments & Cooking"),
    "sweets": ("Sweets & Candy", "Snacks & Sweets"),
    "sandwiches": ("Sandwiches", "Prepared Meals"),
    "one-dish meals": ("One-Dish Meals", "Prepared Meals"),
    "cereals": ("Cereals", "Grains & Bakery"),
    "appetizers": ("Appetizers", "Prepared Meals"),
    "cheese": ("Cheese", "Dairy & Eggs"),
    "breakfast cereals": ("Breakfast Cereals", "Grains & Bakery"),
    "pizza pies and quiches": ("Pizza, Pies & Quiches", "Prepared Meals"),
    "vegetables": ("Vegetables", "Fruits & Vegetables"),
    "sweetened beverages": ("Sweetened Beverages", "Beverages"),
    "artificially sweetened beverages": ("Artificially Sweetened Beverages", "Beverages"),
    "processed meat": ("Processed Meat", "Meat & Seafood"),
    "meat": ("Meat & Poultry", "Meat & Seafood"),
    "fats": ("Fats & Oils", "Condiments & Cooking"),
    "fish and seafood": ("Fish & Seafood", "Meat & Seafood"),
    "fruits": ("Fruits", "Fruits & Vegetables"),
    "nuts": ("Nuts & Seeds", "Fruits & Vegetables"),
    "eggs": ("Eggs", "Dairy & Eggs"),
    "legumes": ("Legumes", "Fruits & Vegetables"),
    "plant-based milk substitutes": ("Plant-Based Milk Alternatives", "Dairy & Eggs"),
    "waters and flavored waters": ("Water & Flavored Water", "Beverages"),
    "chocolate products": ("Chocolate", "Snacks & Sweets"),
    "ice cream": ("Ice Cream", "Dairy & Eggs"),
    "fruit juices": ("Fruit Juice", "Beverages"),
    "dried fruits": ("Dried Fruit", "Fruits & Vegetables"),
    "potatoes": ("Potatoes", "Grains & Bakery"),
    "teas and herbal teas and coffees": ("Tea & Coffee", "Beverages"),
    "salty and fatty products": ("Salty Snacks", "Snacks & Sweets"),
    "alcoholic beverages": ("Alcoholic Beverages", "Alcoholic Beverages"),
    "dairy desserts": ("Dairy Desserts", "Dairy & Eggs"),
    "pastries": ("Pastries", "Grains & Bakery"),
    "soups": ("Soups & Broths", "Prepared Meals"),
    "baby foods": ("Baby Foods", "Baby & Infant Foods"),
    "fruit nectars": ("Fruit Nectars", "Beverages"),
    "baby milks": ("Baby Formula", "Baby & Infant Foods"),
    "offals": ("Offal", "Meat & Seafood"),
}
SUBCATEGORY_RULES = [
    (r"\b(?:artificially sweetened beverages?|diet drinks?|zero[- ]sugar drinks?)\b", "Beverages", "Artificially Sweetened Beverages"),
    (r"\b(?:unsweetened beverages?|plain water|mineral water|spring water)\b", "Beverages", "Unsweetened Beverages"),
    (r"\b(?:sweetened beverages?|soft drinks?|sodas?|energy drinks?|sports drinks?)\b", "Beverages", "Sweetened Beverages"),
    (r"\b(?:fruit juices?|fruit nectars?)\b", "Beverages", "Fruit Juice"),
    (r"\b(?:teas?|coffees?|kombucha)\b", "Beverages", "Tea & Coffee"),
    (r"\b(?:waters?|beverages?|drinks?|juices?)\b", "Beverages", "Other Beverages"),
    (r"\b(?:baby milks?|infant formulas?|baby formulas?)\b", "Baby & Infant Foods", "Baby Formula"),
    (r"\b(?:baby foods?|infant foods?)\b", "Baby & Infant Foods", "Baby Foods"),
    (r"\b(?:processed meat|sausages?|bacon|ham|deli meats?)\b", "Meat & Seafood", "Processed Meat"),
    (r"\b(?:fish|seafood|shrimp|salmon|tuna|shellfish)\b", "Meat & Seafood", "Fish & Seafood"),
    (r"\b(?:meat|beef|pork|chicken|turkey|poultry|offal)\b", "Meat & Seafood", "Meat & Poultry"),
    (r"\b(?:milk|yogurt|yoghurt|dairy|cheese|whey|cream|butter)\b", "Dairy & Eggs", "Milk & Dairy"),
    (r"\b(?:plant[- ]based milk|milk substitutes?|dairy alternatives?)\b", "Dairy & Eggs", "Plant-Based Milk Alternatives"),
    (r"\b(?:eggs?)\b", "Dairy & Eggs", "Eggs"),
    (r"\b(?:ice cream|frozen desserts?)\b", "Dairy & Eggs", "Ice Cream & Frozen Desserts"),
    (r"\b(?:dairy desserts?)\b", "Dairy & Eggs", "Dairy Desserts"),
    (r"\b(?:salty snacks?|chips?|crisps?|popcorn|pretzels?)\b", "Snacks & Sweets", "Salty Snacks"),
    (r"\b(?:chocolate|cocoa)\b", "Snacks & Sweets", "Chocolate"),
    (r"\b(?:sweets?|cand(?:y|ies)|gummies?)\b", "Snacks & Sweets", "Sweets & Candy"),
    (r"\b(?:biscuits?|cookies?|crackers?)\b", "Grains & Bakery", "Biscuits & Crackers"),
    (r"\b(?:cakes?|pastries?|croissants?|baked goods?)\b", "Grains & Bakery", "Cakes & Pastries"),
    (r"\b(?:breakfast cereals?|granola|oatmeal)\b", "Grains & Bakery", "Breakfast Cereals"),
    (r"\b(?:bread|bagels?|buns?|rolls?)\b", "Grains & Bakery", "Bread"),
    (r"\b(?:cereals?|grains?|pasta|rice|noodles?|flour|potatoes)\b", "Grains & Bakery", "Cereals & Starches"),
    (r"\b(?:sandwiches?|wraps?)\b", "Prepared Meals", "Sandwiches"),
    (r"\b(?:pizza|pies?|quiches?)\b", "Prepared Meals", "Pizza, Pies & Quiches"),
    (r"\b(?:soups?|broths?)\b", "Prepared Meals", "Soups & Broths"),
    (r"\b(?:appetizers?|sides?|dumplings?)\b", "Prepared Meals", "Appetizers & Sides"),
    (r"\b(?:one[- ]dish meals?|entrees?|prepared meals?|frozen meals?)\b", "Prepared Meals", "Prepared Meals"),
    (r"\b(?:fruit juices?|fruit nectars?|juices?)\b", "Beverages", "Fruit Juice"),
    (r"\bfruits? and vegetables\b", "Fruits & Vegetables", "Fruits & Vegetables"),
    (r"\b(?:fruits?|berries|dried fruits?)\b", "Fruits & Vegetables", "Fruits"),
    (r"\b(?:vegetables?|salads?|produce)\b", "Fruits & Vegetables", "Vegetables"),
    (r"\b(?:legumes?|beans?|lentils?|peas?|nuts?|seeds?)\b", "Fruits & Vegetables", "Legumes, Nuts & Seeds"),
    (r"\b(?:dressings?|sauces?|condiments?|salsa)\b", "Condiments & Cooking", "Dressings & Sauces"),
    (r"\b(?:oils?|fats?|margarine)\b", "Condiments & Cooking", "Fats & Oils"),
    (r"\b(?:sweeteners?|syrups?|honey)\b", "Condiments & Cooking", "Sweeteners & Syrups"),
    (r"\b(?:spices?|herbs?|seasonings?)\b", "Condiments & Cooking", "Herbs & Spices"),
    (r"\b(?:alcohol|beer|wine|spirits?|liquors?|hard seltzer|cider)\b", "Alcoholic Beverages", "Alcoholic Beverages"),
    (r"\b(?:protein powders?|protein shakes?|bodybuilding supplements?)\b", "Supplements & Nutrition", "Protein Supplements"),
    (r"\b(?:supplements?|vitamins?|protein powders?|protein shakes?|sports nutrition)\b", "Supplements & Nutrition", "Supplements & Nutrition"),
]


def field(df, name):
    return df[name] if name in df else pd.Series(pd.NA, index=df.index, dtype="string")

def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .map(lambda x: unicodedata.normalize("NFKC", x))
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def tableau_categories(raw):
    """Return compact category and subcategory display dimensions."""
    pnns = normalize(field(raw, "pnns_groups_1"))
    category = pnns.map(CATEGORY_LABELS)
    unknown = category.isna()
    category_text = normalize(field(raw, "categories_en"))
    category_text = category_text.str.replace(
        "plant-based foods and beverages", "", regex=False
    )
    detail_text = normalize(field(raw, "pnns_groups_2"))
    fallback_text = category_text + " " + detail_text

    def apply_category_rules(text, mask):
        for pattern, label in CATEGORY_RULES:
            matched = mask & text.str.contains(pattern, regex=True, na=False)
            category.loc[matched] = label
            mask &= ~matched
        return mask

    unresolved = apply_category_rules(fallback_text, unknown.copy())
    product_text = normalize(field(raw, "product_name"))
    apply_category_rules(product_text, unresolved)
    category = category.fillna("Other / Unclassified").astype("string")

    pnns_detail = detail_text.map(PNNS_SUBCATEGORIES)
    subcategory = pnns_detail.map(
        lambda value: value[0] if isinstance(value, tuple) else pd.NA
    ).astype("string")
    detail_parent = pnns_detail.map(
        lambda value: value[1] if isinstance(value, tuple) else pd.NA
    )
    subcategory = subcategory.where(detail_parent.eq(category))

    def apply_subcategory_rules(text):
        for pattern, parent, label in SUBCATEGORY_RULES:
            matched = (
                subcategory.isna()
                & category.eq(parent)
                & text.str.contains(pattern, regex=True, na=False)
            )
            subcategory.loc[matched] = label

    apply_subcategory_rules(fallback_text)
    apply_subcategory_rules(product_text)
    default_subcategory = category.map(
        {
            "Beverages": "Other Beverages",
            "Grains & Bakery": "Other Grains & Bakery",
            "Fruits & Vegetables": "Other Fruits & Vegetables",
            "Dairy & Eggs": "Other Dairy & Eggs",
            "Meat & Seafood": "Other Meat & Seafood",
            "Prepared Meals": "Other Prepared Foods",
            "Snacks & Sweets": "Other Snacks & Sweets",
            "Condiments & Cooking": "Other Condiments",
            "Baby & Infant Foods": "Other Baby Foods",
            "Alcoholic Beverages": "Other Alcoholic Beverages",
            "Supplements & Nutrition": "Other Supplements",
            "Other / Unclassified": "Other / Unclassified",
        }
    )
    subcategory = subcategory.fillna(default_subcategory).astype("string")
    return pd.DataFrame(
        {"unified_category": category, "sub_category": subcategory},
        index=raw.index,
    )


def tableau_category(raw):
    """Return the broad Tableau display category."""
    return tableau_categories(raw)["unified_category"]


def parse_tags(value):
    if pd.isna(value) or str(value).strip().lower() in UNKNOWN:
        return set()
    return {
        x.strip().lower().removeprefix("en:")
        for x in str(value).split(",")
        if x.strip()
    }

def ingredient_features(df):
    text = normalize(field(df, "ingredients_text_norm"))
    present = ~text.isin(UNKNOWN)
    tags = field(df, "additives_tags").map(parse_tags)
    count = pd.to_numeric(field(df, "additives_n"), errors="coerce").astype(float)
    count = count.where(
        np.isfinite(count) & count.ge(0) & count.le(500) & count.eq(count.round())
    )
    complete_tags = count.notna() & count.eq(tags.map(len))
    result = pd.DataFrame(index=df.index)
    result["additives_n_feature"] = count
    result["ingredient_characters"] = text.str.len().astype(float).where(present)
    result["ingredient_words"] = text.str.split().str.len().astype(float).where(present)
    for name, (pattern, codes) in MARKERS.items():
        matched = text.str.contains(pattern, regex=True, na=False) | tags.map(
            lambda t: bool(t & codes)
        )
        known = present | matched | (complete_tags if codes else False)
        result["marker_" + name] = matched.astype(float).where(known)
    return result


def tableau_additive_features(df):
    """Derive the legacy Tableau additive flags from ingredient and tag evidence."""
    ingredients = normalize(field(df, "ingredients_text"))
    additive_names = normalize(field(df, "additives_en"))
    tags = field(df, "additives_tags").map(parse_tags)
    evidence = (
        ~ingredients.isin(UNKNOWN)
        | ~additive_names.isin(UNKNOWN)
        | tags.map(bool)
    )
    text = ingredients + " " + additive_names
    result = pd.DataFrame(index=df.index)
    for name, (pattern, codes) in TABLEAU_ADDITIVE_MARKERS.items():
        matched = text.str.contains(pattern, regex=True, na=False) | tags.map(
            lambda values: bool(values & codes)
        )
        result[name] = matched.astype("Int64").where(evidence)

    count = pd.to_numeric(field(df, "additives_n"), errors="coerce")
    known_flags = result[list(TABLEAU_ADDITIVE_MARKERS)].fillna(0)
    result["has_other_additive"] = (
        count.gt(0) & known_flags.eq(0).all(axis=1)
    ).astype("Int64").where(count.notna())
    return result


def load_processed_data(data_dir, clean_function=None):
    """Read both full exports, preserving identifiers and input-file provenance."""
    data_dir = Path(data_dir)
    files = [
        "us_openfoodfacts_nutri_nova_202407_202606.csv",
        "us_openfoodfacts_unlabeled_202407_202606.csv",
    ]
    parts, audit = [], []
    for name in files:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Full processed file not found: {path}; do not substitute sample_5000."
            )
        frame = pd.read_csv(path, dtype={"code": "string"}, low_memory=False)
        before = len(frame)
        if clean_function is not None:
            frame = clean_function(frame)
        audit.append(
            {
                "file": name,
                "loaded_rows": before,
                "after_exact_deduplication": len(frame),
                "columns": len(frame.columns),
            }
        )
        frame = frame.copy()
        frame["input_file"] = name
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    required = {"code", "nova_group", "nutriscore_grade", "ingredients_text"}
    if required - set(data):
        raise ValueError(f"Missing required fields: {sorted(required-set(data))}")
    return data, pd.DataFrame(audit)

def engineer_features(raw):
    """Create deterministic features on a copy without filling missing values."""
    df = raw.copy()
    df["code"] = field(df, "code").astype("string").str.strip()
    df["_modified"] = pd.to_numeric(field(df, "last_modified_t"), errors="coerce")
    identified = df.code.notna() & df.code.ne("")
    kept = (
        df.loc[identified].sort_values("_modified").drop_duplicates("code", keep="last")
    )
    df = pd.concat([kept, df.loc[~identified]], ignore_index=True)
    df["code_generated"] = df.code.isna() | df.code.eq("")
    generated = pd.Series([f"__missing_code_{i}" for i in df.index], index=df.index)
    df["product_id"] = df.code.where(~df.code_generated, generated)
    assert df.product_id.is_unique
    nova = pd.to_numeric(field(df, "nova_group"), errors="coerce")
    df["nova_target"] = nova.where(nova.isin([1, 2, 3, 4]))
    grade = normalize(field(df, "nutriscore_grade"))
    df["nutri_target"] = grade.where(grade.isin(list("abcde")))
    df["nutri_not_applicable"] = grade.eq("not-applicable")
    df["ingredients_text_norm"] = normalize(field(df, "ingredients_text"))
    df["ingredient_present"] = ~df.ingredients_text_norm.isin(UNKNOWN)
    # Only the declared FVL field is used for formula completion here.
    # Legacy fruit/vegetable/nut fields are not equivalent to the 2023 input.
    audits = []
    numeric_fields = list(
        dict.fromkeys(
            NUTRIENT_CANDIDATES
            + [
                "energy_100g",
                "energy-kj_100g",
                "salt_100g",
                "carbohydrates_100g",
                "fruits-vegetables-legumes_100g",
            ]
        )
    )
    for name in numeric_fields:
        values = pd.to_numeric(field(df, name), errors="coerce").astype(float)
        bound = (
            4200
            if name in {"energy_100g", "energy-kj_100g"}
            else 1000 if name == "energy-kcal_100g" else 100
        )
        invalid = values.notna() & (
            ~np.isfinite(values) | values.lt(0) | values.gt(bound)
        )
        df[name] = values.mask(invalid)
        audits.append(
            {
                "field": name,
                "present_in_input": name in raw,
                "original_numeric_rows": int(values.notna().sum()),
                "invalid_numeric_rows": int(invalid.sum()),
                "usable_numeric_pct_before_conversion": round(
                    df[name].notna().mean() * 100, 2
                ),
            }
        )
    energy_kj = df["energy-kj_100g"].fillna(df["energy_100g"])
    df["energy_kcal_converted_from_kj"] = (
        df["energy-kcal_100g"].isna() & energy_kj.notna()
    )
    df["energy-kcal_100g"] = df["energy-kcal_100g"].fillna(energy_kj / 4.184)
    df["flag_satfat_gt_fat"] = (df["saturated-fat_100g"] > df["fat_100g"] + 0.1).fillna(
        False
    )
    df["flag_sugars_gt_carbs"] = (
        df["sugars_100g"] > df["carbohydrates_100g"] + 0.1
    ).fillna(False)
    # Keep inconsistent products in the dataset, but exclude them from modeling.
    df["needs_review"] = df.flag_satfat_gt_fat | df.flag_sugars_gt_carbs
    pnns = normalize(field(df, "pnns_groups_1"))
    category = normalize(field(df, "categories_en")).str.split(",").str[0].str.strip()
    pnns_known, category_known = ~pnns.isin(UNKNOWN), ~category.isin(UNKNOWN)
    df["category"] = pnns.where(pnns_known, category.where(category_known, "unknown"))
    df["category_source"] = np.select(
        [pnns_known, category_known],
        ["observed_pnns", "observed_OFF_category"],
        default="unknown",
    )
    generated_features = ingredient_features(df)
    df[generated_features.columns] = generated_features
    df["observed_core_nutrients"] = df[CORE_NUTRIENTS].notna().sum(axis=1)
    return df, pd.DataFrame(audits)


def assign_splits(df, seed=33):
    """Keep linked duplicate proxies together in each target-specific split.

    Link identical detailed ingredient texts OR matching known brand/product names.
    Connected records share a group even if one link uses text and another uses
    a name. These proxies can overgroup products and miss some near duplicates.
    """
    df = df.copy()
    recipe = normalize(df.ingredients_text_norm).str.replace(
        r"[^\w\s]", " ", regex=True
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    name = normalize(field(df, "product_name"))
    brand = normalize(field(df, "brands"))
    detailed_recipe = recipe.str.len().ge(40) & recipe.str.split().str.len().ge(6)
    specific_name = name.str.len().ge(12) & name.str.split().str.len().ge(2)
    known_brand = ~brand.isin(UNKNOWN)

    # Each row starts in its own group. Matching keys connect their groups.
    parent = list(range(len(df)))

    def find_root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    recipe_keys = recipe.where(detailed_recipe, "")
    name_keys = (brand + ":" + name).where(known_brand & specific_name, "")
    for keys in [recipe_keys, name_keys]:
        first_seen = {}
        for i, key in enumerate(keys):
            if not key:
                continue
            if key in first_seen:
                root_a = find_root(i)
                root_b = find_root(first_seen[key])
                parent[max(root_a, root_b)] = min(root_a, root_b)
            else:
                first_seen[key] = i
    df["family_group"] = [f"g{find_root(i)}" for i in range(len(df))]

    summaries = []
    for target, column in [("nova_target", "nova_split"), ("nutri_target", "nutri_split")]:
        labeled = df.loc[df[target].notna() & ~df.needs_review]
        df[column] = "inference"
        df.loc[df.needs_review, column] = "excluded_quality"
        if labeled.family_group.nunique() < 5:
            raise ValueError(f"Not enough eligible labeled groups for {target}.")
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, positions) in enumerate(splitter.split(
            np.zeros(len(labeled)), labeled[target].astype(str), labeled.family_group
        )):
            split = "test" if fold == 0 else "validation" if fold == 1 else "train"
            df.loc[labeled.index[positions], column] = split
        assert df.loc[labeled.index].groupby("family_group")[column].nunique().max() == 1
        table = pd.crosstab(df.loc[df[target].notna(), column], df.loc[df[target].notna(), target]).reset_index()
        table.insert(0, "task", target)
        summaries.append(table)
    return df, summaries


def exploratory_analysis(train, validation, output_dir):
    """Fit PCA preprocessing on training rows; project validation rows."""
    correlations = train[CORE_NUTRIENTS].corr(min_periods=100)
    save_table(correlations.rename_axis("feature").reset_index(), output_dir, "nutrient_correlations.csv")
    usable = [c for c in CORE_NUTRIENTS if train[c].nunique() > 1]
    if len(usable) < 2:
        raise ValueError("PCA needs at least two nonconstant nutrients.")
    preparation = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(preparation.fit_transform(train[usable]))
    scores = validation[["product_id", "nova_target"]].copy()
    scores[["PC1", "PC2"]] = pca.transform(preparation.transform(validation[usable]))
    assert np.isfinite(scores[["PC1", "PC2"]]).all().all()
    save_table(scores, output_dir, "pca_validation.csv")
    save_table(pd.DataFrame({"component": ["PC1", "PC2"], "variance_ratio": pca.explained_variance_ratio_}), output_dir, "pca_variance.csv")
    plot_eda(correlations, scores, output_dir)

def select_nutrients(training, minimum_coverage=0.25):
    """Keep varied nutrients with at least 25% training coverage.

    The current Open Food Facts exports have roughly 29-33% coverage for core
    nutrients; the model pipeline imputes missing values and adds indicators.
    """
    rows, selected = [], []
    for name in NUTRIENT_CANDIDATES:
        values = training[name]
        keep = (
            values.notna().mean() >= minimum_coverage
            and values.nunique(dropna=True) >= 5
        )
        rows.append(
            {
                "feature": name,
                "training_coverage_pct": round(values.notna().mean() * 100, 2),
                "training_distinct_values": values.nunique(dropna=True),
                "selected": bool(keep),
            }
        )
        if keep:
            selected.append(name)
    if len(selected) < 4:
        raise ValueError(
            "Fewer than four sufficiently populated nutrient predictors; review the complete data."
        )
    return selected, pd.DataFrame(rows)
