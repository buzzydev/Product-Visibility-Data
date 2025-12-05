"""
train_models.py
----------------

This module provides a **comprehensive, heavily commented** training
pipeline for the cleaned product visibility dataset.  The overarching
goal is to build a suite of machine‑learning models that answer
important questions for both **consumers** and **outlet owners**.  For
instance:

* Where can a consumer reliably find their favourite drink?  Which
  outlets are likely to be well stocked?
* Which brands dominate shelf space?  How many brands does an outlet
  typically carry?  Are there patterns in the way brands are
  co‑stocked?
* How do outlets cluster geographically and by product variety?  Who
  are my nearest competitors and what do they sell?

To address these questions, the script trains a variety of models,
including classifiers for stock condition and brand dominance, binary
classifiers for brand and packaging availability, regression models for
synthetic footfall estimation, clustering for outlet segmentation and
nearest‑neighbour indexing for competitor lookup.  It also contains an
**experimental consumer preference stub** to illustrate how a true
recommendation system might be structured once real user interaction
data is collected.

Each trained model is persisted to disk as a ``.pkl`` file under the
``models/`` directory, making it easy to load them later in a
Streamlit app or other application.  Throughout the code you will see
detailed inline comments explaining what each step is doing and why
certain decisions have been made.  Experimental components are clearly
marked.

### Key artifacts produced

1. **stock_condition_model.pkl** – Random‑Forest classifier predicting
   the stock condition category ("Well stocked", "Partially stocked",
   "Almost empty", "Out of stock" or "Not Applicable").
2. **stock_condition_xgb_model.pkl** – XGBoost version of the stock
   condition classifier with tuned hyperparameters.
3. **stock_condition_catboost_model.pkl** – CatBoost version which
   handles categorical features natively.
4. **dominant_brand_model.pkl** – Multi‑class classifier predicting
   which brand is dominant in an outlet.  Only trained if enough
   variety exists in the data.
5. **multi_brand_model.pkl** – Binary classifier that predicts
   whether an outlet stocks more than one brand.
6. **brand_presence_models/** – Directory containing logistic
   classifiers (and optionally XGBoost classifiers) for each major
   brand.  These estimate the probability that a specific brand is
   available at an outlet.
7. **packaging_preference_models/** – Directory with binary
   classifiers for each packaging format (PET, glass, can), allowing
   you to infer whether an outlet carries a given package type.
8. **outlet_cluster_model.pkl** – K‑means clustering model (and
   associated scaler) grouping outlets based on product variety,
   packaging, display methods and location.
9. **competitor_nn_model.pkl** – Nearest‑neighbour index (and scaler)
   built on latitude and longitude; enables queries for the closest
   competitor outlets.
10. **footfall_regressor_model.pkl** and
    **footfall_xgb_regressor_model.pkl** – Regression models using a
    synthetic target to approximate outlet footfall.  These are
    labelled experimental because no real footfall data is available.
11. **consumer_preference_stub.pkl** – A simple, rule‑based mapping
    providing fallback recommendations within broad drink categories
    (e.g., cola vs fruit soda).  This is a stand‑in for a true
    recommendation engine and is marked experimental.

### Running the script

#Before you run:
Please install the following libraries:

`pip install pandas numpy joblib scikit xgboost catboost


Ensure that your cleaned CSV (e.g., ``cleaned_product_visibility.csv``)
is in the same directory as this script or adjust the ``DATA_PATH``
constant below.  Then execute:

#Run your script with the following command:
```sh
python train_models.py
```

The script will read the data, engineer additional features,
train all models, print evaluation metrics to the console, and
write the models to the ``models/`` folder.  All long‑running tasks
have progress messages to keep you informed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Gradient boosting libraries (installed in this environment)
from xgboost import XGBClassifier, XGBRegressor  # type: ignore
from catboost import CatBoostClassifier  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default path to the cleaned dataset.  Adjust this to point to your
# cleaned CSV.  The file ``cleaned_product_visibility.csv`` is the
# version produced by your cleaning script; if you are using a
# different filename (e.g., ``cleaned_product_visibility-2.csv``), set
# this accordingly.
DATA_PATH = Path("../Files/cleaned_product_visibility.csv")

# Directory for saving trained model objects.  All pickle files and
# subdirectories will be created under this path.  Feel free to change
# ``MODEL_DIR`` to a different location if you want to organise your
# models elsewhere.
model_dir = Path("../Models")
model_dir.mkdir(exist_ok=True)

# Column groups
# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

# ``PRODUCT_COLS`` lists all binary indicator columns for soft drink brands.
# Each entry is 1 if the brand is present at the outlet and 0 otherwise.
PRODUCT_COLS: List[str] = [
    "Coca_Cola",
    "Pepsi",
    "Bigi",
    "RC_Cola",
    "7Up",
    "Fanta",
    "Sprite",
    "La_Casera",
    "Schweppes",
    "Fayrouz",
    "Mirinda",
    "Mountain_Dew",
    "Teem",
    "American_Cola",
    "Product_Others",
]

# ``PACKAGE_COLS`` contains binary indicators for packaging formats.
# Each field is 1 if that package type is stocked.  Note:  some
# datasets may name the combined packaging field differently
# (e.g., ``Package_Type_(Combined_Response)``); that column is used
# only for text analysis and is dropped during modelling.
PACKAGE_COLS: List[str] = [
    "PET_Bottle_(50cl/1L)",
    "Glass_Bottle_(35cl/60cl)",
    "Can_(33cl)",
]

# ``DISPLAY_COLS`` lists binary indicators for where products are
# displayed in the outlet.  As above, the combined text field
# ``Product_Display_(Combined_Response)`` is excluded from numeric
# modelling.
DISPLAY_COLS: List[str] = [
    "On_Shelf/Carton",
    "In_Refrigerator/Cooler",
    "On_Display_Stand",
]

# ``MULTI_RESPONSE_COLS`` enumerates text fields that combine multiple
# responses in a single column (e.g., a comma‑separated list of
# products or display methods).  These columns are excluded from
# modelling because they are not directly numeric.  Some datasets may
# use ``Package_Type_Combined`` while others use
# ``Package_Type_(Combined_Response)``; include both to guard
# against missing columns.
MULTI_RESPONSE_COLS: List[str] = [
    "Type_Of_Product_(Combined_Response)",
    "Product_Display_(Combined_Response)",
    "Package_Type_(Combined_Response)",
    "Package_Type_Combined",
    "Product_With_Higher_Shelf/Refrigerator_Presence",
]


def load_and_engineer(path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset and engineer additional numeric and
    categorical features.

    This function performs several tasks:

    * Reads the CSV file into a DataFrame.  We don't set an index here
      because the original ``S/N`` column is simply retained as a
      regular column.  If you prefer to use ``S/N`` as the index,
      adjust accordingly.
    * Converts all binary indicator columns (brands, packages and
      display methods) to numeric types (int64), coercing any
      unexpected values to 0.
    * Creates derived count features, such as the number of brands
      present at an outlet, the number of package types and the number
      of display methods.  These counts are helpful for models that
      rely on numeric features rather than a long list of dummy
      variables.
    * Creates a binary flag ``is_multiple_brands`` indicating whether
      more than one brand is stocked and a categorical code
      ``brand_variety_code`` (0=single brand, 1=two brands, 2=three or
      more brands).
    * Extracts a ``dominant_brand`` column from
      ``Product_With_Higher_Shelf/Refrigerator_Presence``, if present,
      and title‑cases the values.  If that column does not exist or
      the value is missing, ``dominant_brand`` is set to "Unknown".

    Parameters
    ----------
    path : Path
        Path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with additional engineered columns ready for
        modelling.
    """
    # Load the CSV; ``low_memory=False`` avoids dtype inference issues
    df = pd.read_csv(path, low_memory=False)
    # Ensure all brand/package/display indicators are numeric.  Some
    # cleaning scripts may leave these as strings (e.g., "1" or "0"), so
    # we coerce them to numbers.  Any non‑numeric value becomes NaN and
    # is subsequently filled with 0.
    df[PRODUCT_COLS + PACKAGE_COLS + DISPLAY_COLS] = (
        df[PRODUCT_COLS + PACKAGE_COLS + DISPLAY_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    # Derived numeric features
    df["num_brands_present"] = df[PRODUCT_COLS].sum(axis=1)
    df["num_package_types"] = df[PACKAGE_COLS].sum(axis=1)
    df["num_display_methods"] = df[DISPLAY_COLS].sum(axis=1)
    # Binary flag: more than one brand present
    df["is_multiple_brands"] = (df["num_brands_present"] > 1).astype(int)
    # Brand variety code (0=single, 1=double, 2=3 or more)
    def brand_variety(n: int) -> int:
        if n <= 1:
            return 0
        elif n == 2:
            return 1
        else:
            return 2
    df["brand_variety_code"] = df["num_brands_present"].apply(brand_variety)
    # Extract dominant brand information if available; else mark as Unknown
    if "Product_With_Higher_Shelf/Refrigerator_Presence" in df.columns:
        df["dominant_brand"] = (
            df["Product_With_Higher_Shelf/Refrigerator_Presence"]
            .fillna("Unknown")
            .astype(str)
            .str.title()
        )
    else:
        df["dominant_brand"] = "Unknown"
    return df


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """Construct a ColumnTransformer for preprocessing.

    Categorical columns are one‑hot encoded; numeric columns are scaled.

    Parameters
    ----------
    categorical_cols : List[str]
        Names of categorical features.
    numeric_cols : List[str]
        Names of numeric features.

    Returns
    -------
    ColumnTransformer
        The preprocessing transformer.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )


def train_stock_condition_model(df: pd.DataFrame) -> None:
    """Train a classifier to predict outlet stock condition.

    Saves the fitted pipeline as ``stock_condition_model.pkl``.
    """
    target = "Stock_Condition"
    # Determine which multi‑response text fields exist in the data and
    # exclude them from the feature matrix.  ``Stock_Condition`` itself
    # should remain as the target and therefore not be dropped here.
    # Exclude multi‑response columns and the derived ``dominant_brand``
    # column from the feature matrix.  ``dominant_brand`` contains
    # strings and is not encoded here.
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["dominant_brand"]
    # Filter out targets with very few samples to avoid extreme imbalance
    df_filtered = df.copy()
    # Features and target
    X = df_filtered.drop(columns=[target] + drop_cols)
    y = df_filtered[target]
    categorical_cols = ["Type_Of_Outlet"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight = "balanced")
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Display evaluation metrics for the stock condition model
    print("\nStock Condition Model Performance:")
    print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
    joblib.dump(pipe, "stock_condition_model.pkl")


def train_dominant_brand_model(df: pd.DataFrame) -> None:
    """Train a classifier to predict the dominant brand in an outlet.

    Only trains the model if multiple unique dominant brands exist.
    Saves the fitted pipeline as ``dominant_brand_model.pkl``.
    """
    # Remove rows where dominant_brand is Unknown
    df_non_unknown = df[df["dominant_brand"] != "Unknown"].copy()
    if df_non_unknown["dominant_brand"].nunique() < 2:
        print("Insufficient variety in dominant_brand to train model.")
        return
    target = "dominant_brand"
    # Exclude multi‑response text fields and other targets from the
    # feature matrix.  ``dominant_brand`` and ``Stock_Condition`` are
    # removed separately.
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df_non_unknown.columns] + ["Stock_Condition"]
    X = df_non_unknown.drop(columns=[target] + drop_cols)
    y = df_non_unknown[target]
    categorical_cols = ["Type_Of_Outlet"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight = "balanced")
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Display evaluation metrics for the dominant brand classifier
    print("\nDominant Brand Model Performance:")
    print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
    joblib.dump(pipe, "dominant_brand_model.pkl")


def train_multi_brand_classifier(df: pd.DataFrame) -> None:
    """Train a binary classifier to predict if an outlet stocks multiple brands.

    Saves the model as ``multi_brand_model.pkl``.
    """
    target = "is_multiple_brands"
    # Exclude multi‑response text fields and other targets from the
    # feature matrix.  ``is_multiple_brands`` itself is the target.
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["Stock_Condition", "dominant_brand"]
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    categorical_cols = ["Type_Of_Outlet"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Display evaluation metrics for the multi‑brand classifier
    print("\nMulti‑Brand Classifier Performance:")
    print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
    joblib.dump(pipe, "multi_brand_model.pkl")


def train_brand_presence_models(df: pd.DataFrame) -> None:
    """Train a binary classifier for each major brand's presence.

    Models are saved under ``brand_presence_models/BRAND.pkl``.  These
    classifiers estimate whether a specific brand is stocked at an
    outlet.  They can power consumer recommendations for locating
    desired drinks.
    """
    brand_dir = Path("brand_presence_models")
    brand_dir.mkdir(exist_ok=True)
    # Common feature set (excluding string fields and the target brand)
    # ``drop_base`` contains columns that should always be excluded from
    # brand presence modelling: multi‑response text fields and
    # unrelated targets.  We assemble this list dynamically based on
    # column presence to avoid KeyError.
    drop_base = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["Stock_Condition", "dominant_brand"]
    categorical_cols = ["Type_Of_Outlet"]
    for brand in PRODUCT_COLS:
        print(f"\nTraining presence model for {brand}...")
        X = df.drop(columns=drop_base + [brand])
        y = df[brand]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", model),
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
        joblib.dump(pipe, brand_dir / f"{brand}_presence_model.pkl")


def train_brand_presence_xgb_models(df: pd.DataFrame) -> None:
    """Train XGBoost classifiers for each brand's presence.

    This mirrors ``train_brand_presence_models`` but uses XGBClassifier
    with a default hyperparameter configuration.  Models are saved
    under ``brand_presence_models_xgb/BRAND.pkl``.
    """
    brand_dir = Path("brand_presence_models_xgb")
    brand_dir.mkdir(exist_ok=True)
    drop_base = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["Stock_Condition", "dominant_brand"]
    categorical_cols = ["Type_Of_Outlet"]
    for brand in PRODUCT_COLS:
        print(f"\nTraining XGBoost presence model for {brand}...")
        X = df.drop(columns=drop_base + [brand])
        y = df[brand]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", model),
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
        joblib.dump(pipe, brand_dir / f"{brand}_xgb_presence_model.pkl")


def train_packaging_preference_models(df: pd.DataFrame) -> None:
    """Train a binary classifier for each packaging type.

    Similar to brand presence models, this function trains a logistic
    regression model for each packaging indicator (PET, Glass, Can) to
    estimate whether an outlet carries that format.  Models are
    persisted under ``packaging_preference_models/``.
    """
    package_dir = Path("packaging_preference_models")
    package_dir.mkdir(exist_ok=True)
    drop_base = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["Stock_Condition", "dominant_brand"]
    categorical_cols = ["Type_Of_Outlet"]
    for pkg in PACKAGE_COLS:
        print(f"\nTraining packaging model for {pkg}...")
        X = df.drop(columns=drop_base + [pkg])
        y = df[pkg]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]
        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", model),
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test.values.ravel(), y_pred, zero_division=0))
        # Replace characters in the package name that are not
        # filesystem‑friendly (e.g., slashes) so that model filenames
        # do not accidentally create nested directories.  The original
        # column name is preserved in the model object itself.
        safe_name = pkg.replace("/", "_").replace(" ", "_")
        joblib.dump(pipe, package_dir / f"{safe_name}_preference_model.pkl")


def train_footfall_regression_model(df: pd.DataFrame) -> None:
    """Train a regression model to estimate outlet footfall or popularity.

    Because true traffic counts are not available, this function uses a
    synthetic footfall score as the target.  The score is computed as
    the sum of ``num_brands_present``, ``num_package_types`` and
    ``num_display_methods``, plus a numeric weight derived from the
    ``Stock_Condition`` (Well stocked=3, Partially=2, Almost empty=1,
    Out of stock=0, Not Applicable=1).  The resulting target is a
    rough proxy for outlet popularity and should be labelled clearly as
    experimental.
    """
    # Map stock condition to a numeric weight
    condition_map: Dict[str, float] = {
        "Well stocked": 3.0,
        "Partially stocked": 2.0,
        "Almost empty": 1.0,
        "Out of stock": 0.0,
        "Not Applicable": 1.0,
    }
    df_copy = df.copy()
    df_copy["stock_score"] = df_copy["Stock_Condition"].map(condition_map).fillna(1.0)
    # Synthetic footfall score
    df_copy["footfall_score"] = (
        df_copy["num_brands_present"]
        + df_copy["num_package_types"]
        + df_copy["num_display_methods"]
        + df_copy["stock_score"]
    )
    target = "footfall_score"
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df_copy.columns] + ["Stock_Condition", "dominant_brand"]
    X = df_copy.drop(columns=[target] + drop_cols)
    y = df_copy[target]
    categorical_cols = ["Type_Of_Outlet", "brand_variety_code"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test.values.ravel(), y_pred)
    mae = mean_absolute_error(y_test.values.ravel(), y_pred)
    print("\nFootfall Regression Model Performance:")
    print(f"R²: {r2:.3f}, MAE: {mae:.3f}")
    joblib.dump(pipe, "footfall_regressor_model.pkl")


def train_footfall_xgb_regression_model(df: pd.DataFrame) -> None:
    """Train an XGBoost regressor for synthetic footfall estimation.

    Uses the same synthetic footfall score defined in
    ``train_footfall_regression_model``.  Saves the model as
    ``footfall_xgb_regressor_model.pkl``.  As with the plain
    RandomForest, this model is experimental and should be replaced
    once real traffic data is collected.
    """
    condition_map: Dict[str, float] = {
        "Well stocked": 3.0,
        "Partially stocked": 2.0,
        "Almost empty": 1.0,
        "Out of stock": 0.0,
        "Not Applicable": 1.0,
    }
    df_copy = df.copy()
    df_copy["stock_score"] = df_copy["Stock_Condition"].map(condition_map).fillna(1.0)
    df_copy["footfall_score"] = (
    df_copy["num_brands_present"]
        + df_copy["num_package_types"]
        + df_copy["num_display_methods"]
        + df_copy["stock_score"]
    )
    target = "footfall_score"
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df_copy.columns] + ["Stock_Condition", "dominant_brand"]
    X = df_copy.drop(columns=[target] + drop_cols)
    y = df_copy[target]
    categorical_cols = ["Type_Of_Outlet", "brand_variety_code"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", model),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test.values.ravel(), y_pred)
    mae = mean_absolute_error(y_test.values.ravel(), y_pred)
    print("\nFootfall XGBoost Regression Model Performance:")
    print(f"R²: {r2:.3f}, MAE: {mae:.3f}")
    joblib.dump(pipe, "footfall_xgb_regressor_model.pkl")


def train_outlet_clustering(df: pd.DataFrame, n_clusters: int = 5) -> None:
    """Train a K‑means clustering model to segment outlets.

    Saves the fitted K‑means model as ``outlet_cluster_model.pkl``.
    """
    # Use brand counts, packaging counts, display counts and coordinates
    features = df[
        ["num_brands_present", "num_package_types", "num_display_methods", "Latitude", "Longitude"]
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    # Save both scaler and kmeans in a tuple
    joblib.dump((scaler, kmeans), "outlet_cluster_model.pkl")
    # Print cluster sizes
    labels, counts = pd.Series(kmeans.labels_).value_counts().sort_index().index, pd.Series(kmeans.labels_).value_counts().sort_index().values
    print("\nOutlet Cluster Sizes:")
    for lbl, cnt in zip(labels, counts):
        print(f"Cluster {lbl}: {cnt} outlets")


def train_competitor_nn(df: pd.DataFrame, n_neighbors: int = 5) -> None:
    """Train a k‑nearest neighbours model on outlet locations.

    This model can be used to query the nearest competing outlets to a
    given location.  Saves the model as ``competitor_nn_model.pkl``.
    """
    coords = df[["Latitude", "Longitude"]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(coords_scaled)
    joblib.dump((scaler, nn), "competitor_nn_model.pkl")
    print("\nCompetitor nearest‑neighbour model trained.")


def create_consumer_preference_stub(df: pd.DataFrame) -> None:
    """Create a stub for a consumer preference model.

    A true customer preference recommender requires data on user
    behaviour—what brands users prefer, how they respond when their
    preferred brand is unavailable, and which alternatives they select.
    Since no such data exists yet, this function constructs a simple
    rule‑based fallback system as a placeholder.  The stub maps each
    brand to a broad category (e.g., ``Cola``, ``Fruit Soda``, ``Other``)
    and defines an ordered list of suggestions within that category.

    When integrating into an application, this stub can provide
    category‑level recommendations if a user’s favourite drink is
    unavailable.  Once real user interaction data is collected, you
    should replace this with a trained recommendation model (e.g., a
    ranking model or collaborative filtering system).
    """
    # Define brand categories.  These mappings are based on the
    # description provided by the business: colas vs fruit sodas vs
    # other categories.  They are intentionally coarse and should be
    # refined in collaboration with marketing experts or based on user
    # feedback.  For example, Coca‑Cola, Pepsi, American Cola, RC
    # Cola and Bigi are colas; Fanta, Mirinda, Sprite, La Casera and
    # 7Up are fruit sodas; the remaining brands are grouped under
    # ``Other``.
    brand_categories: Dict[str, str] = {
        "Coca_Cola": "Cola",
        "Pepsi": "Cola",
        "Bigi": "Cola",
        "RC_Cola": "Cola",
        "American_Cola": "Cola",
        "Fanta": "Fruit Soda",
        "Mirinda": "Fruit Soda",
        "Sprite": "Fruit Soda",
        "La_Casera": "Fruit Soda",
        "7Up": "Fruit Soda",
        "Schweppes": "Other",
        "Fayrouz": "Other",
        "Mountain_Dew": "Other",
        "Teem": "Other",
        "Product_Others": "Other",
    }
    # Compute popularity of brands within each category as a proxy for
    # recommendation ranking.  Here we count how many outlets stock
    # each brand.
    popularity: Dict[str, int] = {b: int(df[b].sum()) for b in PRODUCT_COLS}
    # Build a mapping: category -> list of brands sorted by popularity
    category_suggestions: Dict[str, List[str]] = {}
    for brand, cat in brand_categories.items():
        category_suggestions.setdefault(cat, []).append(brand)
    for cat, brands in category_suggestions.items():
        category_suggestions[cat] = sorted(brands, key=lambda b: popularity.get(b, 0), reverse=True)
    # Save stub as a dictionary containing categories and suggestions
    stub = {
        "brand_categories": brand_categories,
        "category_suggestions": category_suggestions,
        "description": (
            "Experimental consumer preference stub.  This mapping can be"
            " used to recommend alternative drinks within the same category"
            " when a user’s preferred brand is unavailable.  Replace with"
            " a trained recommendation model once user interaction data is"
            " collected."
        ),
    }
    joblib.dump(stub, "consumer_preference_stub.pkl")
    print("\nConsumer preference stub created (experimental).")


def train_stock_condition_xgb_model(df: pd.DataFrame) -> None:
    """Train an XGBoost classifier for stock condition prediction.

    This function parallels ``train_stock_condition_model`` but uses
    XGBoost instead of a random forest.  Hyperparameters can be
    tuned manually or via cross‑validation; here we supply a
    reasonable default configuration.  The trained model is saved
    as ``stock_condition_xgb_model.pkl``.
    """
    # Set up target and feature matrix.  Drop any multi‑response fields
    # defined in ``MULTI_RESPONSE_COLS`` to avoid including long text
    # strings in the model.  We do not remove ``Stock_Condition`` here
    # because it is our target.
    target = "Stock_Condition"
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df.columns] + ["dominant_brand"]
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    # Encode target labels to integers because XGBoost does not
    # natively support string labels.  We fit the encoder on the full
    # target column and use it later to decode predictions back to
    # their original form for reporting.
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Identify categorical and numeric columns.  Only ``Type_Of_Outlet``
    # is categorical in this dataset; all other columns are numeric.
    categorical_cols = ["Type_Of_Outlet"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    # Initialise an XGBoost classifier.  The values provided here are
    # reasonable defaults but we will refine them via a grid search.
    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    # Create a pipeline that first applies preprocessing and then fits
    # the XGB classifier.  ``GridSearchCV`` will clone this pipeline
    # internally.
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", base_model),
    ])
    # Define a small hyperparameter grid for demonstration purposes.
    # In a production setting you might expand this grid or use
    # Bayesian optimisation to find better parameters.  Limiting the
    # number of combinations keeps runtime reasonable for this example.
    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }
    # Split the data.  We stratify to preserve the class distribution
    # across folds.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    # Perform grid search with 3‑fold cross‑validation.  The search
    # evaluates each combination of parameters and selects the one with
    # the highest mean cross‑validation score.  We set ``y_test.values.ravel(), y_pred, zero_division=0, n_jobs=1`` to
    # leverage all available CPU cores.
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=1,	
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    # Evaluate the best estimator on the held‑out test set
    best_pipe = grid_search.best_estimator_
    # Predict encoded labels on the test set and decode them back to
    # their original string form for a readable classification report.
    y_pred_encoded = best_pipe.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    print("\nStock Condition XGBoost Model Performance:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(classification_report(y_test_decoded, y_pred))
    # Persist the best model along with the label encoder for later
    # decoding.  When loading the model you must also load the
    # encoder to interpret predictions.
    joblib.dump({"model": best_pipe, "label_encoder": label_encoder}, "stock_condition_xgb_model.pkl")


def train_stock_condition_catboost_model(df: pd.DataFrame) -> None:
    """Train a CatBoost classifier for stock condition prediction.

    CatBoost can handle categorical features natively.  We separate
    categorical and numeric features and pass their indices to the
    CatBoost model.  The model is saved as
    ``stock_condition_catboost_model.pkl``.
    """
    # Prepare the feature matrix and target.  We drop any multi‑response
    # text fields defined in ``MULTI_RESPONSE_COLS`` to avoid
    # including long strings in the model.  ``Stock_Condition`` remains
    # as the target and is removed after features are selected.
    target = "Stock_Condition"
    drop_cols = [c for c in MULTI_RESPONSE_COLS if c in df.columns]
    df_cat = df.drop(columns=drop_cols)
    X = df_cat.drop(columns=[target])
    y = df_cat[target]
    # Encode target labels to integers.  CatBoost can work with
    # string labels, but encoding ensures consistency when
    # evaluating with scikit‑learn tools.
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Identify the indices of categorical columns based on pandas dtype.
    # CatBoost uses these indices to internally handle categorical data
    # without the need for one‑hot encoding.
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]
    # Train/test split with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    # Base model for CatBoost.  We do not specify iterations/depth/learning_rate
    # here because we will tune them via GridSearchCV.
    base_model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced"
    )
    # Define a modest hyperparameter grid.  ``iterations`` controls the
    # number of boosting rounds; ``depth`` determines the depth of each
    # tree; ``learning_rate`` scales the contribution of each tree.  A
    # wider grid would yield potentially better performance at the cost
    # of significantly longer training times.
    param_grid = {
        "iterations": [200, 300],
        "depth": [4, 6],
        "learning_rate": [0.03, 0.1],
    }
    from sklearn.model_selection import GridSearchCV
    # Perform grid search with 3‑fold cross‑validation.  CatBoost's
    # scikit‑learn interface supports cross‑validation and will handle
    # categorical features automatically when provided the index list.
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring= "accuracy",
                                            n_jobs=1,
        verbose=0,
    )
    
    # Fit the grid search.  ``cat_features`` must be passed during
    # fitting to ensure that CatBoost treats categorical columns
    # appropriately.
    grid_search.fit(X_train, y_train, cat_features=cat_features)
    # Select the best model and evaluate on the held‑out test set
    best_model = grid_search.best_estimator_
    y_pred_encoded = best_model.predict(X_test).ravel()
    # Decode labels for a human‑readable report
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test).ravel()
    print("\nStock Condition CatBoost Model Performance:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(classification_report(y_test_decoded, y_pred, zero_division=0))
    # Persist the model along with the feature names and label encoder.
    joblib.dump({"model": best_model, "features": X.columns.tolist(), "label_encoder": label_encoder}, "stock_condition_catboost_model.pkl")


def main() -> None:
    print("Loading and engineering data...")
    df = load_and_engineer(DATA_PATH)
    print(f"Data loaded: {len(df)} rows")
    # Train models
    # Classic random‑forest model for stock condition
    train_stock_condition_model(df)
    # Gradient boosting models for stock condition (XGBoost and CatBoost)
    train_stock_condition_xgb_model(df)
    train_stock_condition_catboost_model(df)
    # Dominant brand classifier
    train_dominant_brand_model(df)
    # Multi‑brand vs single‑brand classification
    train_multi_brand_classifier(df)
    # Brand presence models (logistic regression)
    train_brand_presence_models(df)
    # Brand presence models using XGBoost
    train_brand_presence_xgb_models(df)
    # Packaging preference models
    train_packaging_preference_models(df)
    # Outlet clustering and competitor nearest‑neighbour models
    train_outlet_clustering(df, n_clusters=5)
    train_competitor_nn(df, n_neighbors=5)
    # Footfall regression models (RandomForest and XGBoost)
    train_footfall_regression_model(df)
    train_footfall_xgb_regression_model(df)
    # Create an experimental consumer preference stub
    create_consumer_preference_stub(df)
    print("\nTraining complete. Models saved in", model_dir)


if __name__ == "__main__":
    main()