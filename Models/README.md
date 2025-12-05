# Machine‑Learning Models

This directory contains pre‑trained machine‑learning models and related artifacts for the Product Visibility Challenge.  These models are trained on the cleaned dataset to support various prediction tasks in the application.

## Model Types

The following model families are included:

| Model | Purpose | Location |
|------|---------|---------|
| **Stock condition classifier** | Predicts whether an outlet is well stocked, partially stocked, almost empty or out of stock based on outlet type and the number of brands stocked.  Trained using a RandomForest, XGBoost and CatBoost (the best performing is used in the app). | `stock_condition_model.pkl` |
| **Brand presence models** | One binary classifier per brand (e.g. Pepsi, Fanta, Coca‑Cola) that estimates the probability that the brand is available at a given outlet.  The models are stored in the `brand_presence_models/` subfolder. | `brand_presence_models/*.pkl` |
| **Dominant brand predictor** | A multi‑class classifier that predicts which brand dominates the shelf or fridge at an outlet. | `dominant_brand_model.pkl` |
| **Packaging preference models** | Predict whether an outlet carries PET bottles, glass bottles or cans. | (Models may be added later.) |
| **Footfall estimator** | An experimental regression model that estimates outlet popularity based on brand and packaging variety. | (Experimental – not used in the app.) |
| **Competitor nearest‑neighbour index** | A k‑nearest neighbours model used to find the outlets closest to a given coordinate. | `competitor_nn_model.pkl` |
| **Consumer preference stub** | A placeholder recommender that suggests alternative drinks when a consumer’s favourite is unavailable.  It uses category mappings and brand popularity as a proxy. | `consumer_preference_stub.pkl` |

## Training

The models are trained by running [`train_models.py`](../train_models.py).  This script:

1. Loads the cleaned dataset from `cleaned_product_visibility-2.csv`.
2. Engineers features such as the number of brands present, number of package types and number of display methods.
3. Trains classifiers for stock condition, brand presence and other tasks using scikit‑learn, XGBoost and CatBoost.
4. Saves the fitted models to this directory.  Brand‑specific models are saved under `brand_presence_models/`.

To retrain the models or tune hyperparameters, edit `train_models.py` and run it in your Python environment.  The script prints performance metrics for each model and writes the updated `.pkl` files.

## Usage

In the Streamlit application (`app/app.py`), models are lazily loaded via helper functions.  For example, `load_stock_condition_model()` loads `stock_condition_model.pkl` once and caches it, while `load_brand_presence_models()` loads all brand presence models into a dictionary.  Predictions are made by constructing a small feature vector (outlet type and counts) and calling `.predict()` or `.predict_proba()` on the loaded models.

If you add new models (e.g. packaging or footfall models), make sure to update the helper functions and the app logic accordingly.
