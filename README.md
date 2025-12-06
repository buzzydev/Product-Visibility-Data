# Introduction

I am *Dotun Olatunde* and this is an end to end data processing workflow that started with a simple data analysis challenge by *@getnervs* on X, formerly known as Twitter.

# Project Overview

This repository contains the end‑to‑end work for the **Product Visibility Challenge**.  The goal of the project is to explore, clean and model a dataset of beverage outlets in Lagos, Nigeria, and to build a prototype application that helps consumers find their favourite drinks and helps outlet owners understand their product mix.

## Contents

The project is organised into several subfolders:

| Folder | Description |
|-------|------------|
| [`Analysis`](Analysis/) | Scripts and visuals used for exploratory data analysis (EDA) and visualisation. |
| [`App`](App/) | A Streamlit application that implements user and outlet owner sign‑up, login, and interactive dashboards. It uses trained machine‑learning models to estimate stock conditions and brand availability. |
| [`Models`](models/) | Pre‑trained model artifacts (e.g. `stock_condition_model.pkl`, `brand_presence_models/`) and training scripts (`train_models.py`). |
| [`Database`](App/Database/) | SQLAlchemy setup and migration scripts used to create the PostgreSQL schema. |

| [`Files`](Files/) | Raw and cleaned data used for data analysis and model training. |
## Getting Started

1. **Data cleaning and EDA** – See the scripts in [`Analysis`](Analysis/) to understand how the raw CSV was cleaned and what insights were derived from the data.
2. **Model training** – The script [`train_models.py`](Models/train_models.py) loads the cleaned dataset, engineers features and trains a variety of classifiers and clustering models.  The resulting `.pkl` files are stored in the `models/` folder.
3. **Database setup** – Define your database credentials in `database/database.py` or via environment variables.  Then run [`Database/create_db.py`](Database/create_db.py) to create the tables.  Use [`seed_brands.py`](App/Database/seed_brands.py) to populate the `brands` table.
4. **Streamlit app** – Navigate to [`App`](App/) and install dependencies listed in [`requirements.txt`](App/requirements.txt).  Run `streamlit run app/app.py` or click *[here](https://soda-spot.streamlit.app)* to launch the prototype.  Consumers can sign up, log in, see nearby outlets, get recommendations and leave ratings.  Owners can register their businesses, upload a logo and view a basic dashboard of their outlets.

## Status and Future Work

This repository demonstrates a complete workflow from raw data to a working prototype.  Not all planned features are fully implemented.  The app includes placeholders for ordering drinks, messaging outlet owners, real‑time hawker alerts and advanced analytics.  See the extensive comments in [`App/app.py`](App/app.py) and the model scripts for guidance on how to extend the project.
