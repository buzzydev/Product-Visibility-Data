# Streamlit Application

This folder contains the Streamlit app that demonstrates the core functionality of the Product Visibility Challenge.

## Features

The application implements the following flows:

### Sign‑up

* **Consumers** – Supply a display name, email, password, favourite drink, approximate location (selected from high‑level areas and neighbourhoods) and an optional profile picture.  The picture is resized and stored in the database.  Duplicate display names are rejected.
* **Outlet owners** – Provide a display name, email, password, business name, contact information, outlet type (shop/kiosk/mobile) and an optional logo.  Owners also choose the approximate area and neighbourhood of their business and select which brands they stock.  Each outlet is stored along with its chosen coordinates.  Logos are saved in the database if the schema supports it.

### Login

Users log in with their display name and password.  After authentication, Streamlit stores the user in the session state and redirects to the appropriate dashboard.

### Consumer dashboard

* Displays the user’s profile picture, display name, favourite drink and email.
* Calculates the nearest outlet using the stored latitude/longitude from sign‑up.  If the closest outlet is more than 2 km away, it presents an interactive map of all outlets using Plotly’s `scatter_map` and notes that ordering and messaging are “coming soon”.
* If the nearest outlet is within 2 km, it lists the outlet name, type, available brands, a predicted stock condition (using the trained classifier) and a model‑estimated probability that the favourite drink is available (using the brand presence models).  When the favourite brand is unavailable, the app suggests alternatives from the same category and logs the user’s selection.
* Consumers can rate the outlet and leave a comment.  Ratings are stored in the database for later analysis.

### Owner dashboard

* Lists each outlet owned by the user with its logo, name, type and stocked brands.
* Notes that analytics (e.g. stock forecasts, competitor analysis), ordering and messaging features are under development.

## Running the App

1. Ensure that the database is set up and seeded.  See the repository‐level [README](../README.md) for details.
2. Install the dependencies with:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit server from this directory:

```bash
streamlit run app.py
```

4. Navigate to the URL provided by Streamlit (usually <http://localhost:8501>) to interact with the app.

## Limitations

This MVP focuses on demonstrating the data pipeline and basic user flows.  Many features are intentionally left as placeholders, including ordering drinks, messaging outlet owners, hawker tracking, real‑time location services and personalised recommendations that adapt over time.  The database schema can be extended to support profile pictures and contact information; see the comments in the code for guidance.
