"""
visualizations.py
-------------------

This Streamlit app loads the cleaned soft‑drink visibility dataset, derives a
few additional features to aid exploration, and presents a set of interactive
charts that help to visualize the data. The aim is to help you uncover patterns in outlet
types, stock conditions, brand presence, packaging formats and more. Running
this script with ``streamlit run visualizations.py`` will launch the
dashboard.

Requirements:
    - streamlit
    - pandas
    - plotly

Install missing dependencies with::

    pip install streamlit plotly

Usage:
    streamlit run visualizations.py

Note: This script assumes ``cleaned_product_visibility.csv`` is located
alongside it. If your data is stored elsewhere, update the ``DATA_PATH``
variable accordingly.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to the cleaned dataset (update this if your file lives elsewhere)
DATA_PATH = "cleaned_product_visibility.csv"

# Columns grouped by logical category. These lists will be used both for
# derived feature calculations and for building plots.
PRODUCT_COLS = [
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
    # Note: Product_Others indicates any other drink not listed
    "Product_Others",
]

PACKAGE_COLS = [
    "PET_Bottle_(50cl/1L)",
    "Glass_Bottle_(35cl/60cl)",
    "Can_(33cl)",
]

DISPLAY_COLS = [
    "On_Shelf/Carton",
    "In_Refrigerator/Cooler",
    "On_Display_Stand",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset and create derived features.

    The function is cached by Streamlit to avoid reloading the CSV on every
    interaction.

    Args:
        path: Path to the CSV file containing the cleaned data.

    Returns:
        Pandas DataFrame with additional columns for brand count, package count,
        display count, and a boolean indicating multiple brands.
    """
    df = pd.read_csv(path, index_col=0)
    # Ensure binary columns are numeric
    df[PRODUCT_COLS + PACKAGE_COLS + DISPLAY_COLS] = df[
        PRODUCT_COLS + PACKAGE_COLS + DISPLAY_COLS
    ].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Derived features
    df["num_brands_present"] = df[PRODUCT_COLS].sum(axis=1)
    df["num_package_types"] = df[PACKAGE_COLS].sum(axis=1)
    df["num_display_methods"] = df[DISPLAY_COLS].sum(axis=1)
    df["is_multiple_brands"] = df["num_brands_present"] > 1

    # Brand variety category: Single, Double, Multiple (3+)
    def brand_variety(n: int) -> str:
        if n <= 1:
            return "Single"
        elif n == 2:
            return "Double"
        else:
            return "Multiple"
    df["brand_variety"] = df["num_brands_present"].apply(brand_variety)

    # Clean dominant brand column if present
    if "Product_With_Higher_Shelf/Refrigerator_Presence" in df.columns:
        df["dominant_brand"] = df[
            "Product_With_Higher_Shelf/Refrigerator_Presence"
        ].fillna("Unknown").str.title()
    return df


def bar_chart(series: pd.Series, title: str, x_label: str, y_label: str):
    """Return a Plotly bar chart from a Pandas Series."""
    fig = px.bar(
        x=series.index,
        y=series.values,
        labels={"x": x_label, "y": y_label},
        title=title,
    )
    # Increase the font size for readability
    fig.update_layout(font=dict(size=14))
    return fig


def stacked_bar(df: pd.DataFrame, index_col: str, cat_col: str, title: str):
    """Return a stacked bar chart for a cross-tab of two categorical columns."""
    cross = pd.crosstab(df[index_col], df[cat_col])
    cross = cross.reindex(cross.sum(axis=1).sort_values(ascending=False).index)
    fig = px.bar(
        cross,
        x=cross.index,
        y=cross.columns,
        title=title,
        labels={"value": "Count", "index": index_col, "variable": cat_col},
    )
    fig.update_layout(barmode="stack", xaxis_title=index_col, yaxis_title="Count")
    return fig


def scatter_map(df: pd.DataFrame, color_col: str, title: str):
    """Return a scatter mapbox chart showing outlet locations colored by a category."""
    # Create a color discrete map for consistency across categories
    unique_vals = df[color_col].unique()
    colors = px.colors.qualitative.Set2
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}
    fig = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        color=color_col,
        color_discrete_map=color_map,
        hover_name=color_col,
        zoom=11,
        title=title,
    )
    # Set map style and layout properties separately to avoid deprecated parameters
    fig.update_layout(
        mapbox_style="carto-positron",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main():
    st.set_page_config(page_title="Soft Drink Market Insights", layout="wide")
    st.title("Soft Drink Market Insights Dashboard")

    # Load data and create derived columns
    df = load_data(DATA_PATH)

    # Sidebar filters
    st.sidebar.header("Filters")
    outlet_filter = st.sidebar.multiselect(
        "Select outlet types", options=sorted(df["Type_Of_Outlet"].unique()), default=None
    )
    stock_filter = st.sidebar.multiselect(
        "Select stock conditions", options=sorted(df["Stock_Condition"].unique()), default=None
    )

    # Apply filters
    filtered_df = df.copy()
    if outlet_filter:
        filtered_df = filtered_df[filtered_df["Type_Of_Outlet"].isin(outlet_filter)]
    if stock_filter:
        filtered_df = filtered_df[filtered_df["Stock_Condition"].isin(stock_filter)]

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Outlets", len(filtered_df))
    col2.metric("Average Brands per Outlet", round(filtered_df["num_brands_present"].mean(), 1))
    col3.metric("Average Package Types", round(filtered_df["num_package_types"].mean(), 1))
    col4.metric("Multi‑brand Outlets", int(filtered_df["is_multiple_brands"].sum()))

    # Stock condition distribution
    st.subheader("Stock Condition Distribution")
    stock_counts = filtered_df["Stock_Condition"].value_counts().sort_values(ascending=False)
    st.plotly_chart(bar_chart(stock_counts, "Stock Condition Distribution", "Stock Condition", "Number of Outlets"))
    st.markdown(
        "Most outlets are **partially stocked** or **well stocked**, with only a small number either **almost empty** or **out of stock**."
    )

    # Brand presence counts
    st.subheader("Brand Presence Across Outlets")
    brand_counts = filtered_df[PRODUCT_COLS].sum().sort_values(ascending=False)
    st.plotly_chart(bar_chart(brand_counts, "Brand Presence Across Outlets", "Brand", "Number of Outlets"))
    st.markdown(
        "Coca‑Cola, Pepsi, and Fanta dominate the market, while other brands appear in far fewer outlets."
    )

    # Packaging distribution
    st.subheader("Packaging Type Distribution")
    package_counts = filtered_df[PACKAGE_COLS].sum().sort_values(ascending=False)
    st.plotly_chart(bar_chart(package_counts, "Packaging Type Distribution", "Packaging", "Number of Outlets"))
    st.markdown(
        "Most drinks are sold in **PET bottles**. Glass bottles and cans are much less common."
    )

    # Crosstab of outlet type vs stock condition
    st.subheader("Stock Condition by Outlet Type")
    st.plotly_chart(
        stacked_bar(
            filtered_df,
            index_col="Type_Of_Outlet",
            cat_col="Stock_Condition",
            title="Stock Condition by Outlet Type",
        )
    )
    st.markdown(
        "Shops have the highest counts across all stock conditions. Other outlet types like kiosks and hawkers show much smaller numbers."
    )

    # Crosstab of outlet type vs multiple brands
    st.subheader("Multi‑brand Presence by Outlet Type")
    multi_brand_counts = pd.crosstab(
        filtered_df["Type_Of_Outlet"], filtered_df["is_multiple_brands"].map({True: "Multiple Brands", False: "Single Brand"})
    )
    fig_multi = px.bar(
        multi_brand_counts,
        x=multi_brand_counts.index,
        y=multi_brand_counts.columns,
        barmode="stack",
        title="Multi‑brand vs Single‑brand Outlets by Outlet Type",
        labels={"value": "Number of Outlets", "index": "Outlet Type", "variable": "Brand Variety"},
    )
    st.plotly_chart(fig_multi)
    st.markdown(
        "Most outlets offer more than one brand, especially shops. Single‑brand outlets are relatively uncommon."
    )

    # Brand variety distribution (Single, Double, Multiple)
    st.subheader("Brand Variety Categories")
    variety_counts = filtered_df["brand_variety"].value_counts().sort_index()
    st.plotly_chart(
        bar_chart(variety_counts, "Brand Variety Distribution", "Variety Category", "Number of Outlets")
    )
    st.markdown(
        "This chart categorises outlets as **Single**, **Double**, or **Multiple** based on how many brands they stock. You can see that multi‑brand outlets are the most common."
    )

    # Top 10 product combinations
    st.subheader("Top Product Combinations")
    combo_counts = (
        filtered_df["Type_Of_Product_(Combined_Response)"].value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    # Plot as a horizontal bar chart for readability
    fig_combo = px.bar(
        x=combo_counts.values,
        y=combo_counts.index,
        orientation="h",
        labels={"x": "Number of Outlets", "y": "Product Combination"},
        title="Top 10 Product Combinations",
    )
    fig_combo.update_layout(font=dict(size=14), yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_combo)
    st.markdown(
        "The most common product assortments include single Coca‑Cola, Coca‑Cola with Fanta, and Pepsi with American Cola, among others."
    )
    # Location scatter map
    st.subheader("Outlet Locations")
    map_color_option = st.selectbox(
        "Color outlets by", ["Stock_Condition", "Type_Of_Outlet"]
    )
    st.plotly_chart(
        scatter_map(filtered_df, color_col=map_color_option, title=f"Outlet Locations Colored by {map_color_option}")
    )
    st.markdown(
        "This map shows where each outlet is located. You can colour the points by **Stock Condition** or **Type of Outlet** using the drop‑down above."
    )

    st.write("\n")
    st.caption("Data source: Soft Drink Market Insight Challenge (Alimosho LGA, Lagos, Nigeria)")


if __name__ == "__main__":
    main()