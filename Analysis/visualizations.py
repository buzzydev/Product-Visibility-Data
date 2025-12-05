"""
visualizations.py
-------------------

This Streamlit app loads the cleaned soft‑drink visibility dataset, derives a
few additional features to aid exploration, and presents a set of interactive
charts for visualizing the data. The aim is to uncover patterns in outlet
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


# Configuration
# Path to the cleaned dataset (update this if your file lives elsewhere)

DATA_PATH = "../Files/cleaned_product_visibility.csv"

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
    """Return a scatter map chart showing outlet locations colored by a category.

    This helper uses Plotly Express' ``scatter_map`` function, which avoids the
    deprecated ``scatter_mapbox`` API. It still leverages Mapbox for rendering
    but does so via the updated interface.
    """
    # Create a consistent colour palette for categories
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
    
    # Adjust layout to control height and legend positioning
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
    # Compute percentages for narrative
    total_outlets = len(filtered_df)
    stock_percent = (stock_counts / total_outlets * 100).round(1)
    st.plotly_chart(bar_chart(stock_counts, "Stock Condition Distribution", "Stock Condition", "Number of Outlets"))
    st.markdown(
        f"About **{stock_percent.get('Partially stocked', 0):.1f}%** of outlets are partially stocked and **{stock_percent.get('Well stocked', 0):.1f}%** are well stocked. "
        f"Only **{stock_percent.get('Almost empty', 0):.1f}%** are almost empty, **{stock_percent.get('Out of stock', 0):.1f}%** are out of stock, "
        f"and the remaining **{stock_percent.get('Not Applicable', 0):.1f}%** were not applicable."
    )

    # Brand presence counts
    st.subheader("Brand Presence Across Outlets")
    brand_counts = filtered_df[PRODUCT_COLS].sum().sort_values(ascending=False)
    brand_percent = (brand_counts / total_outlets * 100).round(1)
    st.plotly_chart(bar_chart(brand_counts, "Brand Presence Across Outlets", "Brand", "Number of Outlets"))
    # Compose a short narrative for the top three brands
    top3 = brand_counts.head(3).index.tolist()
    narrative = []
    for brand in top3:
        narrative.append(f"**{brand.replace('_', ' ')}** ({brand_percent[brand]:.1f}%)")
    st.markdown(
        "The market is highly concentrated: " + ", ".join(narrative) + " lead by a wide margin, while other brands trail far behind."
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
    variety_percent = (variety_counts / total_outlets * 100).round(1)
    # Use a pie chart to emphasise proportions of single, double and multiple brand outlets
    fig_variety_pie = px.pie(
        names=variety_counts.index,
        values=variety_counts.values,
        title="Brand Variety Categories",
        labels={"names": "Variety Category", "values": "Number of Outlets"},
    )
    st.plotly_chart(fig_variety_pie)
    st.markdown(
        f"Single‑brand outlets account for {variety_percent.get('Single', 0):.1f}% of the market, double‑brand outlets for {variety_percent.get('Double', 0):.1f}%, "
        f"and multi‑brand outlets for {variety_percent.get('Multiple', 0):.1f}%."
    )
    # ------------------------------------------------------------------
    # Additional Insights
    # ------------------------------------------------------------------

    # Packaging analysis with interactive view selector
    st.subheader("Packaging Analysis")
    pack_view = st.selectbox(
        "Select packaging view",
        ["Overall Distribution", "By Outlet Type", "By Top Brands", "By Dominant Brands"],
        index=0,
        help="Explore packaging formats across different perspectives"
    )
    if pack_view == "Overall Distribution":
        package_counts = filtered_df[PACKAGE_COLS].sum().sort_values(ascending=False)
        package_percent = (package_counts / total_outlets * 100).round(1)
        fig_pack = px.pie(
            names=package_counts.index,
            values=package_counts.values,
            title="Packaging Type Distribution",
            labels={"names": "Packaging Type", "values": "Number of Outlets"},
        )
        st.plotly_chart(fig_pack)
        st.markdown(
            f"PET bottles account for **{package_percent.get('PET_Bottle_(50cl/1L)', 0):.1f}%** of all packages. "
            f"Glass bottles represent {package_percent.get('Glass_Bottle_(35cl/60cl)', 0):.1f}%, and cans make up {package_percent.get('Can_(33cl)', 0):.1f}%."
        )
    elif pack_view == "By Outlet Type":
        # Sum packaging types for each outlet type
        pack_outlet = filtered_df.groupby("Type_Of_Outlet")[PACKAGE_COLS].sum()
        fig_pack_outlet = px.bar(
            pack_outlet,
            x=pack_outlet.index,
            y=pack_outlet.columns,
            barmode="stack",
            labels={"value": "Number of Outlets", "index": "Outlet Type", "variable": "Packaging Type"},
            title="Packaging Formats by Outlet Type",
        )
        st.plotly_chart(fig_pack_outlet)
        st.markdown(
            "This view compares packaging mixes across outlet types. Shops dominate PET usage, while kiosks and hawkers show smaller counts across all formats."
        )
    elif pack_view == "By Top Brands":
        # Recompute brand counts on the filtered data to rank brands
        brand_counts_local = filtered_df[PRODUCT_COLS].sum().sort_values(ascending=False)
        top_brands = brand_counts_local.head(5).index
        packaging_by_brand = {}
        for brand in top_brands:
            packaging_by_brand[brand] = filtered_df.loc[filtered_df[brand] == 1, PACKAGE_COLS].sum()
        packaging_df = pd.DataFrame(packaging_by_brand).T
        fig_pack_brand = px.bar(
            packaging_df,
            x=packaging_df.index,
            y=packaging_df.columns,
            barmode="group",
            labels={"value": "Number of Outlets", "x": "Brand", "variable": "Packaging Type"},
            title="Packaging Formats by Top Brands",
        )
        st.plotly_chart(fig_pack_brand)
        st.markdown(
            "Leading brands rely heavily on PET bottles. Glass and cans appear only in niche quantities across the top sellers."
        )
    else:  # By Dominant Brands
        # Use dominant brand mapping defined earlier
        dom_counts = filtered_df.get("dominant_brand", pd.Series(dtype=str)).value_counts()
        # Filter to known product names
        dominance_map_local = {
            "Coke": "Coca_Cola", "Coca Cola": "Coca_Cola", "Coca-Cola": "Coca_Cola",
            "Pepsi": "Pepsi", "Bigi": "Bigi", "Rc Cola": "RC_Cola", "Rc": "RC_Cola", "7Up": "7Up",
            "Fanta": "Fanta", "Sprite": "Sprite", "La Casera": "La_Casera", "Schweppes": "Schweppes",
            "Fayrouz": "Fayrouz", "Mirinda": "Mirinda", "Mountain Dew": "Mountain_Dew", "Teem": "Teem",
            "American Cola": "American_Cola", "Others": "Product_Others",
        }
        valid_dom = [b for b in dom_counts.index if b in dominance_map_local]
        top_dom_brands = pd.Index(valid_dom)[:3]
        if len(top_dom_brands) > 0:
            pack_by_dom = {}
            for dom in top_dom_brands:
                prod_col = dominance_map_local[dom]
                mask = (filtered_df.get("dominant_brand") == dom) & (filtered_df[prod_col] == 1)
                pack_by_dom[dom] = filtered_df.loc[mask, PACKAGE_COLS].sum()
            pack_dom_df = pd.DataFrame(pack_by_dom).T
            fig_pack_dom = px.bar(
                pack_dom_df,
                x=pack_dom_df.index,
                y=pack_dom_df.columns,
                barmode="group",
                labels={"value": "Number of Outlets", "x": "Dominant Brand", "variable": "Packaging Type"},
                title="Packaging Formats for Top Dominant Brands",
            )
            st.plotly_chart(fig_pack_dom)
            # Build narrative
            expl_lines = []
            for dom in top_dom_brands:
                total = pack_dom_df.loc[dom].sum()
                if total > 0:
                    perc_pet = pack_dom_df.loc[dom, "PET_Bottle_(50cl/1L)"] / total * 100 if "PET_Bottle_(50cl/1L)" in pack_dom_df.columns else 0
                    perc_glass = pack_dom_df.loc[dom, "Glass_Bottle_(35cl/60cl)"] / total * 100 if "Glass_Bottle_(35cl/60cl)" in pack_dom_df.columns else 0
                    perc_can = pack_dom_df.loc[dom, "Can_(33cl)" ] / total * 100 if "Can_(33cl)" in pack_dom_df.columns else 0
                    expl_lines.append(
                        f"When **{dom}** is the dominant brand, it is sold mostly in PET bottles (about {perc_pet:.1f}%); "
                        f"glass bottles ({perc_glass:.1f}%) and cans ({perc_can:.1f}%) play minor roles."
                    )
            st.markdown(
                "This view focuses on outlets where a brand holds the prime shelf space. "
                "It shows that even dominant brands rely overwhelmingly on PET bottles.nn" + "n".join(expl_lines)
            )

    # Relationship between brand variety and package variety
    st.subheader("Brand Variety vs Packaging Variety")
    avg_pkg_per_brand_variety = (
        filtered_df.groupby("brand_variety")["num_package_types"].mean().reindex(["Single", "Double", "Multiple"])
    )
    fig_variety_pkg = px.bar(
        x=avg_pkg_per_brand_variety.index,
        y=avg_pkg_per_brand_variety.values,
        labels={"x": "Brand Variety", "y": "Avg. Package Types"},
        title="Average Number of Package Types by Brand Variety",
    )
    fig_variety_pkg.update_layout(xaxis_title="Brand Variety", yaxis_title="Average Package Types")
    st.plotly_chart(fig_variety_pkg)
    st.markdown(
        "Outlets that stock **multiple** brands also tend to offer more packaging formats, whereas single‑brand outlets typically sell only one or two. "
        "This suggests that variety at the brand level often goes hand‑in‑hand with variety in packaging."
    )
    st.subheader("Brand Co‑occurrence Matrix")
    co_occ = filtered_df[PRODUCT_COLS].T.dot(filtered_df[PRODUCT_COLS])
    # Zero out the diagonal to focus on co‑occurrences only
    for b in co_occ.columns:
        co_occ.loc[b, b] = 0
    fig_coocc = px.imshow(
        co_occ,
        labels=dict(x="Brand", y="Brand", color="Co‑occurrence Count"),
        x=co_occ.columns,
        y=co_occ.index,
        title="Co‑occurrence of Brands Across Outlets",
        color_continuous_scale="Blues",
        text_auto=True,
    )
    # Improve layout for readability
    fig_coocc.update_layout(xaxis_title="Brand", yaxis_title="Brand")
    st.plotly_chart(fig_coocc)
    st.markdown(
        "This heatmap shows how often each pair of brands appears together in the same outlet. "
        "Darker shades indicate more common pairings, highlighting which drinks are frequently stocked together and may reflect complementary demand."
    )

    # ------------------------------------------------------------------
    # Brand presence vs visibility
    # ------------------------------------------------------------------
    st.subheader("Brand Presence vs Visibility")
    # Compute presence counts for each product (number of outlets stocking the brand)
    presence_counts = filtered_df[PRODUCT_COLS].sum().sort_values(ascending=False)
    # Map values in the dominant_brand column to product column names where possible
    dominance_map = {
        "Coke": "Coca_Cola",
        "Coca Cola": "Coca_Cola",
        "Coca-Cola": "Coca_Cola",
        "Pepsi": "Pepsi",
        "Bigi": "Bigi",
        "Rc Cola": "RC_Cola",
        "Rc": "RC_Cola",
        "7Up": "7Up",
        "Fanta": "Fanta",
        "Sprite": "Sprite",
        "La Casera": "La_Casera",
        "Schweppes": "Schweppes",
        "Fayrouz": "Fayrouz",
        "Mirinda": "Mirinda",
        "Mountain Dew": "Mountain_Dew",
        "Teem": "Teem",
        "American Cola": "American_Cola",
        "Others": "Product_Others",
    }
    # Count how many times each product is the dominant brand
    dominance_counts = pd.Series(0, index=presence_counts.index)
    # Only compute if the dominant_brand column exists
    if "dominant_brand" in filtered_df.columns:
        for val, count in filtered_df["dominant_brand"].value_counts().items():
            mapped = dominance_map.get(val, None)
            if mapped in dominance_counts.index:
                dominance_counts[mapped] += count
    # Combine presence and dominance into one DataFrame for plotting (top 5 brands)
    top_avail = presence_counts.head(5)
    avail_vis_df = pd.DataFrame({
        "Presence": top_avail,
        "Dominant": dominance_counts[top_avail.index],
    })
    fig_avail_vis = px.bar(
        avail_vis_df,
        x=avail_vis_df.index,
        y=["Presence", "Dominant"],
        barmode="group",
        labels={"value": "Number of Outlets", "x": "Brand", "variable": "Metric"},
        title="Brand Presence vs Visibility for Top Brands",
    )
    st.plotly_chart(fig_avail_vis)
    # Provide narrative with ratios
    narrative_lines = []
    for brand in avail_vis_df.index:
        presence = avail_vis_df.loc[brand, "Presence"]
        dominant = avail_vis_df.loc[brand, "Dominant"]
        if presence > 0:
            ratio = dominant / presence * 100
            narrative_lines.append(
                f"For **{brand.replace('_', ' ')}**, the drink is stocked in {presence} outlets but is dominant in {dominant} of them (about {ratio:.1f}%)."
            )
    # Build explanatory text for presence vs visibility
    explanation_text = (
        "This chart compares how often each top brand is **present** versus how often it holds the **prime shelf or refrigerator spot**. "
        "A high dominance percentage suggests retailers prioritise that brand in their displays, whereas a lower percentage indicates "
        "that the brand is often stocked but rarely given prominence.nn"
    )
    st.markdown(explanation_text + "n".join(narrative_lines))


    # ------------------------------------------------------------------
    # Display methods by outlet type
    # ------------------------------------------------------------------
    st.subheader("Display Methods by Outlet Type")
    # Sum each display method across outlets grouped by type
    display_by_outlet = filtered_df.groupby("Type_Of_Outlet")[DISPLAY_COLS].sum()
    fig_display_outlet = px.bar(
        display_by_outlet,
        x=display_by_outlet.index,
        y=display_by_outlet.columns,
        barmode="stack",
        title="Display Methods by Outlet Type",
        labels={"value": "Number of Outlets", "index": "Outlet Type", "variable": "Display Method"},
    )
    st.plotly_chart(fig_display_outlet)
    # Build narrative highlighting differences in display methods across outlet types
    disp_narr = []
    for outlet in display_by_outlet.index:
        total_disp = display_by_outlet.loc[outlet].sum()
        if total_disp > 0:
            perc_shelf = display_by_outlet.loc[outlet, "On_Shelf/Carton"] / total_disp * 100 if "On_Shelf/Carton" in display_by_outlet.columns else 0
            perc_refrig = display_by_outlet.loc[outlet, "In_Refrigerator/Cooler"] / total_disp * 100 if "In_Refrigerator/Cooler" in display_by_outlet.columns else 0
            perc_stand = display_by_outlet.loc[outlet, "On_Display_Stand"] / total_disp * 100 if "On_Display_Stand" in display_by_outlet.columns else 0
            disp_narr.append(
                f"**{outlet}** outlets use shelves/cartons for about {perc_shelf:.1f}% of displays, refrigerators/coolers for {perc_refrig:.1f}%, "
                f"and display stands for {perc_stand:.1f}%."
            )
    st.markdown(
        "This chart compares how different outlet types showcase products. "
        "Shops and supermarkets rely heavily on shelves and refrigerators, whereas kiosks and hawkers rarely use refrigerators or display stands.nn"
        + "n".join(disp_narr)
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
    # Location map with density option
    st.subheader("Outlet Locations")
    map_color_option = st.selectbox(
        "Color outlets by", ["Stock_Condition", "Type_Of_Outlet"]
    )
    map_view_option = st.radio(
        "Map view", ["Scatter", "Density"], horizontal=True,
        help="Choose a scatter map of individual outlets or a density heatmap of outlet concentration."
    )
    if map_view_option == "Scatter":
        fig_map = scatter_map(
            filtered_df, color_col=map_color_option, title=f"Outlet Locations Colored by {map_color_option}"
        )
        st.plotly_chart(fig_map)
        st.markdown(
            "This map shows the location of every outlet. Use the drop‑down to colour points by **Stock Condition** or **Type of Outlet**."
        )
    else:
        # Density heatmap using Plotly Express. z=None counts each point equally
        fig_density = px.density_map(
            filtered_df,
            lat="Latitude",
            lon="Longitude",
            z=None,
            radius=15,
            center={"lat": filtered_df["Latitude"].mean(), "lon": filtered_df["Longitude"].mean()},
            zoom=11,
            mapbox_style="carto-positron",
            title="Outlet Density Heatmap",
        )
        fig_density.update_layout(height=500)
        st.plotly_chart(fig_density)
        st.markdown(
            "This density heatmap highlights areas with a high concentration of outlets. Darker spots correspond to clusters of retailers, "
            "while lighter areas indicate sparser coverage."
        )

    st.write("n")
    st.caption("Data source: Soft Drink Market Insight Challenge (Alimosho LGA, Lagos, Nigeria)")


if __name__ == "__main__":
    main()