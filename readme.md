# Soft Drink Market Insight Challenge – Analysis & Findings

This document summarizes an exploratory analysis of the Soft Drink Market Insight Challenge data from Alimosho Local Government Area of Lagos, Nigeria. The goal was to clean the raw survey data, derive useful features and visualise patterns in soft‑drink availability, packaging, outlet types and brand presence. Although the dataset captures supply‑side information (what outlets stock and how prominently), careful analysis can still reveal much about consumer demand and market dynamics.


The original data set is saved with the name **"product_visibility_challenge_data.csv"** and the script for cleaning the data set is saved as **"data_cleaning.py".**
After cleaning the data, a file named **"Cleaned_product_visibility.csv"** will be created. This file contains a cleaner version of the data that will be used for analysis and visualizations.

Visualizations will be carried out through streamlit and plotly, as well as pandas and you can find the code in **"visualizations.py".**

## Running the code
To get the most accurate results from this workflow, start by cleaning the data with the **"data_cleaning.py"** script. You can run it in visual studio or on the command line with the command **python data_cleaning.py**.
Next, run the **"visualizations.py"** script but first install dependencies if you don't already have them...

* Step 1: Search for **cmd** on your apps menu on your computer
* Step 2: Open the app called **Command Prompt** that shows up and type the following command **pip install streamlit plotly**
* Step 3: please ensure you are connected to the internet and that you are patient.
* Step 4: Now execute the following command to run the script **streamlit run visualizations.py**

## Data Cleaning & Feature Engineering

The original CSV contained grouping rows, inconsistent column names and several nearly empty fields. Cleaning steps included:

- Skipping the first grouping row when loading the file (header=1).
- Stripping and standardising column names: converting spaces to underscores, fixing misspellings (refridgerator → refrigerator) and retaining meaningful punctuation.
- Splitting the three ambiguous “Others” columns into Product_Others, Display_Others and Package_Others.
- Dropping columns with more than 80 % missing values and those with only a single non‑zero value.
- Converting binary indicator columns to numeric (0/1) and fixing their types.
- Creating several derived features to support analysis:
  - num_brands_present – number of different brands stocked at each outlet.
  - num_package_types – number of distinct package formats (PET, glass, can).
  - brand_variety – a categorical flag (Single, Double, Multiple) based on num_brands_present.
  - dominant_brand – the drink given the most shelf/refrigerator space.

After cleaning, the dataset had 1 500 rows and 30 columns with no missing values in the retained fields. Latitude and longitude were left in raw form for mapping, while the serial number (S/N) was set as the DataFrame index.

## Visualisations & Insights

Below are the key findings from the charts and analyses:

- *Stock condition distribution:* Most outlets are either partially stocked or well stocked. Very few locations are completely out of stock, suggesting good overall supply. About 15 % of outlets fall into the “Almost empty” category, which may point to localized shortages.

![Static snapshot of stock condition distribution chart](plot/stock_condition_distribution.png)

- *Brand presence:* The distribution of brands is highly skewed. Coca‑Cola, Pepsi and Fanta lead the market by a large margin. American Cola and 7Up form a second tier, while drinks like Mountain Dew, RC Cola and Fayrouz are stocked in only a small minority of outlets. This suggests a concentrated competitive landscape with a few dominant players.

![Static snapshot of brand presence chart](plot/brand_presence.png)

- *Packaging type distribution:* Over two‑thirds of all drinks are sold in PET bottles. Glass bottles appear in roughly 20 % of outlets, often alongside PET, while cans make up a tiny share. The dominance of PET likely reflects consumer preference for resealable, portable packaging and cost advantages for retailers.

![Static snapshot of packaging type distribution chart](plot/package_distribution.png)

- *Stock condition by outlet type:* Shops account for the vast majority of outlets and contain the largest number of well‑stocked cases. Kiosks and hawkers show smaller counts and a higher proportion of “Almost empty” statuses. Supermarkets, though few in number, rarely run out of stock. This pattern indicates that larger retail formats maintain deeper inventories.

![Static snapshot of stock condition by outlet chart](plot/stock_condition_by_outlet.png)

- *Brand variety distribution:* Only about 15 % of outlets stock a single brand; most carry two or more. “Multiple” (three or more brands) is the largest category, pointing to a broad competitive set at most locations. Shops are especially likely to offer several brands, while kiosks and hawkers tend toward single or double.

![Static snapshot of brand variety distribution chart](plot/brand_variety_distribution.png)

- *Top product combinations:* The most common assortments include single Coca‑Cola, Coca‑Cola with Fanta, and Pepsi with American Cola. These combinations highlight which brands are frequently purchased together, providing clues about complementary demand or cross‑promotion at the point of sale.

![Static snapshot of top product combinations chart](plot/top_product_combinations.png)

- *Outlet locations:* A scatter plot of outlets by latitude and longitude shows that most points cluster in central Alimosho. Colours indicating stock condition reveal no obvious geographic pattern to stockouts, suggesting that shortages are dispersed rather than concentrated.

![Static snapshot of outlet map chart](plot/outlet_map.png)

## Consumer Trend Inferences

Although the survey doesn’t track individual purchases, it does reveal several demand‑side signals:

- *Dominant brands receive the most shelf space:* Coca‑Cola and Pepsi not only appear most often but are frequently listed as the product with the highest shelf/refrigerator presence. This implies strong consumer pull and retailer confidence.
- *Variety matters:* The majority of outlets stock multiple brands, which suggests that shoppers expect choice and that no single brand completely satisfies local demand.
- *Portable packaging dominates:* The prevalence of PET bottles aligns with on‑the‑go consumption habits typical of busy urban areas. Glass and cans play a secondary role, likely tied to dine‑in or premium experiences.
- *Common brand combinations reflect complementary demand:* The frequent pairing of Coca‑Cola with Fanta and Pepsi with American Cola indicates that consumers often buy these drinks together, prompting retailers to stock them side by side.

## Conclusion

This analysis shows a soft‑drink market dominated by a handful of brands and packaging formats, with retailers largely able to keep products on the shelf. Shops carry the broadest assortments and maintain the healthiest inventories, while smaller formats like kiosks and hawkers exhibit more variability in stock levels. By combining thoughtful data cleaning, feature engineering and visual storytelling, we can infer consumer preferences and market dynamics even from outlet‑level data.

Feel free to explore the interactive dashboard (via the Streamlit app in this repository) for a deeper dive into the data and to experiment with different filters.

Should you have any questions or comments, please reach out to the author of this project by mail: olatundedotun6@gmail.com, olatundedotun@outlook.com
By phone: +234(0) 802 783 4543, 0805 442 5729
By WhatsApp: wa.me/+2348027834543