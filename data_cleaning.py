"""Dotun's Data Cleaning and Pre-Processing Workflow"""

#Importing required libraries

print("Importing required libraries...")

import pandas as pd
import numpy as np
import re

#Loading the data into a dataframe

print("Loading raw data...")

data = pd.read_csv("product_visibility_challenge_data.csv")
frame = pd.DataFrame(data)

print("Data loaded successfully...")

#Take a quick look at the raw data structure
print("Initial data set shape:", frame.shape)
#print(frame.head())

"""The data set has the real column names inside the first row (index 0)
 So we extract that row and assign it as the new header."""

print("Setting header...")

# Save the first row as header

new_header = frame.iloc[0]

# Remove that header row from the data
frame = frame[1:]

# Assign the header
frame.columns = new_header

# Reset the index so the table starts at 0 again
frame.reset_index(drop=True, inplace=True)
print("Header is set successfully...")
#print(frame.head())


#We need to rename duplicate columns so that we won't be getting an attribute error later in the code
print("Renaming duplicate columns...")

# First strip any stray spaces from column names
frame.columns = [c.strip() for c in frame.columns]

# Find the indices of all columns that are exactly named "Others"
duplicate_positions = [i for i, col in enumerate(frame.columns) 
                       if col == 'Others']

# Define the new names in the order you want them applied
new_names = ['Product_Others', 'Display_Others', 'Package_Others']

# Assign the new names by position
for pos, new_name in zip(duplicate_positions, new_names):
    frame.columns.values[pos] = new_name

# Verify the result
print("Duplicates renamed successfully...")
print(frame.columns.tolist())


#Rename specific columns
# Check whether 'Package_Type_Combined' is present

if 'Package Type Combined' in frame.columns:
    frame = frame.rename(columns={
        'Package Type Combined': 'Package_Type_(Combined_Response)'
    })

# Verify the rename
print('Package' in frame.columns)  # Should return True for the new name


"""We are going to be dropping some columns, columns that don't hold any value or columns with high missing values based on the importance or otherwise of the column to our data analysis"""

print("Dropping useless columns...")

#Check the columns with missing values
print(frame.isna().sum().sort_values(ascending = False))

"""I think we should drop the first five.
I would have said we should leave the "Product with Higher shelf/refrigerator presence - Others" but it will introduce a lot of noise in our data analysis so we just have to let it go
Also, we are dropping the "With consumer (actively being consumed) because its just filled with lots of zeros and NaNs.""
Lastly, maybe let's drop the "Package_Others" because its only one (1) digit thats there and it doesn't hold so much value to our analysis."
It would just cause a lot of noise"""

#Check these out
#print(frame["With consumer (actively being consumed)"].unique())
#print(frame["Package_Others"].value_counts())
drop_cols = ["Package_Others", ]
#Drop columns with more than 95% missing values
for col in frame.columns:
    missing_ratio = frame[col].isna().mean()
    if missing_ratio > 0.80:
        drop_cols.append(col)

#Drop columns with only one unique non-null value
for col in frame.columns:
    if frame[col].nunique(dropna=True) == 1:
        # constant value across all rows
        drop_cols.append(col)

#Drop binary columns that are effectively unused
for col in frame.columns:
    # check if the column is numeric and has only two possible values
    if pd.api.types.is_numeric_dtype(frame[col]):
        unique_vals = set(frame[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            # if there are 0 or 1 occurrences of the '1' value, drop it
            if unique_vals in frame[col].sum() <= 1:
                drop_cols.append(col)

# Remove any duplicates in drop_cols
drop_cols = list(set(drop_cols))

#print("Columns to drop:", drop_cols)

# Drop them from the DataFrame
frame = frame.drop(columns=drop_cols)

print("Redundant columns dropped successfuly...")

# Verify
#print("Remaining columns:", frame.columns.tolist())

#Cleaning column names
print("Cleaning column names...")

def refine_col(col):
    col = col.strip()
    # Replace hyphens and spaces with underscores
    col = col.replace('-', '_').replace(' ', '_')
    # Correct common misspelling
    col = col.replace('refridgerator', 'refrigerator')
    # Remove any trailing underscore
    col = col.rstrip('_')
    # Capitalize each word segment separated by underscores or slashes
    parts = re.split(r'(_|/)', col)  # keep delimiters
    refined_parts = []
    for part in parts:
        if part not in ['_', '/'] and part and part[0].isalpha():
            refined_parts.append(part[0].upper() + part[1:])
        else:
            refined_parts.append(part)
    return ''.join(refined_parts)

# Apply the column-refinement function to every column
frame.columns = [refine_col(c) 
                 for c in frame.columns]

print("Column names cleaned successfully...")

# Review the cleaned column names
#print(frame.columns.tolist())


"""Many product indicator columns like "Coca_Cola", "RC_Cola", "Bigi", etc are stored as strings so we need to convert them to numbers."""

# Step 1: Identify binary columns
binary_cols = []

for col in frame.columns:

    # Get the unique non-null values    
    unique_vals = frame[col].dropna().unique()
    unique_numeric = set()
    # Try to convert each unique value to a number
    for val in unique_vals:
        try:
            unique_numeric.add(int(val))
        except Exception:
            # If conversion fails, include the raw value
            unique_numeric.add(val)
    # If the numeric set contains only 0 and 1, it’s binary
    if unique_numeric.issubset({0, 1}):
        binary_cols.append(col)

# Step 2: Convert detected binary columns to integer type and print them
frame[binary_cols] = frame[binary_cols].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).astype(int))
print("Binary columns detected and converted:", binary_cols)


# Standardize latitude and longitude (z-score).
#Although not necessary for analysis, I'm planning to do predictive modelling later so I'll just keep it 

for col in ['Latitude', 'Longitude']:
    #Ensure they are floats
    frame[col] = pd.to_numeric(frame[col], errors='coerce')

    #Then standardize values
    #frame[col] = (frame[col] - frame[col].mean())/frame[col].std()

    #print(frame[["Longitude", "Latitude"]])


print("Still processing data...")
print("Now cleaning categorical columns...")


#Let's clean the categorical/text columns

    #We will start by cleaning up the values in Product_With_Higher_Shelf/Refrigerator_Presence:
    #Side note: We can use pd.get_dummies and one hot encoding later if we wish to carry out predictive modelling

col = 'Product_With_Higher_Shelf/Refrigerator_Presence'
frame[col] = (
    frame[col]
    .fillna('Unknown')          # ensure no missing values
    .str.strip()                # remove leading/trailing whitespace
    .str.title()                # capitalize each word: 'all equal' -> 'All Equal'
)

# Check the standardized categories
print(frame[col].value_counts())


#Let's clean the column Type of Outlet
# Inspect the original values (optional, but useful for understanding)
#print("Original unique values and counts:")
#print(frame['Type_Of_Outlet'].value_counts())

#Clean the text: remove leading/trailing spaces and capitalize words
frame['Type_Of_Outlet'] = (
    frame['Type_Of_Outlet']
    .str.strip()  # remove any extra whitespace
    .str.title()  # convert to title case: e.g. 'open market stall' -> 'Open Market Stall'
)

# Verify the results
#print("\nCleaned unique values and counts:")
#print(frame['Type_Of_Outlet'].value_counts())


print("Still processing data, please wait...")


#Let's continue cleaning type of product
#Because there are more than one entry per row of the column, we need to split each entry to show that they are independent, even though they are together within the same field
#We will also do something similar for display and packaging columns

# Map of multi-word products to temporary single-word tokens
multi_word_map = {
    'American Cola': 'American_Cola',
    'RC Cola': 'RC_Cola',
    'La Casera': 'La_Casera',
    'Mountain Dew': 'Mountain_Dew'
}

def clean_product_combination(entry):
    if isinstance(entry, str):
        temp = entry.strip()
        # Replace multi-word products with a single token
        for k, v in multi_word_map.items():
            temp = temp.replace(k, v)
        # Now split on spaces and strip each part
        parts = [p.strip() for p in temp.split() if p.strip()]
        # Convert to title case and restore spaces from underscores
        cleaned = [p.title().replace('_', ' ') for p in parts]
        # Rejoin with comma and space
        return ', '.join(cleaned)
    return entry

# Apply the cleaning function in place (no new column created)
frame['Type_Of_Product_(Combined_Response)'] = (
    frame['Type_Of_Product_(Combined_Response)']
    .apply(clean_product_combination)
)

# Examine the result
#print(frame['Type_Of_Product_(Combined_Response)'].head())
#print(frame["Type_Of_Product_(Combined_Response)"].value_counts())


#This is for display and packaging
# First, define maps to temporarily replace multi-word phrases with single tokens
display_map = {
    'With consumer (actively being consumed)': 'With_consumer_(actively_being_consumed)',
    'On shelf/carton': 'On_shelf/carton',
    'In refridgerator/cooler': 'In_refrigerator/cooler',  # fix the misspelling
    'In refrigerator/cooler': 'In_refrigerator/cooler',
    'On display stand': 'On_display_stand'
}

package_map = {
    'PET bottle (50cl/1L)': 'PET_bottle_(50cl/1L)',
    'Glass bottle (35cl/60cl)': 'Glass_bottle_(35cl/60cl)',
    'Can (33cl)': 'Can_(33cl)'
}

def clean_display_combination(entry):
    if not isinstance(entry, str):
        return entry
    temp = entry.strip().replace('refridgerator', 'refrigerator')
    # Substitute multi-word display phrases with temporary tokens
    for k, v in display_map.items():
        temp = temp.replace(k, v)
    parts = [p.strip() for p in temp.split() if p.strip()]
    # Restore spaces and slashes, join with comma + space
    cleaned = [p.replace('_', ' ') for p in parts]
    return ', '.join(cleaned)

def clean_package_combination(entry):
    if not isinstance(entry, str):
        return entry
    temp = entry.strip()
    for k, v in package_map.items():
        temp = temp.replace(k, v)
    parts = [p.strip() for p in temp.split() if p.strip()]
    cleaned = [p.replace('_', ' ') for p in parts]
    return ', '.join(cleaned)

# Apply in place on your DataFrame (still called 'frame' here)
frame['Product_Display_(Combined_Response)'] = (
    frame['Product_Display_(Combined_Response)']
    .apply(clean_display_combination)
)

frame['Package_Type_(Combined_Response)'] = (
    frame['Package_Type_(Combined_Response)']
    .apply(clean_package_combination)
)

# Inspect the result if desired
#print(frame["Product_Display_(Combined_Response)"].value_counts())
#print(frame["Package_Type_Combined"].value_counts())


#Let's make the "S/N column the index of our dataframe since we don't need it for analysis"
# First convert the S/N column to numeric in case it was read as text

"""frame['S/N'] = pd.to_numeric(frame['S/N'], errors='coerce')

# Set the S/N column as the index (in‑place)
frame.set_index('S/N', inplace=True)"""

# Optional: inspect the head to confirm
#print(frame.head())
#print("Index name:", frame.index.name)
#print("Sample index values:", frame.index[:5])


#Final Checks 
print("Final data set shape:", frame.shape)
#print(frame.info())
#print(frame.head())

print("Data cleaning complete !")

frame.to_csv("cleaned_product_visibility.csv", index=False)
print("Cleaned data set saved as: cleaned_product_visibility.csv")


#This is the end of the data cleaning workflow
#Now we will perform some visualizations using matplotlib and django for interactivity
#You can find the scripts in the current directory under the name "visualizations.py"