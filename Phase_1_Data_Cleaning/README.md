# Phase 1: Data Cleaning

This directory contains all files related to the first phase of the data science project: **Data Cleaning**.

## Contents

### Files

- `data_cleaning.ipynb` - Main notebook containing the data cleaning workflow
- `cleaned/` - Directory containing cleaned datasets

### Data Cleaning Process

The data cleaning notebook performs the following steps:

1. **Import Required Libraries** - Import pandas for data manipulation
2. **Load Raw Data** - Read the raw CSV file into a pandas DataFrame
3. **Explore Data Structure** - Display first few rows and basic info
4. **Initial Data Inspection** - Check for missing values and outliers
5. **Handle Missing Values** - Remove rows with missing values
6. **Remove Duplicates** - Drop duplicate rows to ensure uniqueness
7. **Standardize Column Names** - Convert to lowercase with underscores
8. **Data Type Conversion** - Convert columns to appropriate data types
9. **Save Cleaned Data** - Export cleaned dataset to CSV

## Results

- **Initial dataset**: 1,067,371 rows
- **After cleaning**: 797,885 rows
- **Removed**: 269,486 rows (243,007 missing values + 26,479 duplicates)
- **Final columns**: 8 (invoice, stockcode, description, quantity, invoicedate, price, customer_id, country)

## Data Quality

✅ No missing values
✅ No duplicate records
✅ Consistent column naming
✅ Appropriate data types
✅ Ready for analysis

## Next Steps

Proceed to **Phase 2: Exploratory Data Analysis** for comprehensive data insights.
