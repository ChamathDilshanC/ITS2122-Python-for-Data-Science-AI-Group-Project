# ITS2122: Python for Data Science & AI - Group Project

## Online Retail Dataset Analysis

This project demonstrates a comprehensive data science workflow using the Online Retail dataset. The project is organized into distinct phases, each with its own directory and documentation.

## Project Structure

```
📁 ITS2122 Python for Data Science & AI - Group Project/
├── 📁 Phase_1_Data_Cleaning/
│   ├── 📓 data_cleaning.ipynb         # Data cleaning workflow
│   ├── 📁 cleaned/                    # Cleaned datasets
│   │   └── 📄 online_retail_cleaned.csv
│   └── 📄 README.md                   # Phase 1 documentation
├── 📁 Phase_2_Exploratory_Data_Analysis/
│   ├── 📓 exploratory_data_analysis.ipynb  # EDA workflow
│   └── 📄 README.md                   # Phase 2 documentation
├── � online_retail.csv               # Original raw dataset
├── 📄 ITS2122_ Python for Data Science & AI - Group Project Specification.pdf
└── 📄 README.md                       # This file
```

## Phase Overview

### ✅ Phase 1: Data Cleaning

**Status**: COMPLETED
**Location**: `Phase_1_Data_Cleaning/`

**Objectives**:

- Load and explore raw data
- Handle missing values and duplicates
- Standardize data formats and types
- Export cleaned dataset

**Results**:

- Cleaned 1,067,371 → 797,885 records
- Removed 269,486 problematic rows
- Standardized 8 columns with proper data types
- Ready for analysis

### ✅ Phase 2: Exploratory Data Analysis

**Status**: COMPLETED
**Location**: `Phase_2_Exploratory_Data_Analysis/`

**Objectives**:

- Comprehensive data exploration
- Time-based sales analysis
- Customer behavior analysis
- Product performance analysis
- Geographic distribution analysis
- Statistical correlations
- Business insights generation

**Key Findings**:

- 797,885 transactions across 38 countries
- 4,382 unique customers, 3,958 unique products
- 2-year data span (Dec 2009 - Dec 2011)
- Comprehensive sales patterns and trends identified

### 🚀 Phase 3: Advanced Analytics (Future)

**Status**: READY FOR IMPLEMENTATION

**Planned Analysis**:

- Customer segmentation (RFM analysis)
- Market basket analysis
- Time series forecasting
- Machine learning models
- Predictive analytics

## Dataset Information

**Source**: Online Retail Dataset
**Format**: CSV
**Size**: ~65MB (raw), ~50MB (cleaned)
**Records**: 797,885 (after cleaning)
**Features**: 8 columns
**Time Range**: December 2009 - December 2011

**Columns**:

- `invoice`: Invoice number
- `stockcode`: Product stock code
- `description`: Product description
- `quantity`: Quantity purchased
- `invoicedate`: Transaction date/time
- `price`: Unit price
- `customer_id`: Customer identifier
- `country`: Customer country

## Data Quality

✅ **Clean Data**: No missing values or duplicates
✅ **Consistent Format**: Standardized column names and data types
✅ **Ready for Analysis**: Prepared for advanced analytics
✅ **Well Documented**: Comprehensive documentation for each phase

## Getting Started

1. **Start with Phase 1**: Open `Phase_1_Data_Cleaning/data_cleaning.ipynb`
2. **Proceed to Phase 2**: Open `Phase_2_Exploratory_Data_Analysis/exploratory_data_analysis.ipynb`
3. **Review Results**: Check the README files in each phase directory

## Requirements

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn

## Business Value

This analysis provides:

- Customer behavior insights
- Product performance metrics
- Geographic market analysis
- Seasonal trend identification
- Revenue optimization opportunities
- Data-driven business recommendations

## Next Steps

1. **Customer Segmentation**: Implement RFM analysis
2. **Market Basket Analysis**: Product recommendation system
3. **Forecasting**: Time series prediction models
4. **Machine Learning**: Advanced predictive analytics
5. **Dashboard**: Interactive business intelligence dashboard

---

**Project Team**: [NovaScript]
**Course**: ITS2122 Python for Data Science & AI
**Completion Date**: 30 August 2025
