# %% [markdown]
# # Phase 3: Predictive Modeling
#
# This notebook demonstrates the process of building and evaluating predictive models using the cleaned online retail dataset. Each step is explained with comments and visualizations for clarity.
#
# ## 1. Import Required Libraries
#
# We use pandas, numpy, matplotlib, seaborn, and requests (for API integration).

# %%
# Import essential libraries for data science and visualization
import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical data visualization
import requests  # For API integration
# Set visualization style for consistency
sns.set(style='whitegrid')

# %% [markdown]
# ## 2. Load and Prepare Data
#
# We load the cleaned dataset and perform basic checks.

# %%
# Load the cleaned retail data from Phase 1
data_path = '../Phase_1_Data_Cleaning/cleaned/online_retail_cleaned.csv'
df = pd.read_csv(data_path)
# Display the first few rows to understand the data structure
df.head()

# %%
# Check for missing values and data types
print('DataFrame info:')
df.info()
print('\nMissing values per column:')
print(df.isnull().sum())

# %% [markdown]
# ## 3. Feature Engineering
#
# We create new features that may help improve model performance.

# %%
# Feature Engineering: Create new features for modeling
df['TotalPrice'] = df['quantity'] * df['price']  # Total price for each transaction
df['invoicedate'] = pd.to_datetime(df['invoicedate'])  # Ensure datetime type
df['InvoiceMonth'] = df['invoicedate'].dt.month  # Extract month
df['InvoiceDay'] = df['invoicedate'].dt.day      # Extract day
df['InvoiceHour'] = df['invoicedate'].dt.hour    # Extract hour
# Preview the updated dataframe
df.head()

# %% [markdown]
# ## 4. Model Selection and Training
#
# We select a simple regression model to predict 'TotalPrice'.

# %%
# Prepare features and target variable
features = ['quantity', 'price', 'InvoiceMonth', 'InvoiceDay', 'InvoiceHour']
X = df[features]
y = df['TotalPrice']
# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train a Linear Regression model to predict TotalPrice
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)

# %% [markdown]
# ## 5. Model Evaluation
#
# We evaluate the model using standard regression metrics and visualize the results.

# %%
# Calculate evaluation metrics for regression model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')

# %%
# Visualize actual vs predicted TotalPrice values
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel('Actual TotalPrice')
plt.ylabel('Predicted TotalPrice')
plt.title('Actual vs Predicted TotalPrice')
plt.show()

# %% [markdown]
# ## 6. Interpretation and Visualization
#
# We interpret the model coefficients and visualize feature importance.

# %%
# Show model coefficients for each feature to interpret importance
coefficients = pd.Series(model.coef_, index=features)
print('Feature Coefficients:')
print(coefficients)
# Visualize feature importance
plt.figure(figsize=(6,4))
coefficients.plot(kind='barh')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.show()

# %% [markdown]
# ## 7. API Integration Example
#
# Demonstrate using the `requests` library to fetch data from an external API.

# %%
# Example: Fetch data from a public API (e.g., exchange rates) using requests
response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
if response.status_code == 200:
    data = response.json()
    print('Exchange rates for USD:')
    print(data['rates'])
else:
    print('Failed to fetch data from API')

# %% [markdown]
# ## Enhanced Data Visualizations
#
# Below are additional visualizations to better understand the data and model results. All use only matplotlib and seaborn for styling.

# %%
# Distribution of TotalPrice (Histogram and Boxplot)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.histplot(df['TotalPrice'], bins=50, color='royalblue', kde=True)
plt.title('Distribution of TotalPrice')
plt.xlabel('TotalPrice')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.subplot(1,2,2)
sns.boxplot(x=df['TotalPrice'], color='orange')
plt.title('Boxplot of TotalPrice')
plt.xlabel('TotalPrice')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Correlation heatmap of main numerical features
plt.figure(figsize=(8,6))
corr = df[['quantity', 'price', 'TotalPrice', 'InvoiceMonth', 'InvoiceDay', 'InvoiceHour']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Key Features')
plt.tight_layout()
plt.show()

# %%
# Monthly sales trend line plot
monthly_sales = df.groupby('InvoiceMonth')['TotalPrice'].sum().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x='InvoiceMonth', y='TotalPrice', data=monthly_sales, marker='o', color='seagreen')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales (£)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1,13))
plt.tight_layout()
plt.show()

# %%
# Top 10 transactions by TotalPrice (bar plot)
top_transactions = df.nlargest(10, 'TotalPrice')
plt.figure(figsize=(10,6))
sns.barplot(x=top_transactions['TotalPrice'], y=top_transactions.index, palette='mako')
plt.title('Top 10 Transactions by TotalPrice')
plt.xlabel('TotalPrice (£)')
plt.ylabel('Transaction Index')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
