import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Dell/Downloads/5000-Sales-Records/5000 Sales Records.csv")

print("Initial Dataset Shape:", df.shape)

# Count of the initial duplicates
initial_duplicates = df.duplicated().sum()

# Removing spaces
df.columns = df.columns.str.strip()

# Converting date columns with error coercion
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

# Counting missing values before dropping
print("\n📉 Missing Values Before Cleaning:")
print(df.isnull().sum())

# Dropping duplicate rows
df = df.drop_duplicates()

# Dropping rows with any missing/null values
df = df.dropna()

# Removing white spaces
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

# Removing zero or negative values
num_cols = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']
for col in num_cols:
    df = df[df[col] > 0]

# Removing outliers in 'Units Sold'
Q1 = df['Units Sold'].quantile(0.25)
Q3 = df['Units Sold'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Units Sold'] >= Q1 - 1.5 * IQR) & (df['Units Sold'] <= Q3 + 1.5 * IQR)]

# Cleaned
print(f"\nDuplicates Removed: {initial_duplicates}")
print(f"Dataset Shape After Cleaning: {df.shape}")
print("\nColumns in Cleaned Data:", list(df.columns))
print("\nSample of Cleaned Data:")
print(df.head(3))

# Top 10 countries by Total Revenue
top_countries = df.groupby('Country')['Total Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_countries.values, y=top_countries.index,hue=top_countries.index, palette="viridis",legend=False)
plt.title("Top 10 Countries by Total Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# Pie Chart
region_sales = df.groupby('Region')['Units Sold'].sum()

plt.figure(figsize=(8,8))
plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
plt.title("Sales Distribution by Region")
plt.axis('equal')
plt.show()

# Line Plot
df['Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Total Revenue'].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot(kind='line', marker='o', color='blue')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']].corr(), 
            annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Financial Metrics")
plt.show()

# Bar Plot
item_profit = df.groupby('Item Type')['Total Profit'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=item_profit.values, y=item_profit.index,hue=item_profit.index, palette="crest",legend=False)
plt.title("Top 10 Most Profitable Item Types")
plt.xlabel("Total Profit")
plt.ylabel("Item Type")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='Order Priority', y='Total Profit',hue='Order Priority', palette="Set2",legend=False)
plt.title("Total Profit by Order Priority")
plt.xlabel("Order Priority")
plt.ylabel("Total Profit")
plt.tight_layout()
plt.show()

# Stacked Bar
sales_channel_region = df.groupby(['Region', 'Sales Channel'])['Total Revenue'].sum().unstack()

sales_channel_region.plot(kind='bar', stacked=True, figsize=(12,6), colormap="Paired")
plt.title("Total Revenue by Region and Sales Channel")
plt.xlabel("Region")
plt.ylabel("Total Revenue")
plt.legend(title="Sales Channel")
plt.tight_layout()
plt.show()
