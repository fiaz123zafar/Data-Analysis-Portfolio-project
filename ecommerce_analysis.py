# E-Commerce Sales Analysis
# Domain: Business/Retail

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate synthetic e-commerce data
np.random.seed(42)

# Create date range
dates = pd.date_range(start='2023-01-01', end='2024-10-20', freq='D')
n_records = 5000

data = {
    'order_id': range(1, n_records + 1),
    'date': np.random.choice(dates, n_records),
    'customer_id': np.random.randint(1000, 5000, n_records),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'], n_records),
    'product_price': np.random.uniform(10, 500, n_records).round(2),
    'quantity': np.random.randint(1, 5, n_records),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records)
}

df = pd.DataFrame(data)
df['total_amount'] = (df['product_price'] * df['quantity']).round(2)
df['date'] = pd.to_datetime(df['date'])

print("=" * 80)
print("E-COMMERCE SALES ANALYSIS")
print("=" * 80)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total Records: {len(df)}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nData Types:")
print(df.dtypes)
print(f"\nMissing Values:")
print(df.isnull().sum())

# 2. Descriptive Statistics
print("\n\n2. DESCRIPTIVE STATISTICS")
print("-" * 80)
print(df.describe())

# 3. Sales Analysis
print("\n\n3. SALES ANALYSIS")
print("-" * 80)
total_revenue = df['total_amount'].sum()
avg_order_value = df['total_amount'].mean()
total_orders = len(df)
unique_customers = df['customer_id'].nunique()

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Order Value: ${avg_order_value:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Unique Customers: {unique_customers:,}")

# 4. Category Analysis
print("\n\n4. SALES BY CATEGORY")
print("-" * 80)
category_sales = df.groupby('product_category').agg({
    'total_amount': 'sum',
    'order_id': 'count'
}).round(2)
category_sales.columns = ['Total Revenue', 'Number of Orders']
category_sales = category_sales.sort_values('Total Revenue', ascending=False)
print(category_sales)

# 5. Regional Analysis
print("\n\n5. REGIONAL PERFORMANCE")
print("-" * 80)
regional_sales = df.groupby('region').agg({
    'total_amount': 'sum',
    'order_id': 'count'
}).round(2)
regional_sales.columns = ['Total Revenue', 'Number of Orders']
regional_sales = regional_sales.sort_values('Total Revenue', ascending=False)
print(regional_sales)

# 6. Payment Method Analysis
print("\n\n6. PAYMENT METHOD DISTRIBUTION")
print("-" * 80)
payment_dist = df['payment_method'].value_counts()
payment_revenue = df.groupby('payment_method')['total_amount'].sum().round(2)
payment_analysis = pd.DataFrame({
    'Transaction Count': payment_dist,
    'Total Revenue': payment_revenue
})
print(payment_analysis)

# 7. Time Series Analysis
print("\n\n7. MONTHLY SALES TREND")
print("-" * 80)
df['month'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('month')['total_amount'].sum().round(2)
print(monthly_sales.tail(12))

# 8. Customer Insights
print("\n\n8. CUSTOMER INSIGHTS")
print("-" * 80)
customer_analysis = df.groupby('customer_id').agg({
    'order_id': 'count',
    'total_amount': 'sum'
}).round(2)
customer_analysis.columns = ['Total Orders', 'Total Spent']
customer_analysis = customer_analysis.sort_values('Total Spent', ascending=False)
print(f"\nTop 10 Customers:")
print(customer_analysis.head(10))
print(f"\nAverage orders per customer: {customer_analysis['Total Orders'].mean():.2f}")
print(f"Average spend per customer: ${customer_analysis['Total Spent'].mean():.2f}")

# 9. Visualizations
print("\n\n9. GENERATING VISUALIZATIONS")
print("-" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Sales by Category
ax1 = plt.subplot(2, 3, 1)
category_sales['Total Revenue'].plot(kind='bar', color='skyblue')
plt.title('Revenue by Product Category', fontsize=12, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot 2: Regional Distribution
ax2 = plt.subplot(2, 3, 2)
regional_sales['Total Revenue'].plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Revenue Distribution by Region', fontsize=12, fontweight='bold')
plt.ylabel('')

# Plot 3: Payment Methods
ax3 = plt.subplot(2, 3, 3)
payment_dist.plot(kind='bar', color='lightcoral')
plt.title('Orders by Payment Method', fontsize=12, fontweight='bold')
plt.xlabel('Payment Method')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45, ha='right')

# Plot 4: Monthly Sales Trend
ax4 = plt.subplot(2, 3, 4)
monthly_data = df.groupby(df['date'].dt.to_period('M'))['total_amount'].sum()
monthly_data.index = monthly_data.index.to_timestamp()
plt.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2, color='green')
plt.title('Monthly Sales Trend', fontsize=12, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Plot 5: Price Distribution
ax5 = plt.subplot(2, 3, 5)
plt.hist(df['product_price'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title('Product Price Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

# Plot 6: Category vs Region Heatmap
ax6 = plt.subplot(2, 3, 6)
heatmap_data = df.pivot_table(values='total_amount', index='product_category', 
                                columns='region', aggfunc='sum')
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Revenue Heatmap: Category vs Region', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ecommerce_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'ecommerce_analysis_dashboard.png'")

# 10. Key Insights & Recommendations
print("\n\n10. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 80)
print("\nKey Findings:")
print(f"1. Top Category: {category_sales.index[0]} (${category_sales.iloc[0]['Total Revenue']:,.2f})")
print(f"2. Best Region: {regional_sales.index[0]} (${regional_sales.iloc[0]['Total Revenue']:,.2f})")
print(f"3. Most Popular Payment: {payment_dist.index[0]} ({payment_dist.iloc[0]} orders)")
print(f"4. Average Order Value: ${avg_order_value:.2f}")

print("\nRecommendations:")
print("• Focus marketing efforts on top-performing categories")
print("• Investigate underperforming regions for growth opportunities")
print("• Optimize payment gateway for most popular methods")
print("• Implement customer retention strategies for high-value customers")
print("• Consider seasonal promotions based on monthly trends")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save dataset for reference
df.to_csv('ecommerce_sales_data.csv', index=False)
print("\n✓ Dataset saved as 'ecommerce_sales_data.csv'")