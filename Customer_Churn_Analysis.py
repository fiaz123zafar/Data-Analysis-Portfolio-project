# Customer Churn Analysis - Telecom Industry
# Domain: Business/Telecommunications

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# Generate synthetic telecom customer data
np.random.seed(789)

n_customers = 4000

data = {
    'customer_id': range(10000, 10000 + n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 120, n_customers).round(2),
    'total_charges': None,  # Will calculate based on tenure and monthly
    'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_customers, p=[0.35, 0.45, 0.2]),
    'online_security': np.random.choice(['Yes', 'No', 'No Internet'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No', 'No Internet'], n_customers),
    'streaming_tv': np.random.choice(['Yes', 'No', 'No Internet'], n_customers),
    'streaming_movies': np.random.choice(['Yes', 'No', 'No Internet'], n_customers),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4]),
    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
    'partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
    'dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
    'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
    'multiple_lines': np.random.choice(['Yes', 'No', 'No Phone'], n_customers)
}

df = pd.DataFrame(data)

# Calculate total charges
df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)

# Create churn based on risk factors
churn_probability = 0.2  # Base probability
churn_factors = (
    (df['contract_type'] == 'Month-to-Month') * 0.3 +
    (df['tenure_months'] < 12) * 0.25 +
    (df['monthly_charges'] > 80) * 0.15 +
    (df['tech_support'] == 'No') * 0.1 +
    (df['senior_citizen'] == 1) * 0.1
)

df['churn'] = np.where(
    np.random.random(n_customers) < (churn_probability + churn_factors),
    'Yes', 'No'
)

print("=" * 80)
print("CUSTOMER CHURN ANALYSIS - TELECOM INDUSTRY")
print("=" * 80)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total Customers: {len(df)}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing Values:")
print(df.isnull().sum())

# 2. Churn Overview
print("\n\n2. CHURN ANALYSIS")
print("-" * 80)
churn_counts = df['churn'].value_counts()
churn_rate = (churn_counts['Yes'] / len(df) * 100)
print(f"Churn Distribution:")
print(churn_counts)
print(f"\nChurn Rate: {churn_rate:.2f}%")
print(f"Retained Customers: {churn_counts['No']} ({churn_counts['No']/len(df)*100:.2f}%)")

# 3. Demographics and Churn
print("\n\n3. DEMOGRAPHIC ANALYSIS")
print("-" * 80)
print(f"Senior Citizens:")
senior_churn = pd.crosstab(df['senior_citizen'], df['churn'], normalize='index') * 100
print(senior_churn.round(2))

print(f"\nPartner Status:")
partner_churn = pd.crosstab(df['partner'], df['churn'], normalize='index') * 100
print(partner_churn.round(2))

print(f"\nDependents:")
dependent_churn = pd.crosstab(df['dependents'], df['churn'], normalize='index') * 100
print(dependent_churn.round(2))

# 4. Contract Analysis
print("\n\n4. CONTRACT TYPE ANALYSIS")
print("-" * 80)
contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index') * 100
print("Churn Rate by Contract Type (%):")
print(contract_churn.round(2))
print(f"\nContract Distribution:")
print(df['contract_type'].value_counts())

# 5. Payment Method Analysis
print("\n\n5. PAYMENT METHOD ANALYSIS")
print("-" * 80)
payment_churn = pd.crosstab(df['payment_method'], df['churn'], normalize='index') * 100
print("Churn Rate by Payment Method (%):")
print(payment_churn.round(2))

# 6. Service Usage Analysis
print("\n\n6. SERVICE USAGE ANALYSIS")
print("-" * 80)
print("Internet Service:")
internet_churn = pd.crosstab(df['internet_service'], df['churn'], normalize='index') * 100
print(internet_churn.round(2))

print("\nOnline Security:")
security_churn = pd.crosstab(df['online_security'], df['churn'], normalize='index') * 100
print(security_churn.round(2))

print("\nTech Support:")
support_churn = pd.crosstab(df['tech_support'], df['churn'], normalize='index') * 100
print(support_churn.round(2))

# 7. Financial Analysis
print("\n\n7. FINANCIAL ANALYSIS")
print("-" * 80)
print("Monthly Charges by Churn Status:")
print(df.groupby('churn')['monthly_charges'].describe().round(2))

print("\nTotal Charges by Churn Status:")
print(df.groupby('churn')['total_charges'].describe().round(2))

# Revenue Impact
churned_revenue = df[df['churn'] == 'Yes']['monthly_charges'].sum()
total_revenue = df['monthly_charges'].sum()
revenue_at_risk = (churned_revenue / total_revenue * 100)

print(f"\nRevenue Impact:")
print(f"Total Monthly Revenue: ${total_revenue:,.2f}")
print(f"Revenue from Churned Customers: ${churned_revenue:,.2f}")
print(f"Revenue at Risk: {revenue_at_risk:.2f}%")

# 8. Tenure Analysis
print("\n\n8. TENURE ANALYSIS")
print("-" * 80)
print("Tenure Statistics by Churn:")
print(df.groupby('churn')['tenure_months'].describe().round(2))

# Create tenure groups
df['tenure_group'] = pd.cut(df['tenure_months'], 
                            bins=[0, 12, 24, 48, 72], 
                            labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months'])
tenure_churn = pd.crosstab(df['tenure_group'], df['churn'], normalize='index') * 100
print("\nChurn Rate by Tenure Group (%):")
print(tenure_churn.round(2))

# 9. Visualizations
print("\n\n9. GENERATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Churn Distribution
ax1 = plt.subplot(3, 3, 1)
churn_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
plt.title('Churn Distribution', fontsize=12, fontweight='bold')
plt.ylabel('')

# Plot 2: Churn by Contract Type
ax2 = plt.subplot(3, 3, 2)
contract_data = pd.crosstab(df['contract_type'], df['churn'])
contract_data.plot(kind='bar', ax=ax2, color=['green', 'red'])
plt.title('Churn by Contract Type', fontsize=12, fontweight='bold')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(['No Churn', 'Churn'])

# Plot 3: Monthly Charges Distribution
ax3 = plt.subplot(3, 3, 3)
df[df['churn'] == 'No']['monthly_charges'].hist(alpha=0.7, bins=30, label='No Churn', color='green')
df[df['churn'] == 'Yes']['monthly_charges'].hist(alpha=0.7, bins=30, label='Churn', color='red')
plt.title('Monthly Charges Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Frequency')
plt.legend()

# Plot 4: Tenure vs Churn
ax4 = plt.subplot(3, 3, 4)
df[df['churn'] == 'No']['tenure_months'].hist(alpha=0.7, bins=30, label='No Churn', color='green')
df[df['churn'] == 'Yes']['tenure_months'].hist(alpha=0.7, bins=30, label='Churn', color='red')
plt.title('Tenure Distribution by Churn', fontsize=12, fontweight='bold')
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.legend()

# Plot 5: Internet Service vs Churn
ax5 = plt.subplot(3, 3, 5)
internet_data = pd.crosstab(df['internet_service'], df['churn'])
internet_data.plot(kind='bar', ax=ax5, color=['green', 'red'])
plt.title('Churn by Internet Service', fontsize=12, fontweight='bold')
plt.xlabel('Internet Service Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(['No Churn', 'Churn'])

# Plot 6: Payment Method vs Churn
ax6 = plt.subplot(3, 3, 6)
payment_data = pd.crosstab(df['payment_method'], df['churn'])
payment_data.plot(kind='bar', ax=ax6, color=['green', 'red'])
plt.title('Churn by Payment Method', fontsize=12, fontweight='bold')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(['No Churn', 'Churn'])

# Plot 7: Tech Support Impact
ax7 = plt.subplot(3, 3, 7)
support_data = pd.crosstab(df['tech_support'], df['churn'])
support_data.plot(kind='bar', ax=ax7, color=['green', 'red'])
plt.title('Churn by Tech Support Status', fontsize=12, fontweight='bold')
plt.xlabel('Tech Support')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(['No Churn', 'Churn'])

# Plot 8: Tenure Groups
ax8 = plt.subplot(3, 3, 8)
tenure_group_data = pd.crosstab(df['tenure_group'], df['churn'])
tenure_group_data.plot(kind='bar', ax=ax8, color=['green', 'red'])
plt.title('Churn by Tenure Group', fontsize=12, fontweight='bold')
plt.xlabel('Tenure Group')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.legend(['No Churn', 'Churn'])

# Plot 9: Senior Citizen Impact
ax9 = plt.subplot(3, 3, 9)
senior_data = pd.crosstab(df['senior_citizen'], df['churn'])
senior_data.plot(kind='bar', ax=ax9, color=['green', 'red'])
plt.title('Churn by Senior Citizen Status', fontsize=12, fontweight='bold')
plt.xlabel('Senior Citizen (0=No, 1=Yes)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.savefig('churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'churn_analysis_dashboard.png'")

# 10. Predictive Model
print("\n\n10. CHURN PREDICTION MODEL")
print("-" * 80)

# Prepare data for modeling
df_model = df.copy()
le = LabelEncoder()

# Encode categorical variables
categorical_cols = ['contract_type', 'payment_method', 'internet_service', 'online_security',
                   'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing',
                   'partner', 'dependents', 'phone_service', 'multiple_lines']

for col in categorical_cols:
    df_model[col] = le.fit_transform(df_model[col])

df_model['churn'] = le.fit_transform(df_model['churn'])

# Select features
features = ['tenure_months', 'monthly_charges', 'total_charges', 'contract_type', 'payment_method',
           'internet_service', 'online_security', 'tech_support', 'senior_citizen', 'partner', 'dependents']

X = df_model[features]
y = df_model['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Model performance
print("Model Accuracy:", rf_model.score(X_test, y_test).round(4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 11. Key Insights
print("\n\n11. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 80)
print("\nKey Findings:")
print(f"1. Overall Churn Rate: {churn_rate:.2f}%")
print(f"2. Highest Risk Group: Month-to-Month contracts")
print(f"3. Revenue at Risk: ${churned_revenue:,.2f} ({revenue_at_risk:.2f}%)")
print(f"4. Average Tenure of Churned Customers: {df[df['churn']=='Yes']['tenure_months'].mean():.1f} months")
print(f"5. Tech Support reduces churn significantly")

print("\nBusiness Recommendations:")
print("• Incentivize long-term contracts (1-2 years) with discounts")
print("• Improve first-year customer experience to reduce early churn")
print("• Promote tech support and online security services")
print("• Target high-value customers with retention campaigns")
print("• Offer loyalty rewards for customers beyond 24 months")
print("• Focus on Month-to-Month customers with personalized offers")
print("• Implement proactive support for customers showing churn signals")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save dataset
df.to_csv('telecom_churn_data.csv', index=False)
print("\n✓ Dataset saved as 'telecom_churn_data.csv'")