# Healthcare Patient Analysis
# Domain: Healthcare

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Generate synthetic healthcare data
np.random.seed(123)

n_patients = 3000

data = {
    'patient_id': range(1000, 1000 + n_patients),
    'age': np.random.randint(18, 85, n_patients),
    'gender': np.random.choice(['Male', 'Female'], n_patients),
    'bmi': np.random.normal(26, 5, n_patients).clip(15, 45).round(1),
    'blood_pressure_systolic': np.random.normal(125, 15, n_patients).clip(90, 180).round(0),
    'blood_pressure_diastolic': np.random.normal(80, 10, n_patients).clip(60, 120).round(0),
    'cholesterol': np.random.choice(['Normal', 'Above Normal', 'High'], n_patients, p=[0.5, 0.3, 0.2]),
    'glucose_level': np.random.normal(100, 20, n_patients).clip(70, 200).round(0),
    'smoking': np.random.choice(['Yes', 'No'], n_patients, p=[0.25, 0.75]),
    'alcohol': np.random.choice(['Yes', 'No'], n_patients, p=[0.35, 0.65]),
    'physical_activity': np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.3, 0.5, 0.2]),
    'diagnosis': np.random.choice(['Healthy', 'Hypertension', 'Diabetes', 'Heart Disease', 'Obesity'], 
                                   n_patients, p=[0.45, 0.25, 0.15, 0.10, 0.05])
}

df = pd.DataFrame(data)

# Add readmission risk based on conditions
conditions = (
    (df['age'] > 60) | 
    (df['bmi'] > 30) | 
    (df['smoking'] == 'Yes') | 
    (df['cholesterol'] == 'High')
)
df['readmission_risk'] = np.where(conditions, 
                                   np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.2, 0.4, 0.4]),
                                   np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.6, 0.3, 0.1]))

print("=" * 80)
print("HEALTHCARE PATIENT ANALYSIS")
print("=" * 80)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total Patients: {len(df)}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nData Info:")
print(df.info())

# 2. Demographic Analysis
print("\n\n2. DEMOGRAPHIC ANALYSIS")
print("-" * 80)
print(f"Age Statistics:")
print(df['age'].describe())
print(f"\nGender Distribution:")
print(df['gender'].value_counts())
print(f"\nAge Groups:")
age_groups = pd.cut(df['age'], bins=[18, 30, 45, 60, 85], labels=['18-30', '31-45', '46-60', '60+'])
print(age_groups.value_counts().sort_index())

# 3. Health Metrics Analysis
print("\n\n3. HEALTH METRICS ANALYSIS")
print("-" * 80)
print(f"BMI Statistics:")
print(df['bmi'].describe())
print(f"\nBMI Categories:")
bmi_categories = pd.cut(df['bmi'], 
                        bins=[0, 18.5, 25, 30, 100], 
                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
print(bmi_categories.value_counts())

print(f"\nBlood Pressure Statistics:")
print(f"Systolic - Mean: {df['blood_pressure_systolic'].mean():.1f}, Std: {df['blood_pressure_systolic'].std():.1f}")
print(f"Diastolic - Mean: {df['blood_pressure_diastolic'].mean():.1f}, Std: {df['blood_pressure_diastolic'].std():.1f}")

print(f"\nCholesterol Levels:")
print(df['cholesterol'].value_counts())

print(f"\nGlucose Level Statistics:")
print(df['glucose_level'].describe())

# 4. Lifestyle Factors
print("\n\n4. LIFESTYLE FACTORS")
print("-" * 80)
print(f"Smoking Status:")
print(df['smoking'].value_counts())
print(f"\nAlcohol Consumption:")
print(df['alcohol'].value_counts())
print(f"\nPhysical Activity:")
print(df['physical_activity'].value_counts())

# 5. Diagnosis Distribution
print("\n\n5. DIAGNOSIS DISTRIBUTION")
print("-" * 80)
diagnosis_counts = df['diagnosis'].value_counts()
diagnosis_pct = (diagnosis_counts / len(df) * 100).round(2)
diagnosis_summary = pd.DataFrame({
    'Count': diagnosis_counts,
    'Percentage': diagnosis_pct
})
print(diagnosis_summary)

# 6. Risk Analysis
print("\n\n6. READMISSION RISK ANALYSIS")
print("-" * 80)
risk_counts = df['readmission_risk'].value_counts()
print(risk_counts)
print(f"\nHigh Risk Patients: {risk_counts.get('High', 0)} ({risk_counts.get('High', 0)/len(df)*100:.1f}%)")

# 7. Correlation Analysis
print("\n\n7. CORRELATION ANALYSIS")
print("-" * 80)
numeric_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'glucose_level']
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

# 8. Gender-based Analysis
print("\n\n8. GENDER-BASED HEALTH METRICS")
print("-" * 80)
gender_analysis = df.groupby('gender')[numeric_cols].mean().round(2)
print(gender_analysis)

# 9. Age Group Analysis
print("\n\n9. AGE GROUP HEALTH ANALYSIS")
print("-" * 80)
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 85], labels=['18-30', '31-45', '46-60', '60+'])
age_group_analysis = df.groupby('age_group')[numeric_cols].mean().round(2)
print(age_group_analysis)

# 10. Diagnosis by Risk Factors
print("\n\n10. DIAGNOSIS BY RISK FACTORS")
print("-" * 80)
risk_diagnosis = pd.crosstab(df['diagnosis'], df['smoking'], normalize='columns') * 100
print("Smoking vs Diagnosis (%):")
print(risk_diagnosis.round(2))

# 11. Visualizations
print("\n\n11. GENERATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Age Distribution
ax1 = plt.subplot(3, 3, 1)
plt.hist(df['age'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Plot 2: BMI Distribution by Gender
ax2 = plt.subplot(3, 3, 2)
df.boxplot(column='bmi', by='gender', ax=ax2)
plt.title('BMI Distribution by Gender', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Gender')
plt.ylabel('BMI')

# Plot 3: Diagnosis Distribution
ax3 = plt.subplot(3, 3, 3)
diagnosis_counts.plot(kind='bar', color='coral')
plt.title('Diagnosis Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45, ha='right')

# Plot 4: Blood Pressure Scatter
ax4 = plt.subplot(3, 3, 4)
plt.scatter(df['blood_pressure_systolic'], df['blood_pressure_diastolic'], 
           alpha=0.5, c=df['age'], cmap='viridis')
plt.colorbar(label='Age')
plt.title('Blood Pressure Analysis', fontsize=12, fontweight='bold')
plt.xlabel('Systolic BP')
plt.ylabel('Diastolic BP')

# Plot 5: Cholesterol Levels
ax5 = plt.subplot(3, 3, 5)
df['cholesterol'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Cholesterol Distribution', fontsize=12, fontweight='bold')
plt.ylabel('')

# Plot 6: Readmission Risk
ax6 = plt.subplot(3, 3, 6)
risk_counts.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Readmission Risk Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)

# Plot 7: Correlation Heatmap
ax7 = plt.subplot(3, 3, 7)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Health Metrics Correlation', fontsize=12, fontweight='bold')

# Plot 8: Lifestyle Factors
ax8 = plt.subplot(3, 3, 8)
lifestyle_data = pd.DataFrame({
    'Smoking': df['smoking'].value_counts(),
    'Alcohol': df['alcohol'].value_counts()
})
lifestyle_data.plot(kind='bar', ax=ax8)
plt.title('Lifestyle Factors', fontsize=12, fontweight='bold')
plt.xlabel('Response')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend()

# Plot 9: Physical Activity
ax9 = plt.subplot(3, 3, 9)
df['physical_activity'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Physical Activity Levels', fontsize=12, fontweight='bold')
plt.xlabel('Activity Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('healthcare_analysis_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'healthcare_analysis_dashboard.png'")

# 12. Key Insights
print("\n\n12. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 80)
print("\nKey Findings:")
print(f"1. Average Patient Age: {df['age'].mean():.1f} years")
print(f"2. Patients with Elevated BMI (>25): {(df['bmi'] > 25).sum()} ({(df['bmi'] > 25).sum()/len(df)*100:.1f}%)")
print(f"3. High-Risk Patients: {risk_counts.get('High', 0)} ({risk_counts.get('High', 0)/len(df)*100:.1f}%)")
print(f"4. Most Common Diagnosis: {diagnosis_counts.index[0]} ({diagnosis_counts.iloc[0]} cases)")
print(f"5. Smoking Rate: {(df['smoking'] == 'Yes').sum()/len(df)*100:.1f}%")

print("\nClinical Recommendations:")
print("• Implement weight management programs for high-BMI patients")
print("• Increase cardiovascular screening for patients over 60")
print("• Develop smoking cessation programs")
print("• Focus preventive care on high-risk categories")
print("• Promote physical activity programs, especially for low-activity patients")
print("• Regular monitoring for patients with multiple risk factors")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Save dataset
df.to_csv('healthcare_patient_data.csv', index=False)
print("\n✓ Dataset saved as 'healthcare_patient_data.csv'")