import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/a.msalarraza/Downloads/Customer-Churn-EDA/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------------
# 1. Dataset Overview
# -----------------------------
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# -----------------------------
# 2. Target Variable Analysis
# -----------------------------
plt.figure()
df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customer Count")
plt.show()

# -----------------------------
# 3. Numerical Feature Analysis
# -----------------------------

# Tenure
plt.figure()
plt.hist(df['tenure'], bins=30)
plt.title("Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Frequency")
plt.show()

# Monthly Charges
plt.figure()
plt.hist(df['MonthlyCharges'], bins=30)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")
plt.show()
plt.figure()

# Tenure vs Churn
df.boxplot(column='tenure', by='Churn')
plt.title("Tenure vs Churn")
plt.suptitle("")  # Removes default boxplot title
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()

# Total Charges
plt.figure()
plt.hist(df['TotalCharges'].dropna(), bins=30)
plt.title("Total Charges Distribution")
plt.xlabel("Total Charges")
plt.ylabel("Frequency")
plt.show()


# -----------------------------
# 4. Categorical Feature Analysis vs Churn
# -----------------------------

# Contract vs Churn
plt.figure()
pd.crosstab(df['Contract'], df['Churn']).plot(kind='bar')
plt.title("Churn vs Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()

# Internet Service vs Churn
plt.figure()
pd.crosstab(df['InternetService'], df['Churn']).plot(kind='bar')
plt.title("Churn vs Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Count")
plt.show()

# Senior Citizen vs Churn
plt.figure()
pd.crosstab(df['SeniorCitizen'], df['Churn']).plot(kind='bar')
plt.title("Churn vs Senior Citizen")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
