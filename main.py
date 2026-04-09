import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Change file name if needed
data = pd.read_csv("data.csv")

# =========================
# 1. Dataset Preview
# =========================
print("===== Dataset Preview =====")
print(data.head())

# =========================
# 2. Dataset Info
# =========================
print("\n===== Dataset Info =====")
print(data.info())

# =========================
# 3. Statistical Summary
# =========================
print("\n===== Statistical Summary =====")
print(data.describe())

# =========================
# 4. Data Cleaning
# =========================
# Remove duplicates
data = data.drop_duplicates()

# Fill missing values
data = data.fillna(method='ffill')

# =========================
# 5. Correlation Heatmap
# =========================
plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.show()

# =========================
# 6. Histogram Plots
# =========================
data.hist(figsize=(10, 8))
plt.suptitle("Histogram Plots")
plt.savefig("images/histograms.png")
plt.show()

# =========================
# 7. Sales Trend (if exists)
# =========================
if "Sales" in data.columns:
    plt.figure()
    plt.plot(data["Sales"], label="Sales")
    plt.title("Sales Trend")
    plt.legend()
    plt.savefig("images/sales_trend.png")
    plt.show()

    # Moving Average
    data["Moving_Avg"] = data["Sales"].rolling(window=5).mean()

    plt.figure()
    plt.plot(data["Sales"], label="Original")
    plt.plot(data["Moving_Avg"], label="Moving Average")
    plt.legend()
    plt.title("Moving Average")
    plt.savefig("images/moving_avg.png")
    plt.show()

# =========================
# 8. Report Generation
# =========================
with open("analysis_report.txt", "w") as f:
    f.write("===== DATA ANALYSIS REPORT =====\n\n")

    f.write("Columns:\n")
    f.write(str(data.columns.tolist()) + "\n\n")

    f.write("Statistical Summary:\n")
    f.write(str(data.describe()))

print("\n✅ Report generated successfully!")
