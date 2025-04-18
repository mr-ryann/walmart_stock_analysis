import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

# Load the dataset
df = pd.read_csv(r'data/walmart_stocks.csv')
# Data Cleaning
df.dropna(inplace=True)
# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'], utc=True)

# Calculate daily returns
# pct_change computes the percentage change in closing price to analyze stock performance
df['daily_return'] = df['close'].pct_change()
# Remove rows with NaN returns (first row after pct_change)
# dropna ensures no NaN values in daily_return, which could affect calculations
df.dropna(subset=['daily_return'], inplace=True)

# Create binary target for logistic regression (1 if positive return, 0 otherwise)
# apply with lambda creates a binary variable for classification in logistic regression
df['positive_return'] = df['daily_return'].apply(lambda x: 1 if x > 0 else 0)

# Remove extreme outliers in daily returns (e.g., beyond 3 standard deviations)
# np.abs and std help filter extreme values to improve model reliability
mean_return = df['daily_return'].mean()
std_return = df['daily_return'].std()
df = df[np.abs(df['daily_return'] - mean_return) <= 3 * std_return]

# Save cleaned sample dataset
df.to_csv(r'output/walmart_stocks_cleaned.csv', index=False)

# Simple Random Sampling
# sample randomly selects a subset of data for analysis, reducing computation while maintaining representativeness
sample_size = 1000
sample = df.sample(n=sample_size, random_state=42)

# Z-test: Compare mean daily returns of two periods (before and after 2010)
# Split data by date to compare historical vs. recent performance
# loc filters data based on date conditions for temporal analysis
sample_before_2010 = sample.loc[sample['date'] < pd.to_datetime('2010-01-01', utc=True), 'daily_return']
sample_after_2010 = sample.loc[sample['date'] >= pd.to_datetime('2010-01-01', utc=True), 'daily_return']

# Perform z-test
# ztest from statsmodels compares means of two independent samples, assuming large sample size for normality
z_stat, p_value = ztest(sample_before_2010, sample_after_2010, value=0)

# Print z-test results
# format ensures clear, readable output of statistical results
print("Z-test Results (Comparing mean daily returns before and after 2010):")
print(f"Z-statistic: {z_stat:.2f}")
print(f"P-value: {p_value:.4f}")
# Conditional statement interprets statistical significance
if p_value < 0.05:
    print("Reject the null hypothesis: Significant difference in mean daily returns.")
else:
    print("Fail to reject the null hypothesis: No significant difference in mean daily returns.")

# Logistic Regression: Predict positive return based on volume and previous day's return
# Create lagged return feature
# shift creates a feature of the previous day's return, capturing momentum effects
sample['lag_return'] = sample['daily_return'].shift(1)

# Remove rows with NaN lag_return
# dropna ensures no missing values in features used for regression
sample.dropna(subset=['lag_return'], inplace=True)

# Define features and target
# Select features relevant to predicting stock movement
X = sample[['volume', 'lag_return']]
# Add constant for intercept in logistic regression
# add_constant includes an intercept term for the regression model
X = sm.add_constant(X)
y = sample['positive_return']

# Fit logistic regression model
# Logit from statsmodels fits a logistic regression model for binary classification
logit_model = sm.Logit(y, X).fit()

# Print regression results
# summary provides detailed output of coefficients, p-values, and model fit
print("\nLogistic Regression Results:")
print(logit_model.summary())

# Visualization: Plot actual vs predicted probabilities
# predict computes predicted probabilities for visualization
sample['predicted_prob'] = logit_model.predict(X)
# Scatter plot to visualize relationship between volume and positive return probability
plt.figure(figsize=(8, 6))
plt.scatter(sample['volume'], sample['positive_return'], alpha=0.3, label='Actual')
plt.scatter(sample['volume'], sample['predicted_prob'], alpha=0.3, label='Predicted Probability')
plt.xlabel('Trading Volume')
plt.ylabel('Positive Return (1=Yes, 0=No) / Predicted Probability')
plt.title('Trading Volume vs Positive Return')
plt.legend()

# savefig saves the plot for documentation
plt.savefig(r'output/volume_vs_return.png')
# Save cleaned sample dataset
sample.to_csv(r'output/walmart_stocks_cleaned_sample.csv', index=False)
