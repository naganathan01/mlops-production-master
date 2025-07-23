# Generate sample_training_data.csv
import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 2000
n_features = 10

# Generate realistic features with some correlation structure
print("Generating sample training data...")

# Create base signals
base_signal_1 = np.random.normal(0, 1, n_samples)
base_signal_2 = np.random.normal(0, 1, n_samples)

data = {}

# Generate features with realistic correlation structure
for i in range(n_features):
    # Features 0-4: correlated with base_signal_1
    if i < 5:
        correlation = 0.3 + (i * 0.1)  # Varying correlation
        noise = np.random.normal(0, 1, n_samples)
        data[f'feature_{i}'] = correlation * base_signal_1 + np.sqrt(1 - correlation**2) * noise
    
    # Features 5-9: correlated with base_signal_2  
    else:
        correlation = 0.2 + ((i-5) * 0.1)
        noise = np.random.normal(0, 1, n_samples)
        data[f'feature_{i}'] = correlation * base_signal_2 + np.sqrt(1 - correlation**2) * noise

# Create target variable with complex decision boundary
X = np.column_stack(list(data.values()))

# Complex target function
target = (
    (X[:, 0] + X[:, 1] * 0.8 + X[:, 2] * 0.3) +
    (X[:, 5] * 0.5 + X[:, 6] * 0.4) +
    np.random.normal(0, 0.1, n_samples)  # Add some noise
) > 0.2

# Convert to binary classification
data['target'] = target.astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic data quality issues (small percentage)
# Add a few missing values
missing_indices = np.random.choice(n_samples, size=int(0.01 * n_samples), replace=False)
df.loc[missing_indices, 'feature_3'] = np.nan

# Add a few outliers
outlier_indices = np.random.choice(n_samples, size=int(0.005 * n_samples), replace=False)
df.loc[outlier_indices, 'feature_7'] = np.random.normal(0, 5, len(outlier_indices))

# Save to CSV
output_path = 'data/sample_training_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Generated {len(df)} samples with {len(df.columns)-1} features")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Saved to: {output_path}")

# Print basic statistics
print("\nDataset Statistics:")
print("="*50)
print(f"Shape: {df.shape}")
print(f"Features: {df.columns.tolist()[:-1]}")
print(f"Target classes: {sorted(df['target'].unique())}")
print(f"Class balance: {df['target'].value_counts(normalize=True).round(3).to_dict()}")

print("\nFeature Statistics:")
numeric_features = [col for col in df.columns if col != 'target']
for feature in numeric_features[:3]:  # Show first 3 features
    print(f"{feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")

print(f"\nData saved successfully to {output_path}")
print("You can now run the MLOps pipeline with this data!")