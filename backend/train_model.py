import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load CSV
df = pd.read_csv("career_data.csv")

# Step 2: Convert multi-select string columns into individual binary columns
def multi_hot_encode(df, column):
    # Split string by comma, remove extra spaces, explode the lists
    exploded = df[column].str.get_dummies(sep=',')
    # Join back to original dataframe
    df = df.drop(column, axis=1)
    df = pd.concat([df, exploded], axis=1)
    return df

# Apply to your multi-select fields
df = multi_hot_encode(df, 'Technical_Skills')
df = multi_hot_encode(df, 'Soft_Skills')
df = multi_hot_encode(df, 'Interests')

# Step 3: One-hot encode Personality Type (categorical)
df = pd.get_dummies(df, columns=['Personality_Type'])

# Step 4: Split features and label
X = df.drop('Career', axis=1)
y = df['Career']

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Save model
# After model.fit(...)
joblib.dump(model, "career_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")  # <- save feature names

print(" Model trained and saved successfully!")
