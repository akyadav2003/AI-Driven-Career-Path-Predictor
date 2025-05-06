import pandas as pd
import joblib
import numpy as np

# Load model and expected column order
model = joblib.load("career_model.pkl")
model_columns = joblib.load("model_columns.pkl")  # ✅ Load expected column names

# Sample input
input_data = {
    "10th": 85,
    "12th": 82,
    "BTech_GPA": 8.1,
    "Technical_Skills": "python,java,ml",
    "Soft_Skills": "communication,teamwork",
    "Interests": "coding,research",
    "Personality_Type": "INTJ",
    "Internship": 1,
    "Project_Experience": 1
}

# Helper for encoding multi-select
def multi_hot_encode_input(input_data, known_columns, column_name):
    col_prefix = column_name + "_"
    selected = input_data.get(column_name, "").split(",")
    encoded = {col: 0 for col in known_columns if col.startswith(col_prefix)}
    for skill in selected:
        key = col_prefix + skill.strip()
        if key in encoded:
            encoded[key] = 1
    return encoded

# Start forming input row
input_row = {
    "10th": input_data["10th"],
    "12th": input_data["12th"],
    "BTech_GPA": input_data["BTech_GPA"],
    "Internship": input_data["Internship"],
    "Project_Experience": input_data["Project_Experience"]
}

# Update with encoded multi-selects
input_row.update(multi_hot_encode_input(input_data, model_columns, "Technical_Skills"))
input_row.update(multi_hot_encode_input(input_data, model_columns, "Soft_Skills"))
input_row.update(multi_hot_encode_input(input_data, model_columns, "Interests"))
input_row.update(multi_hot_encode_input(input_data, model_columns, "Personality_Type"))

# Make sure all columns exist
final_input = {}
for col in model_columns:
    final_input[col] = input_row.get(col, 0)  # Fill missing ones with 0

# Create DataFrame
input_df = pd.DataFrame([final_input])

# Predict
probs = model.predict_proba(input_df)[0]
labels = model.classes_
top3 = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]

print("🎯 Top 3 Career Recommendations:")
for label, prob in top3:
    print(f"- {label} (Confidence: {prob:.2f})")
