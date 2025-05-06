import streamlit as st
import pandas as pd
import joblib

# Load the trained model and columns
model = joblib.load("career_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🎓 AI-Driven Career Path Predictor")

# User Input Fields
tenth = st.number_input("10th Marks", min_value=0, max_value=100, value=80)
twelfth = st.number_input("12th Marks", min_value=0, max_value=100, value=85)
gpa = st.number_input("B.Tech GPA", min_value=0.0, max_value=10.0, value=8.0)
internship = st.selectbox("Internship Experience", ["Yes", "No"])
project = st.selectbox("Project Experience", ["Yes", "No"])

tech_skills = st.multiselect("Technical Skills", ["python", "java", "ml", "dbms", "networking"])
soft_skills = st.multiselect("Soft Skills", ["communication", "leadership", "teamwork", "creativity"])
interests = st.multiselect("Interests", ["coding", "design", "research"])
personality = st.selectbox("Personality Type", ["INTJ", "ENTP", "ISFP", "INFJ", "ISTJ", "INFP"])

if st.button("Predict Career Path"):

    # Prepare input
    input_data = {
        "10th": tenth,
        "12th": twelfth,
        "BTech_GPA": gpa,
        "Internship": 1 if internship == "Yes" else 0,
        "Project_Experience": 1 if project == "Yes" else 0,
        "Technical_Skills": ",".join(tech_skills),
        "Soft_Skills": ",".join(soft_skills),
        "Interests": ",".join(interests),
        "Personality_Type": personality
    }

    # Same encoding logic
    def multi_hot_encode_input(input_data, known_columns, column_name):
        col_prefix = column_name + "_"
        selected = input_data.get(column_name, "").split(",")
        encoded = {col: 0 for col in known_columns if col.startswith(col_prefix)}
        for skill in selected:
            key = col_prefix + skill.strip()
            if key in encoded:
                encoded[key] = 1
        return encoded

    # Build full row
    input_row = {
        "10th": input_data["10th"],
        "12th": input_data["12th"],
        "BTech_GPA": input_data["BTech_GPA"],
        "Internship": input_data["Internship"],
        "Project_Experience": input_data["Project_Experience"]
    }
    input_row.update(multi_hot_encode_input(input_data, model_columns, "Technical_Skills"))
    input_row.update(multi_hot_encode_input(input_data, model_columns, "Soft_Skills"))
    input_row.update(multi_hot_encode_input(input_data, model_columns, "Interests"))
    input_row.update(multi_hot_encode_input(input_data, model_columns, "Personality_Type"))

    final_input = {col: input_row.get(col, 0) for col in model_columns}
    input_df = pd.DataFrame([final_input])

    # Predict
    probs = model.predict_proba(input_df)[0]
    labels = model.classes_
    top3 = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:3]

    st.success("🎯 Top 3 Career Suggestions:")
    for career, score in top3:
        st.write(f"**{career}** - Confidence: {score:.2f}")
