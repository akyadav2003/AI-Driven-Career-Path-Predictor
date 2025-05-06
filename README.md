# AI-Driven Career Path Predictor

This project is a machine learning-based web application designed to recommend the most suitable career paths for students based on their academic background, skills, interests, and personality traits. Built using **Python**, **Pandas**, **Scikit-learn**, and a simple **Streamlit** UI, it provides personalized career guidance for engineering students.

---

## Features

* Input-based prediction using:

  * 10th and 12th marks
  * B.Tech GPA
  * Technical skills (e.g., Python, Java, ML, etc.)
  * Soft skills (e.g., Communication, Leadership)
  * Interests (e.g., Coding, Design, Research)
  * Personality Type (MBTI-style)
  * Internship and Project Experience

* Machine learning model (Random Forest) trained on sample career datasets

* Encodes and preprocesses categorical and multi-select inputs

* Outputs the top career path predictions with confidence scores

* User interface built with Streamlit for ease of use

---

## Tech Stack

* **Frontend**: Streamlit (can be replaced with React)
* **Backend**: Python
* **ML Libraries**: scikit-learn, pandas, joblib
* **IDE**: Visual Studio Code (VS Code)

---

## Project Structure

```
AI-Driven Career Path Predictor/
│
├── backend/
│   ├── train_model.py        # Train and save the ML model
│   ├── predict.py            # Predict based on user input
│   ├── model.pkl             # Saved model
│   └── data.csv              # Dataset
│
├── app.py                    # Streamlit frontend app
├── requirements.txt          # Required Python libraries
└── README.md                 # Project description
```

---

## How to Run

1. **Clone the repository**
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model** (only once):

   ```bash
   python backend/train_model.py
   ```
4. **Run the app**:

   ```bash
   streamlit run app.py
   ```

---

## Future Improvements

* Replace Streamlit with a fully functional React frontend
* Use real-world datasets for improved accuracy
* Add career descriptions and resources based on predictions
* Integrate user login and result history
