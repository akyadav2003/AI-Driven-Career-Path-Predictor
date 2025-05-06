import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv("career_data.csv")

# Check the first few rows
print("Original Data:")
print(df.head())

# 1. Convert multi-select fields to binary values (e.g., Technical Skills, Soft Skills, Interests)
mlb_technical = MultiLabelBinarizer()
mlb_soft = MultiLabelBinarizer()
mlb_interests = MultiLabelBinarizer()

# Transform the multi-select columns
technical_skills = mlb_technical.fit_transform(df['Technical_Skills'].str.split(','))
soft_skills = mlb_soft.fit_transform(df['Soft_Skills'].str.split(','))
interests = mlb_interests.fit_transform(df['Interests'].str.split(','))

# Create DataFrames from the binary features
df_technical = pd.DataFrame(technical_skills, columns=mlb_technical.classes_)
df_soft = pd.DataFrame(soft_skills, columns=mlb_soft.classes_)
df_interests = pd.DataFrame(interests, columns=mlb_interests.classes_)

# Concatenate the new DataFrames with the original one (dropping the old columns)
df = pd.concat([df.drop(['Technical_Skills', 'Soft_Skills', 'Interests'], axis=1), df_technical, df_soft, df_interests], axis=1)

# 2. Normalize numerical features
scaler = MinMaxScaler()
df[['10th', '12th', 'BTech_GPA']] = scaler.fit_transform(df[['10th', '12th', 'BTech_GPA']])

# 3. Encode the categorical 'Career' column using LabelEncoder
label_encoder = LabelEncoder()
df['Career'] = label_encoder.fit_transform(df['Career'])

# 4. Split the dataset into features (X) and target (y)
X = df.drop('Career', axis=1)
y = df['Career']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data for training
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessing complete and data saved!")
