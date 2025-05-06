import pandas as pd

# Sample data
data = {
    '10th': [85, 78, 91, 88, 92],
    '12th': [82, 76, 88, 81, 90],
    'BTech_GPA': [8.1, 7.8, 8.9, 8.3, 8.7],
    'Technical_Skills': ['python,java,ml', 'java,dbms', 'python,java,ml,dbms', 'python,ml', 'python,java,ml,dbms'],
    'Soft_Skills': ['communication,leadership', 'teamwork,communication', 'leadership,teamwork', 'communication,leadership', 'leadership'],
    'Interests': ['coding,design', 'design,research', 'coding,design', 'coding,design', 'research,coding'],
    'Personality_Type': ['INTJ', 'ENTP', 'INFJ', 'INTP', 'ISFP'],
    'Internship': [1, 0, 1, 1, 1],
    'Project_Experience': [1, 0, 1, 1, 1],
    'Career': ['Data Scientist', 'UX Designer', 'ML Engineer', 'Cybersecurity Analyst', 'Data Analyst']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('career_data.csv', index=False)

print("CSV file created successfully!")
