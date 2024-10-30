import re
import pandas as pd

# Sample Llama output (replace this with the actual output text from Llama)
llama_output = """
Age: 42
Workclass: Private
Education: Bachelors
Marital Status: Married-civ-spouse
Occupation: Exec-managerial
Relationship: Husband
Race: White
Sex: Male
Capital Gain: 5000
Capital Loss: 0
Hours per Week: 40
Native Country: United-States
"""

# Define mappings for categorical variables to convert them to numerical values
workclass_mapping = {
    "Private": 1,
    "Self-emp-not-inc": 2,
    "Self-emp-inc": 3,
    "Federal-gov": 4,
    "Local-gov": 5,
    "State-gov": 6,
    "Without-pay": 7,
    "Never-worked": 8
}

education_mapping = {
    "Bachelors": 1,
    "Some-college": 2,
    "11th": 3,
    "HS-grad": 4,
    "Prof-school": 5,
    "Assoc-acdm": 6,
    "Assoc-voc": 7,
    "Doctorate": 8,
    "Masters": 9,
    "10th": 10,
    "7th-8th": 11,
    "5th-6th": 12,
    "Preschool": 13
}

marital_status_mapping = {
    "Married-civ-spouse": 1,
    "Divorced": 2,
    "Never-married": 3,
    "Separated": 4,
    "Widowed": 5,
    "Married-spouse-absent": 6,
    "Married-AF-spouse": 7
}

occupation_mapping = {
    "Tech-support": 1,
    "Craft-repair": 2,
    "Other-service": 3,
    "Sales": 4,
    "Exec-managerial": 5,
    "Prof-specialty": 6,
    "Handlers-cleaners": 7,
    "Machine-op-inspct": 8,
    "Adm-clerical": 9,
    "Farming-fishing": 10,
    "Transport-moving": 11,
    "Priv-house-serv": 12,
    "Protective-serv": 13,
    "Armed-Forces": 14
}

relationship_mapping = {
    "Wife": 1,
    "Own-child": 2,
    "Husband": 3,
    "Not-in-family": 4,
    "Other-relative": 5,
    "Unmarried": 6
}

race_mapping = {
    "White": 1,
    "Asian-Pac-Islander": 2,
    "Amer-Indian-Eskimo": 3,
    "Other": 4,
    "Black": 5
}

sex_mapping = {
    "Female": 0,
    "Male": 1
}

country_mapping = {
    "United-States": 1,
    "Cambodia": 2,
    "England": 3,
    "Puerto-Rico": 4,
    "Canada": 5,
    "Germany": 6,
    "India": 7,
    "Japan": 8,
    "Greece": 9,
    "China": 10,
    "Cuba": 11,
    "Iran": 12,
    # Add more countries as needed
}

# Parse Llama output
data = {}
for line in llama_output.strip().split('\n'):
    key, value = map(str.strip, line.split(':'))
    data[key] = value

# Convert parsed data to numerical format
numerical_data = {
    "Age": int(data["Age"]),
    "Workclass": workclass_mapping.get(data["Workclass"], None),
    "Education": education_mapping.get(data["Education"], None),
    "Marital Status": marital_status_mapping.get(data["Marital Status"], None),
    "Occupation": occupation_mapping.get(data["Occupation"], None),
    "Relationship": relationship_mapping.get(data["Relationship"], None),
    "Race": race_mapping.get(data["Race"], None),
    "Sex": sex_mapping.get(data["Sex"], None),
    "Capital Gain": int(data["Capital Gain"]),
    "Capital Loss": int(data["Capital Loss"]),
    "Hours per Week": int(data["Hours per Week"]),
    "Native Country": country_mapping.get(data["Native Country"], None)
}

# Convert numerical data to DataFrame for easy viewing and further processing
df = pd.DataFrame([numerical_data])
print("Numerical Data in DataFrame format:")
print(df)
