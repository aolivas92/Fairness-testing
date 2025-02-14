import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import csv
import sys
sys.path.append("../")

from DICE_utils.config2 import census

def census_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("../datasets/census", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def census_data2():
    """
    Prepare the data of dataset Census
    :return: 
            X, 
            Y, 
            input shape, 
            number of classes
    """

    raw_data = []
    i = 0
    col_names = None

    # Read in the raw data and format it
    with open("../datasets/heart.csv", "r", newline="") as ins:
        reader = csv.reader(ins)  # csv.reader automatically handles quoted fields
        for i, row in enumerate(reader):
            if i == 0:
                # First row is column headers
                col_names = row
            else:
                raw_data.append(row)

    df = pd.DataFrame(raw_data, columns=col_names)
    # TODO: Used for testing, the original config doesn't have the 'eduction.num' column.
    df.drop('education.num', axis=1, inplace=True)
    col_names.remove('education.num')

    # Replace the ages with old/young if they are above/below 40 years old.
    df['age'] = df['age'].astype(int)
    df['age'] = df['age'].apply(lambda x: 'old' if x >=40 else 'young')


    # Try to find the categorical columns 5 times.
    X_raw = df.iloc[:, :-1]
    categorical_cols = []
    i = 5
    for i in range(0, 5):
        i *= 5
        for indx, value in enumerate(X_raw.iloc[i]):
            try:
                float(value)
            except Exception:
                col_name = col_names[indx]
                if col_name not in categorical_cols:
                    categorical_cols.append(col_names[indx])

    # Get all the values that are possible for categorical columns
    categorical_unique_values = dict()
    for col in categorical_cols:
        categorical_unique_values[col] = X_raw[col].unique().tolist()

    # Transform the categorical data in numerical data.
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Convert to X, Y arrays
    X = df_encoded.iloc[:, :-1].values
    Y_raw = df_encoded.iloc[:, -1].values

    # Conver Y_raw
    Y = []
    for val in Y_raw:
        if val == 0:
            Y.append([1, 0])
        else:
            Y.append([0, 1])

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, X.shape[1])
    nb_classes = 2

    # Build the system message and the input_bounds based on the attributes in the data.
    # system_message = "You are a counterfactual estimator trained to generate hypothetical scenarios based on census-style data attributes, assisting users in exploring alternative realities. Your task is to adjust one or more specified attributes in the user-provided input data to create a realistic counterfactual estimate while maintaining logical and statistical consistency. When altering the specified attributes, you should also adjust other related attributes to reflect realistic statistical patterns observed in the population. For example, when generating the counterfactual, not only adjust the specified attribute(s) but also modify other related attributes to create a realistic scenario based on demographic and occupational statistics. For instance, if the 'Sex' attribute changes from 'Male' to 'Female', consider that females may statistically work fewer hours per week, have different income levels, and occupy different occupations. Adjust 'Occupation' to reflect jobs more commonly held by females, 'Hours per Week' to represent average working hours for females, 'Income' to align with average earnings for females in similar roles, and other relevant attributes accordingly to maintain realism. Each user input contains a JSON dictionary with detailed demographic, socioeconomic, and occupational attributes of a real-world individual. Ensure logical and statistical consistency across all attributes when adjusting them (e.g., education level should align with age and occupation, sex should align with relationship, and consider interdependencies between features). Use knowledge of typical demographic patterns to adjust related attributes, creating a plausible and realistic counterfactual individual. Return a JSON dictionary formatted exactly like the input, with only the specified and related attributes altered to reflect realistic statistical adjustments. Do not add explanations, comments, or any additional information beyond the altered JSON data. The data attributes include: Age (continuous numerical value, if the user is younger than 40 then change the counterfactual age will be 50 and if the user is older than 40 than the counterfactual age will be 25), Workclass (employment category such as 'Private', 'Federal-gov', 'Self-employed', or 'Without-pay'), fnlwgt (Census Bureau-assigned weight indicating demographic characteristics), Education (highest education level completed, e.g., 'HS-grad', 'Bachelor's', 'Doctorate'), Education.num (numerical representation of education level, e.g., 'HS-grad=9', 'Bachelor's=13'), Marital Status (e.g., 'Married', 'Divorced', 'Separated'), Occupation (e.g., 'Tech-support', 'Sales', 'Exec-managerial', 'Craft-repair'), Relationship (e.g., 'Husband', 'Wife', 'Own-child', 'Unmarried'), Race (e.g., 'White', 'Black', 'Asian-Pac-Islander'), Sex ('Male' or 'Female'), Capital Gain (continuous numerical value for gains from investments), Capital Loss (continuous numerical value for losses from investments), Hours per Week (average weekly work hours), Native Country (e.g., 'United States', 'Germany', 'Japan'), and Income (income category, either '<=50K' or '>50K'); attributes must align logically (e.g., a 'Doctorate' education level implies a higher age range and professional occupation, and a 'Part-time' work schedule should reflect lower 'Hours per Week'). Respond only with the adjusted data in JSON format, formatted exactly as follows with double quotes: {\"Age\": ..., \"Workclass\": ..., \"fnlwgt\": ..., \"Education\": ..., \"Education.num\": ..., \"Marital.status\": ..., \"Occupation\": ..., \"Relationship\": ..., \"Race\": ..., \"Sex\": ..., \"Capital.gain\": ..., \"Capital.loss\": ..., \"Hours.per.week\": ..., \"Native.country\": ..., \"Counterfactual.request\": ...}. Focus on delivering concise, realistic outputs that align with the user's request and reflect statistical realities."
    list_of_attributes_with_values = ""
    list_of_attributes_formatted = "{"
    input_bounds = []

    for col in col_names[:-1]: # ignores the y value
        minimum = math.floor(float(df_encoded[col].min()))
        maximum = math.ceil(float(df_encoded[col].max()))

        # -----------system_message-----------
        if col in categorical_unique_values.keys():
            list_of_attributes_with_values += f" {col} ({categorical_unique_values[col]})"
        elif col.lower() == "age":
            list_of_attributes_with_values += f" {col} (categorical value, if the user is younger than 40 than they classify as young and if the user is older than 40 than they classify as old.)"
        else:
            list_of_attributes_with_values += f" {col} (numerical value, range is {minimum} - {maximum})"
        
        if col == col_names[-2]: # ignores the y value
            list_of_attributes_formatted += "\\\"" + col + "\\\":...}"    
        else:
            list_of_attributes_formatted += f"\\\"{col}\\\":..., "
        
        # -----------input_bounds-----------
        input_bounds.append([minimum, maximum])



    system_message = f"You are a counterfactual estimator trained to generate hypothetical scenarios based on census-style data attributes, assisting users in exploring alternative realities. Your task is to adjust one or more specified attributes in the user-provided input data to create a realistic counterfactual estimate while maintaining logical and statistical consistency. When altering the specified attributes, you should also adjust other related attributes to reflect realistic statistical patterns observed in the population. For example, when generating the counterfactual, not only adjust the specified attribute(s) but also modify other related attributes to create a realistic scenario based on demographic and occupational statistics. For instance, if the 'Sex' attribute changes from 'Male' to 'Female', consider that females may statistically work fewer hours per week, have different income levels, and occupy different occupations. Adjust 'Occupation' to reflect jobs more commonly held by females, 'Hours per Week' to represent average working hours for females, 'Income' to align with average earnings for females in similar roles, and other relevant attributes accordingly to maintain realism. Each user input contains a JSON dictionary with detailed demographic, socioeconomic, and occupational attributes of a real-world individual. Ensure logical and statistical consistency across all attributes when adjusting them (e.g., education level should align with age and occupation, sex should align with relationship, and consider interdependencies between features). Use knowledge of typical demographic patterns to adjust related attributes, creating a plausible and realistic counterfactual individual. Return a JSON dictionary formatted exactly like the input, with only the specified and related attributes altered to reflect realistic statistical adjustments. Do not add explanations, comments, or any additional information beyond the altered JSON data. The data attributes include: {list_of_attributes_with_values} attributes must align logically (e.g., a 'Doctorate' education level implies a higher age range and professional occupation, and a 'Part-time' work schedule should reflect lower 'Hours per Week'). Respond only with the adjusted data in JSON format, formatted exactly as follows with double quotes: {list_of_attributes_formatted} Focus on delivering concise, realistic outputs that align with the user's request and reflect statistical realities."
    

    # Pass the Known information to the config file.
    census.params = len(col_names[:-1])
    census.input_bounds = input_bounds
    census.feature_name = col_names[:-1]
    # census.class_name =
    # census.categorical_features =
    census.system_message = system_message
    census.label_encoders = label_encoders
    census.categorical_unique_values = categorical_unique_values
    
    # print(census.params)
    # print(census.input_bounds)
    # print(census.feature_name)
    # # print(census.system_message)
    # print(census.label_encoders)
    # print(census.categorical_unique_values)

    return X, Y, input_shape, nb_classes
