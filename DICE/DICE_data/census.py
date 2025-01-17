import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../")

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
    Prepare the data of dataset Bank Marketing
    :return: 
            X, 
            Y, 
            input shape, 
            number of classes,
            system message for LLM, 
            encoder used to transform the data,
            the unique values used to encode the data, 
            the name of the columns
    """

    system_message = "You are a counterfactual estimator trained to generate hypothetical scenarios based on census-style data attributes, assisting users in exploring alternative realities. Your task is to adjust one or more specified attributes in the user-provided input data to create a realistic counterfactual estimate while maintaining logical and statistical consistency. When altering the specified attributes, you should also adjust other related attributes to reflect realistic statistical patterns observed in the population. For example, when generating the counterfactual, not only adjust the specified attribute(s) but also modify other related attributes to create a realistic scenario based on demographic and occupational statistics. For instance, if the 'Sex' attribute changes from 'Male' to 'Female', consider that females may statistically work fewer hours per week, have different income levels, and occupy different occupations. Adjust 'Occupation' to reflect jobs more commonly held by females, 'Hours per Week' to represent average working hours for females, 'Income' to align with average earnings for females in similar roles, and other relevant attributes accordingly to maintain realism. Each user input contains a JSON dictionary with detailed demographic, socioeconomic, and occupational attributes of a real-world individual. Ensure logical and statistical consistency across all attributes when adjusting them (e.g., education level should align with age and occupation, sex should align with relationship, and consider interdependencies between features). Use knowledge of typical demographic patterns to adjust related attributes, creating a plausible and realistic counterfactual individual. Return a JSON dictionary formatted exactly like the input, with only the specified and related attributes altered to reflect realistic statistical adjustments. Do not add explanations, comments, or any additional information beyond the altered JSON data. The data attributes include: Age (continuous numerical value, if the user is younger than 40 then change the counterfactual age will be 50 and if the user is older than 40 than the counterfactual age will be 25), Workclass (employment category such as 'Private', 'Federal-gov', 'Self-employed', or 'Without-pay'), fnlwgt (Census Bureau-assigned weight indicating demographic characteristics), Education (highest education level completed, e.g., 'HS-grad', 'Bachelor's', 'Doctorate'), Education.num (numerical representation of education level, e.g., 'HS-grad=9', 'Bachelor's=13'), Marital Status (e.g., 'Married', 'Divorced', 'Separated'), Occupation (e.g., 'Tech-support', 'Sales', 'Exec-managerial', 'Craft-repair'), Relationship (e.g., 'Husband', 'Wife', 'Own-child', 'Unmarried'), Race (e.g., 'White', 'Black', 'Asian-Pac-Islander'), Sex ('Male' or 'Female'), Capital Gain (continuous numerical value for gains from investments), Capital Loss (continuous numerical value for losses from investments), Hours per Week (average weekly work hours), Native Country (e.g., 'United States', 'Germany', 'Japan'), and Income (income category, either '<=50K' or '>50K'); attributes must align logically (e.g., a 'Doctorate' education level implies a higher age range and professional occupation, and a 'Part-time' work schedule should reflect lower 'Hours per Week'). Respond only with the adjusted data in JSON format, formatted exactly as follows with double quotes: {\"Age\": ..., \"Workclass\": ..., \"fnlwgt\": ..., \"Education\": ..., \"Education.num\": ..., \"Marital.status\": ..., \"Occupation\": ..., \"Relationship\": ..., \"Race\": ..., \"Sex\": ..., \"Capital.gain\": ..., \"Capital.loss\": ..., \"Hours.per.week\": ..., \"Native.country\": ..., \"Counterfactual.request\": ...}. Focus on delivering concise, realistic outputs that align with the user's request and reflect statistical realities."
    
    # List all the column names so you can define them.
    col_names = [
    "Age", # 0
    "Workclass", # 1 
    "fnlwgt", # 2
    "Education", # 3
    "Education.num", # 4
    "Marital.status", # 5
    "Occupation", # 6
    "Relationship", # 7
    "Race", # 8 
    "Sex", # 9
    "Capital.gain", # 10
    "Capital.loss", # 11
    "Hours.per.week", # 12
    "Native.country", # 13
    "Income" # 14
    ]

    # Read the data and make a copy so you can leave one untouched.
    df = pd.read_csv("../datasets/adult.data",
                     names=col_names,
                     header=None,
                     skipinitialspace=True)
    df_encoded = df.copy()

    # print("Initial data:\n", df.head())

    # List out which columns needs to be transformed and which don't.
    categorical_cols = [
    'Workclass', 'Education', 'Marital.status', 
    'Occupation', 'Relationship', 'Race', 
    'Sex', 'Native.country'
    ]

    # Transform the categorical data into numerical data.
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # TODO: Delete later, just used for reading purpose.
    df_encoded.to_csv("../datasets/census2", index=False, header=True)

    # Convert to X, Y arrays right here
    # Suppose 'income' is the last column in the CSV
    X = df_encoded.iloc[:, :-1].values  # all but the last column
    Y_raw = df_encoded.iloc[:, -1].values

    # Convert Y_raw
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

    # Get all the values that are possible so we can encode the generated result.
    categorical_unique_values = dict()
    for col in categorical_cols:
        categorical_unique_values[col] = df[col].unique().tolist()

    return X, Y, input_shape, nb_classes, system_message, label_encoders, categorical_unique_values, col_names[:-1]

    # # TODO: Delete this, it's only for testing.
    # print(df_encoded.iloc[60])
    
    # for col in categorical_cols:
    #     le = label_encoders[col]
    #     df_encoded[col] = le.inverse_transform(df_encoded[col])

    # print(categorical_unique_values)

    # find_closest_regex_match("private-sector", categorical_unique_values['Workclass'], 10)

    # my_dict = {value: index for index, value in enumerate(col_names[:-1])}
    # print(my_dict)

    # print(df_encoded.iloc[60])
    # print(df.iloc[60])


# census_data2()