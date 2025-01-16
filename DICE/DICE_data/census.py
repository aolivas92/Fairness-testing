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
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    
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
    'Sex', 'Native.country', 'Income'
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

    # Convert Y_raw into one-hot if you want:
    # e.g. if the label_encoders["income"] mapped <=50K -> 0 and >50K -> 1
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

    return X, Y, input_shape, nb_classes, label_encoders, categorical_unique_values, col_names

    # # TODO: Delete this, it's only for testing.
    # print(df_encoded.iloc[60])
    
    # for col in categorical_cols:
    #     le = label_encoders[col]
    #     df_encoded[col] = le.inverse_transform(df_encoded[col])

    # print(categorical_unique_values)

    # find_closest_regex_match("private-sector", categorical_unique_values['Workclass'], 10)

    # print(df_encoded.iloc[60])
    # print(df.iloc[60])
