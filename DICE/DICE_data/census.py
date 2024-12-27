import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../")

def census_data(input_file="../datasets/census", label_encoders=None):
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open(input_file, "r") as ins:
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

    return X, Y, input_shape, nb_classes, label_encoders

def census_data2():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    
    # List all the column names so you can define them.
    col_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income"
    ]

    # Read the data and make a copy so you can leave one untouched.
    df = pd.read_csv("../datasets/adult.data",
                     names=col_names,
                     header=None,
                     skipinitialspace=True)
    df_encoded = df.copy()

    print("Initial data:\n", df.head())

    # List out which columns needs to be transformed and which don't.
    categorical_cols = [
    'workclass', 'education', 'marital.status', 
    'occupation', 'relationship', 'race', 
    'sex', 'native.country', 'income'
    ]
    numeric_cols = [
        'age', 'fnlwgt', 'education.num', 
        'capital.gain', 'capital.loss', 'hours.per.week'
    ]

    # Transform the categorical data into numerical data.
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    print("Transformed data:\n", df_encoded.head())

    df_encoded.to_csv("../datasets/census2", index=False, header=True)

    # Passing the location of the new file and the encoders
    return census_data("../datasets/census2", label_encoders)

    # # TODO: Delete this, it's only for testing.
    # print(df_encoded.iloc[60])
    
    # for col in categorical_cols:
    #     le = label_encoders[col]
    #     df_encoded[col] = le.inverse_transform(df_encoded[col])

    # print(df_encoded.iloc[60])
    # print(df.iloc[60])
    