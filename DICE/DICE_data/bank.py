import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("../")

def bank_data():
    """
    Prepare the data of dataset Bank Marketing
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0
    with open("../datasets/bank", "r") as ins:
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

    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def bank_data2():
    """
    Prepare the data of dataset Bank Marketing
    :return: X, Y, input shape and number of classes
    """

    raw_data = []
    i = 0
    col_names = None

    # Read in the raw data and format it
    with open("../datasets/bank.data.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(';')
            
            # Strip double quotes from each value
            line1 = [part.strip('"') for part in line1]

            if (i == 0):
                col_names = line1
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [i for i in line1]
            raw_data.append(L)

    df = pd.DataFrame(raw_data, columns=col_names)

    # Find all the categorical cols
    categorical_cols = []
    for indx, value in enumerate(df.iloc[0]):
        try:
            int(value)
        except Exception:
            categorical_cols.append(col_names[indx])

    # Get all the values that are possible for categorical columns
    categorical_unique_values = dict()
    for col in categorical_cols:
        categorical_unique_values[col] = df[col].unique().tolist()

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

    return X, Y, input_shape, nb_classes, label_encoders, categorical_unique_values, col_names[:-1]
