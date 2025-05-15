# DICE Project Setup Guide

## Pre-Processing the Dataset

### Step 1

Make sure the raw dataset is in this location as a **CSV file**:  
```bash
DICE/datasets/
```

---

### Step 2

Create or modify a corresponding preprocessing file inside:  
```bash
DICE/DICE_data/
```

- **If there is already a file**, add your logic to it and update the method used inside `DICE_Search.py`.
- **If not**, create a new preprocessing file and register it in `DICE_Search.py`.

Depending on how your dataset is structured:
- Use `DICE_data/bank.py` as a reference for **clean datasets** with **no missing values**.
- Use `DICE_data/law_school.py` as a reference for datasets with **missing values** or where **only specific features are used**.

#### Preprocessing Template (Pseudocode)

```python
def preprocess_data(file_path, delimiter=",", quote_strip='"', drop_columns=None, binary_target=True):
    """
    General preprocessing for structured CSV data for ML.
    """

    # Step 1: Load and clean data
    raw_lines = []
    column_names = None
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(delimiter)
            parts = [val.strip(quote_strip) for val in parts]
            if i == 0:
                column_names = parts
                continue
            raw_lines.append(parts)

    # Step 2: Create DataFrame
    df = pd.DataFrame(raw_lines, columns=column_names)

    # Step 3: Drop unwanted columns if needed
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)
        column_names = [c for c in column_names if c not in drop_columns]

    # Step 4: Optionally bin 'age' into 'young'/'old'
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)
        df['age'] = df['age'].apply(lambda x: 'old' if x >= 40 else 'young')

    # Step 5: Detect categorical columns using sampling
    X_raw = df.iloc[:, :-1]
    categorical_cols = set()
    sample_indices = [i * 5 for i in range(5)]
    for i in sample_indices:
        for idx, val in enumerate(X_raw.iloc[i]):
            try:
                float(val)
            except:
                categorical_cols.add(column_names[idx])
    categorical_cols = list(categorical_cols)

    # Step 6: Store unique values for categorical attributes
    categorical_unique_values = {
        col: X_raw[col].unique().tolist()
        for col in categorical_cols
    }

    # Step 7: Encode categorical features
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Step 8: Split X and Y
    X = df_encoded.iloc[:, :-1].astype(float).values
    Y_raw = df_encoded.iloc[:, -1].astype(int).values

    # Step 9: Convert Y to one-hot encoding
    Y = [[1, 0] if y == 0 else [0, 1] for y in Y_raw]
    Y = np.array(Y, dtype=float)

    # Step 10: Metadata
    input_shape = (None, X.shape[1])
    nb_classes = 2 if binary_target else len(np.unique(Y_raw))

    # Step 11: Pass the known information to the config
    DICE_utils.config2.params = len(column_names[:-1])
    DICE_utils.config2.input_bounds = input_bounds
    DICE_utils.config2.feature_name = column_names[:-1]
    DICE_utils.config2.system_message = system_message
    DICE_utils.config2.label_encoders = label_encoders
    DICE_utils.config2.categorical_unique_values = categorical_unique_values

    return X, Y, input_shape, nb_classes, label_encoders, categorical_unique_values
```

---

### Step 3

Add your dataset configuration to:  
```bash
DICE/DICE_utils/config2.py
```

> Most datasets follow a similar config format. You can copy an existing setup and modify it for your dataset.

---

## Setting up `DICE_Search.py`

1. **Import your data and config:**

```python
from DICE_data.law_school import law_school_data
from DICE_utils.config2 import lawschool
```

2. **Update the `data` and `data_config` variables** before the call to `m_instance()`.

3. **Verify the log output path** (It shouldn't need updating as it uses the dataset and sens_params that are passed.)

---

### Changing the LLM

Inside the `m_instance_real_counterfactual()` method, you’ll find a variable named `llm`.  
- Set `llm = 0` for **LLaMA**
- Set `llm = 1` for **Claude**

---

## Running the Bash Script

Located at:  
```bash
DICE/DICE_tutorial/run_expiriments.bash
```

This script:
- Runs `DICE_Search` 10 times
- Renames logs after each run
- Appends additional logging

### Required Setup:
Make sure the following are defined:
- The dataset you want to use, will give it to the command and is used for searching in the folders. 
- The sensitive_index, will give it to the command and is used for searching for the file.
- The LLM used
- The output directory path (must match the one `DICE_Search` uses, shouldn't need updating if the top three are set up correctly).
- The command to run (shouldn't need updating if the top three are set up correctly).

> I usually create the output directory before running it, it will create the directory for you in case you miss this step.

---

## Running the Log Processor

Located at:  
```bash
DICE/DICE_tutorial/Process_Log_File.py
```

This script:
- Scans log files
- Extracts metrics
- Outputs summary CSVs

### Setup Instructions:
At the bottom of the file, **specify the following**:
- `llm_type`: Either `"llama"` or `"claude"`
- The directory path where the logs are stored