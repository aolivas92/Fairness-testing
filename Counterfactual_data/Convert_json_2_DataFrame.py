import json
import pandas as pd
import numpy as np
filename = './Generated_Counterfactuals.json'
f = open(filename)
json_file = json.load(f)
columns = json_file['messages'][1]['content'].keys()
df = pd.DataFrame(columns=columns)
for i in range(1,len(json_file['messages']),2):
    ind = json_file['messages'][i]['content']['Index']
    
    if isinstance(json_file['messages'][i]['content'],dict): 
        org_sample = pd.DataFrame(json_file['messages'][i]['content'],index=[ind])
    else:
        
        org_sample = pd.DataFrame(np.array(['NaN' for i in range(len(columns))]).reshape(1,17),columns=columns)
        
    if isinstance(json_file['messages'][i+1]['content'],dict): 
        cf_sample = pd.DataFrame(json_file['messages'][i+1]['content'],index=[ind])
    else:
        cf_sample = pd.DataFrame(np.array(['NaN' for i in range(len(columns))]).reshape(1,17),columns=columns)

    df = pd.concat([df,org_sample])


 
    df = pd.concat([df,cf_sample])
df.to_csv(f'{filename}.csv')