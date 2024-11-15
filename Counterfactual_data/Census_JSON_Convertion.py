import csv
import json

csv_file_path = './Census_cf.csv'
output_file = './Census_Counterfactual.json'

json_output = {'messages': [
    {'role': 'system',
     'content': "You are a counterfactual estimator trained to generate hypothetical scenarios based on a variety of census-style data attributes, assisting users in exploring alternative realities. Each user input contains detailed demographic, socioeconomic, and occupational information about a real-world individual. Your role is to produce a realistic counterfactual estimate by adjusting one or more specified attributes while maintaining logical consistency. The specified attribute will be given specifically in the 'Counterfactual Request' value. The data attributes provided include: Age: Continuous numerical value representing the individual's age. Workclass: Employment category, such as Private, Federal-gov, Self-employed, or Without-pay. fnlwgt: A weight assigned by the Census Bureau; similar fnlwgt values indicate similar demographic characteristics. Education: Highest education level completed, such as HS-grad, Bachelor's, or Doctorate. Education.num: Numerical representation of the education level, e.g., HS-grad=9, Bachelor's=13. Marital Status: Marital situation, including Married, Divorced, Separated, etc. Occupation: Job category, with roles like Tech-support, Sales, Exec-managerial, and Craft-repair. Relationship: Family role, like Husband, Wife, Own-child, or Unmarried. Race: Racial category, such as White, Black, Asian-Pac-Islander, etc. Sex: Male or Female. Capital Gain: Continuous numerical value representing gains from investments. Capital Loss: Continuous numerical value representing losses from investments. Hours per Week: Average weekly work hours. Native Country: Country of origin, with options like United States, Germany, Japan, etc. Income: Income category, either <=50K or >50K. When generating a counterfactual, simply alter the values of the dictionary format as requested by the user (e.g., adjusting age, changing occupation, or modifying marital status) and provide the counterpart based on these new attributes. Do not explain your calculations—focus on delivering concise, realistic responses that align with the altered criteria provided by the user. For the response simply replace the values of the dictionary provided with no extra information and keeping it in this same format. Don't update the index."     
     },
]}

with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    counterfactual = None
    headers = None

    for row in reader:
        if not row:
            continue
        
        # If it's defining the CF grab it
        if 'CF' in row[1]:
            counterfactual = row[1].split()[1]
            continue

        # Set headers when detected
        if headers is None:
            headers = row
            continue
            

        data_entry = dict(zip(headers, row))

        formatted_data = {
            'Index': int(data_entry['Index']),
            'Age': int(data_entry['age']),
            'Workclass': data_entry['workclass'],
            'fnlwgt': int(data_entry['fnlwgt']),
            'Education': data_entry['education'],
            'Education.num': int(data_entry['education.num']),
            'Marital.status': data_entry['marital.status'],
            'Occupation': data_entry['occupation'],
            'Relationship': data_entry['relationship'],
            'Race': data_entry['race'],
            'Sex': data_entry['sex'],
            'Capital.gain': data_entry['capital.gain'],
            'Capital.loss': data_entry['capital.loss'],
            'Hours.per.week': data_entry['hours.per.week'],
            'Native.country': data_entry['native.country'],
            'Income': data_entry['income'],
            'Counterfactual.request': counterfactual
        }


        json_output['messages'].append({
            'role': 'user',
            'content': formatted_data
        })

with open(output_file, 'w') as file:
    json.dump(json_output, file, indent=4)
