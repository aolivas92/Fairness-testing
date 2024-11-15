import time
import ollama
import json

input_file = './Census_Counterfactual.json'
output_file = './Generated_Counterfactuals.json'

def main():
    print("Loading data...\n")
    data = load_data(input_file)

    # Split the system and user prompts from the json file
    system_message = None
    user_messages = []

    for msg in data['messages']:
        if msg['role'] == 'system':
            system_message = msg
        if msg['role'] == 'user':
            user_messages += [msg]

    # Prompt llama3:8b to generate the counterfactuals
    print('Starting Generation...\n')
    print('SYSTEM MESSAGE:')
    print(system_message['content'])
    assistant_responses = []
    assistant_responses.append({'role': 'system', 'content': system_message['content']})

    for user_message in user_messages:
        assistant_responses.append({'role': 'user', 'content': user_message['content']})
        # Convert the user message content to a string
        user_message_content = json.dumps(user_message['content'])
        print('USER MESSAGE:')
        print(user_message_content)

        message = [
            {'role': 'system', 'content': system_message['content']},
            {'role': 'user', 'content': user_message_content}
        ]
        response = llama3_8b_generator(message=message)
        
        # Add the extra information that was missing
        data = {'role': 'user', 'content': response}
        assistant_responses.append(data)

    # Upload the generated data to a file
    output_data = {'messages': assistant_responses}
    export_data(output_data, output_file)

def llama3_8b_generator(message):
    start_generation_time = time.time()
    valid_response = False

    while not valid_response:
        response = ollama.chat(model='llama3.1:8b', messages=message, format='json')
        converted_response = json.loads(response['message']['content'])
        valid_response = check_response(converted_response)

    print('GENERATED TEXT:')
    print(converted_response)

    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')
    
    return converted_response

def check_response(converted_response):
    attributes = {'Index', 'Age', 'Workclass', 'fnlwgt', 'Education', 'Education.num', 'Marital.status',
     'Occupation', 'Relationship', 'Race', 'Sex', 'Capital.gain', 'Capital.loss',
     'Hours.per.week', 'Native.country', 'Income'}
    
    # Verify the response has every attribute above
    for attribute in attributes:
        if attribute not in converted_response.keys():
            print('RESPONSE FAILED VERIFICATION WITH:', attribute)
            return False
    
    return True

def load_data(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def export_data(data, file_path):
    with open(file_path, 'w') as file:  
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    main()
