import time
import ollama
import json

input_file = './Counterfactual_input.json'
output_file = './Generated_counterfactuals.json'

def load_data(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def export_data(data, file_path):
    with open(file_path, 'w') as file:  
        json.dump(data, file, indent=4)

def llama3_8b_generator(message):
    start_generation_time = time.time()

    response  = ollama.chat(model='llama3:8b', messages=message, format='json')
    print('GENERATED TEXT:')
    # print(response['message']['content'])
    print()
    # print(json.loads(response['message']['content']))

    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')
    return response['message']['content']

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

print('Starting Generation...\n')
# Prompt llama3:8b to generate the counterfactuals
assistant_responses = []
with open(output_file, 'a') as file:
    for user_message in user_messages:
        user_message_content = json.dumps(user_message['content'])
        message = [
            {'role': 'system', 'content': system_message['content']},
            {'role': 'user', 'content': user_message_content}
        ]
        response = llama3_8b_generator(message=message)
        data = {'messages': response}
        json.dump(data, file, indent=4)
        # assistant_responses.append(response)
        # print(assistant_responses)
        break

# output_data = {'messages': assistant_responses}
# export_data(output_data, output_file)
