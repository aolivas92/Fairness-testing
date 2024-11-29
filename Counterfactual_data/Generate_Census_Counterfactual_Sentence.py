import sys
import time
import ollama
import json
import openai
import anthropic

input_file = './3_Census_Counterfactual.json'
output_file = './Generated_Counterfactuals.json'
num_retries = 0
num_format_errors = 0
num_errors = 0
num_max_retries = 0
max_retries = 5

def main(model):
    global num_retries
    global num_format_errors
    global num_errors
    global num_max_retries
    too_many_reties = 100
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
    print(system_message['content'], end='\n\n')
    assistant_responses = []
    assistant_responses.append({'role': 'system', 'content': system_message['content']})

    # TODO: Remove temporary code
    count = 0

    total_generation_time = time.time()
    for i, user_message in enumerate(user_messages):
        assistant_responses.append({'role': 'user', 'content': user_message['content']})
        # Convert the user message content to a string
        user_message_content = json.dumps(user_message['content'])
        print('USER MESSAGE:')
        print(user_message_content)

        # Message for llama and gpt
        message = None
        if model != 'claude':
            message = [
                {'role': 'system', 'content': system_message['content']},
                {'role': 'user', 'content': user_message_content}
            ]
        else:
            message = [
                {'role': 'user', 'content': user_message_content},
            ]


        # Call the different LLM's
        response = None
        if model == 'llama3':
            response = llama3_8b_generator(message=message)
        elif model == 'llama3.1':
            response = llama31_8b_generator(message=message)
        elif model == 'claude':
            response = claude3_generator(message=message, system_message=system_message['content'])
        elif model == 'gpt-4o-mini':
            response = gpt_4o_mini_generator(message=message)
        elif model == 'gpt-4o':
            response = gpt_4o_generator(message=message)
        else:
            break
        
        # Add the extra information that was missing
        data = {'role': 'user', 'content': response}
        assistant_responses.append(data)

        if num_retries >= too_many_reties:
            print(f'\nMAX NUMBER OF TOTAL RETRIES ALLOWED HIT AT PROMPT: {i} \n')
            break

        if count >= 20:
            break
        count += 1


    print(f'Total Generation Time = {time.time()-total_generation_time}')
    print(f'Number of retries = {num_retries}')
    print(f'Number of format errors = {num_format_errors}')
    print(f'Number of unkown errors = {num_errors}')
    print(f'Number of max retries hit = {num_max_retries}')
    # Upload the generated data to a file
    output_data = {'model': '','messages': assistant_responses}
    export_data(output_data, output_file)

def llama3_8b_generator(message):
    global num_max_retries
    global max_retries
    retries = 0
    valid_response = False

    start_generation_time = time.time()

    while not valid_response and retries != max_retries:
        response = ollama.chat(model='llama3:8b', messages=message, format='json')
        converted_response = json.loads(response['message']['content'])
        valid_response = check_response(converted_response)

        retries += 1

    if not valid_response:
        print('MAX RETRIES HIT\n')
        num_max_retries += 1
        return 'error retries ran out'

    print('GENERATED TEXT:')
    print(converted_response)

    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')
    
    return converted_response

def llama31_8b_generator(message):
    global num_max_retries
    global max_retries
    retries = 0
    valid_response = False

    start_generation_time = time.time()

    while not valid_response and retries != max_retries:
        response = ollama.chat(model='llama3.1:8b', messages=message, format='json')
        converted_response = json.loads(response['message']['content'])
        valid_response, attribute_missed = check_response(converted_response)

        retries += 1
        print('\n\nMESSAGE:', message['user']['content'], '\n\n')

    if not valid_response:
        print('MAX RETRIES HIT\n')
        num_max_retries += 1
        return 'error retries ran out'

    print('GENERATED TEXT:')
    print(converted_response)

    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')
    
    return converted_response

def gpt_4o_mini_generator(message):
    global num_format_errors
    global num_errors
    global num_max_retries
    global max_retries
    start_generation_time = time.time()
    valid_response = False
    retries = 0

    keyfile = open('gpt_key.txt', 'r')
    openai.api_key = keyfile.readline().rstrip()
    keyfile.close()
    temperature=0.7

    while not valid_response and retries != max_retries:
        response = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=message,
            # temperature=temperature,
        )
        try:
            converted_response = json.loads(response.choices[0].message.content)
            valid_response = check_response(converted_response, dictionary=False)
        except json.decoder.JSONDecodeError:
            print("ERROR: response not complete and JSON can't convert it")
            num_format_errors +=1
        except Exception as e:
            print("ERROR: ", e)
            num_errors += 1

        retries += 1
    
    if not valid_response:
        print('MAX RETRIES HIT\n')
        num_max_retries += 1
        return 'error retries ran out'

    print('GENERATD TEXT:')
    print(converted_response)
    
    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')

    return converted_response

def gpt_4o_generator(message):
    global num_format_errors
    global num_errors
    global num_max_retries
    global max_retries
    start_generation_time = time.time()
    valid_response = False
    retries = 0

    keyfile = open('gpt_key.txt', 'r')
    openai.api_key = keyfile.readline().rstrip()
    keyfile.close()
    temperature=0.7

    while not valid_response and retries != max_retries:
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=message,
            # temperature=temperature,
        )
        try:
            converted_response = json.loads(response.choices[0].message.content)
            valid_response = check_response(converted_response, dictionary=False)
        except json.decoder.JSONDecodeError:
            print("ERROR: response not complete and JSON can't convert it")
            num_format_errors +=1
        except Exception as e:
            print("ERROR: ", e)
            num_errors += 1

        retries += 1
    
    if not valid_response:
        print('MAX RETRIES HIT\n')
        num_max_retries += 1
        return 'error retries ran out'

    print('GENERATD TEXT:')
    print(converted_response)
    
    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')

    return converted_response
        
def claude3_generator(message, system_message):
    global num_format_errors
    global num_errors
    global num_max_retries
    global max_retries
    start_generation_time = time.time()
    valid_response = False
    retries = 0

    keyfile = open('claude_key.txt', 'r')
    api_key = keyfile.readline().rstrip()
    keyfile.close()
    client = anthropic.Anthropic(api_key=api_key)

    while not valid_response and retries != max_retries:
        response = client.messages.create(
            model='claude-3-haiku-20240307',
            system=system_message,
            messages=message,
            max_tokens=1000,
        )
        try:
                converted_response = json.loads(response.content[0].text)
                valid_response = check_response(converted_response, dictionary=False)
        except json.decoder.JSONDecodeError:
            print("ERROR: response not complete and JSON can't convert it")
            num_format_errors +=1
        except Exception as e:
            print("ERROR: ", e)
            num_errors += 1

        retries += 1
    
    if not valid_response:
        print('MAX RETRIES HIT\n')
        num_max_retries += 1
        return 'error retries ran out'

    print('GENERATD TEXT:')
    print(converted_response)
    
    end_generation_time = time.time() - start_generation_time
    print(f'Time for message above: {end_generation_time}\n')

    return converted_response

def check_response(converted_response, dictionary=True):
    global num_retries
    valid = True
    attributes = {'Index', 'Age', 'Workclass', 'fnlwgt', 'Education', 'Education.num', 'Marital.status',
     'Occupation', 'Relationship', 'Race', 'Sex', 'Capital.gain', 'Capital.loss',
     'Hours.per.week', 'Native.country', 'Income'}
    
    # Verify the response has every attribute above
    for attribute in attributes:
        if dictionary and attribute not in converted_response.keys():
            valid = False
            break
        if not dictionary and attribute not in converted_response:
            valid = False
            break
    
    if not valid:
        print('RESPONSE FAILED VERIFICATION WITH:', attribute)
        print("RESPONSE:")
        print(converted_response)
        num_retries += 1
        return False, attribute
    return True, None

def load_data(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def export_data(data, file_path):
    with open(file_path, 'w') as file:  
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    models = ['llama3', 'llama3.1', 'claude', 'gpt-4o-mini', 'gpt-4o']
    
    model = None
    try:
        model = sys.argv[1]
    except IndexError:
        print('ERROR no model argument provided')   
        sys.exit() 

    if model in models:
        main(model)
    else:
        print(f'MODEL INCORRECT OR NOT PROVIDED, models: \n{models}')
        sys.exit()
