from transformers import file_utils
import transformers
import torch
from huggingface_hub import login

login(token='')


# Original Prompt
orginal_messages = [
    # Telling the model how to behave and how to respond to the user when given information.
    # Describing it's main purpose without much detail on how it got it's conclusion.
    {"role": "system", "content": " you are a counterfacual estimator of the individual that user gives. \
    The user provides the features of the real-world individual. Then you estimate the counterpart based on\
    the requested feature like sex, race, age, and etc. do not provide any explanation on how you estimate, \
    just give the features of the counterpart same as the provided features by the user." },

    # User requesting counterfactual estimates.
    {"role": "user", "content":"I am Male, 42, income 80K. what would be my 41 years old counterfactual?"},
    # Add other census features to user prompt to  create counterfactuals.
]


#-------------------------------Define System Prompt-------------------------------#

sys_messages = []

# Small System prompt describing a bit of the features
sys_messages += [
    {"role": "system", "content": "You are a counterfactual estimator that creates realistic hypothetical scenarios based \
    on user provided census data attributes. Given demographic and socioeconomic details (e.g., age, workclass, education, \
    marital status, occupation, race, sex, financial information, work hours, and country), generate a counterpart by \
    adjusting one or more specified attributes. Provide concise responses without explaining your calculations focus on \
    realistic, data-driven estimates."}
]

# Medium System prompt describing some features
sys_messages += [
    {"role": "system", "content": "You are a counterfactual estimator designed to create hypothetical scenarios based on \
    census style data attributes, helping users explore 'what-if' scenarios. Users provide demographic, occupational, and \
    socioeconomic information like age, workclass, education, marital status, and country of origin. Your job is to adjust \
    specified attributes to generate a counterfactual estimate. For example, if a user provides their age, marital status, \
    and occupation, you may be asked to estimate how their scenario would change if they were older, divorced, or in a \
    different occupation. When responding, maintain internal consistency with the altered attributes and avoid explaining \
    the calculations. Simply provide the counterfactual details based on the user's new criteria."}
]

# Large System prompt describing all the features and what to expect
sys_messages += [
    {"role": "system", "content": "You are a counterfactual estimator trained to generate hypothetical scenarios based on \
    a variety of census-style data attributes, assisting users in exploring alternative realities. Each user input contains \
    detailed demographic, socioeconomic, and occupational information about a real-world individual. Your role is to \
    produce a realistic counterfactual estimate by adjusting one or more specified attributes while maintaining logical \
    consistency. The data attributes provided include: Age: Continuous numerical value representing the individual's \
    age. Workclass: Employment category, such as Private, Federal-gov, Self-employed, or Without-pay. Education: Highest \
    education level completed, such as HS-grad, Bachelor's, or Doctorate. Marital Status: Marital situation, including \
    Married, Divorced, Separated, etc. Occupation: Job category, with roles like Tech-support, Sales, Exec-managerial, and \
    Craft-repair. Relationship: Family role, like Husband, Wife, Own-child, or Unmarried. Race: Racial category, such as \
    White, Black, Asian-Pac-Islander, etc. Sex: Male or Female. Capital Gain and Capital Loss: Continuous numerical values \
    representing gains or losses from investments. Hours per Week: Average weekly work hours. Native Country: Country of \
    origin, with options like United States, Germany, Japan, etc. When generating a counterfactual, simply alter the \
    attributes as requested by the user (e.g., adjusting age, changing occupation, or modifying marital status) and \
    provide the counterpart based on these new attributes. Do not explain your calculations—focus on delivering concise, \
    realistic responses that align with the altered criteria provided by the user."}
]


#-------------------------------Define User Prompt-------------------------------#

user_messages = []

# User prompts with all the features:

# Sex: Female, Wife
user_messages += [{'role': 'user', 'content': "I am a 35-year-old Female, working in the private sector, with a capital gain of \
5000 and no capital loss, working 40 hours per week. I have a Bachelor's degree, am married-civ-spouse, and my relationship\
status is 'Wife.' I work as a Prof-specialty and am of Asian-Pac-Islander race. My native country is the United States. \
What would my counterfactual look like if I were Male?"}]

# Sex: Female, Wife
user_messages += [{'role': 'user', 'content': "I am a 45-year-old Female, employed in local government, with a capital gain of \
0 and a capital loss of 1500, working 35 hours per week. I completed HS-grad, am married-civ-spouse, and am 'Wife' in the \
family structure. I work in Tech-support, identify as White, and my native country is Canada. What would my scenario look \
like if I were Male?"}]

# Sex: Male, Husband
user_messages += [{'role': 'user', 'content': "I am a 50-year-old Male, self-employed (not incorporated), with a capital gain \
of 10000 and a capital loss of 0, working 60 hours per week. I hold a Master's degree, am married-civ-spouse, and have a \
relationship status of 'Husband.' I work as an Exec-managerial, am Black, and my native country is India. How would my \
counterfactual look if I were Female?"}]

# Sex: Male, Husband
user_messages += [{'role': 'user', 'content': "I am a 40-year-old Male, working in the federal government, with no capital gain \
but a capital loss of 500, working 50 hours per week. I have an Assoc-acdm degree, am married-civ-spouse, with a \
relationship status of 'Husband.' I work in Farming-fishing, am White, and am originally from Germany. What would my \
counterpart be if I were Female?"}]


#-------------------------------Define Model and Dataset-------------------------------#
def setup_model():
    # Llama model version we're using
    model_id = 'meta-llama/Meta-Llama-3-8B-instruct'

    # Create a text generation pipeline that uses the model defined above
    pipline = transformers.pipeline(
        'text-generation',
        model=model_id,
        model_kwargs={'torch_dtype': torch.bfloat16},
        device_map='auto',
        pad_token_id=50256
    )

    print('Set up finished')
    return pipline


#-------------------------------Generate Output-------------------------------#
def generate_counterfactual(messages, pipeline):
    # Converts the conversation in messages into a single text prompt to give to the llm
    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generates the prompts response.
    print('Starting generation......')
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.99,
    )
    print(outputs[0]["generated_text"][len(prompt):])

if __name__ == '__main__':
    pipeline = setup_model()

    # for sys_message in sys_messages:
    #     for user_message in user_messages:
    #         message = [sys_message, user_message]

    message = [sys_messages[1], user_messages[0]]
    generate_counterfactual(messages=message, pipeline=pipeline)
