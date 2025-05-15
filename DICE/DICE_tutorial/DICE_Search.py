import numpy as np
import csv
from itertools import product # Computes the Cartesian product of input iterables (useful 
# for generating all combinations of parameters)
import itertools
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import sys, os
sys.path.append("../")
import copy
import pandas as pd
from tensorflow.python.platform import flags
from scipy.optimize import basinhopping
from scipy.stats import entropy
import time
from sklearn.preprocessing import LabelEncoder
import ollama
import json
import anthropic
import regex
# TODO: Delete Later
import logging

from DICE_data.census import census_data, census_data2
from DICE_data.credit import credit_data
from DICE_data.compas import compas_data, compas_data2
from DICE_data.default import default_data, default_data2
from DICE_data.bank import bank_data, bank_data2
from DICE_data.heart import heart_data, heart_data2
from DICE_data.diabetes import diabetes_data
from DICE_data.students import students_data
from DICE_data.meps15 import meps15_data
from DICE_data.meps16 import meps16_data
from DICE_data.law_school import law_school_data2

from DICE_model.tutorial_models import dnn
from DICE_utils.utils_tf import model_prediction, model_argmax , layer_out
# TODO: Restore config import
from DICE_utils.config import credit, diabetes, students , meps15, meps16
from DICE_utils.config2 import bank, census, heart, default, compas, lawschool
from DICE_tutorial.utils import cluster, gradient_graph
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help='The name of dataset: census, credit, bank, default, meps21 ', required=True)
parser.add_argument("-sensitive_index", help='The index for sensitive features', required=True)
parser.add_argument("-timeout", help='Max. running time', default = 3600, required=False)
parser.add_argument("-RQ", help='1 for RQ, 2 for RQ2', default = 1, required=False)
args = parser.parse_args()

FLAGS = flags.FLAGS

counterfactual_number = 0

# step size of perturbation
perturbation_size = 1

def check_for_error_condition(conf, sess, x, preds, t, sens_params, input_shape, epsillon):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """

    t = [t.astype('int')]   
    samples = m_instance( np.array(t), sens_params, conf ) 
    pred = pred_prob(sess, x, preds, samples , input_shape )
    partition = clustering(pred,samples, sens_params , epsillon)
    #entropy_min = np.log2(len(partition)-1)#sh_entropy(pred, ent_tresh)
    #entropy_sh = sh_entropy(pred, epsillon)
    

    return  max(list(partition.keys())[1:]) - min(list(partition.keys())[1:]), \
                                            len(partition)-1, conf#(len(partition) -1),
    
def seed_test_input(clusters, limit):
    """
    Select the seed inputs for fairness testing
    :param clusters: the results of K-means clustering
    :param limit: the size of seed inputs wanted
    :return: a sequence of seed inputs
    """
    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])
    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]
            rows.append(row)
            if len(rows) == limit:
                break
        i += 1
    return np.array(rows)

def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input

class Local_Perturbation(object):
    """
    The  implementation of local perturbation
    """

    def __init__(self, sess, grad, x, n_values, sens_params, input_shape, conf):
        """
        Initial function of local perturbation
        :param sess: TF session
        :param grad: the gradient graph
        :param x: input placeholder
        :param n_value: the discriminatory value of sensitive feature
        :param sens_param: the index of sensitive feature
        :param input_shape: the shape of dataset
        :param conf: the configuration of dataset
        """
        self.sess = sess
        self.grad = grad
        self.x = x
        self.n_values = n_values
        self.input_shape = input_shape
        self.sens = sens_params
        self.conf = conf

    def __call__(self, x):
        """
        Local perturbation
        :param x: input instance for local perturbation
        :return: new potential individual discriminatory instance
        """

        # perturbation
        s = np.random.choice([1.0, -1.0]) * perturbation_size

        n_x = x.copy()
        for i in range(len(self.sens)):
            n_x[self.sens[i] - 1] = self.n_values[i]
       
        # compute the gradients of an individual discriminatory instance pairs
        ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([x])})
        n_ind_grad = self.sess.run(self.grad, feed_dict={self.x:np.array([n_x])})

        if np.zeros(self.input_shape).tolist() == ind_grad[0].tolist() and \
           np.zeros(self.input_shape).tolist() == n_ind_grad[0].tolist():
            
            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens :
                probs[sens - 1] = 0

                

        else:
            # nomalize the reciprocal of gradients (prefer the low impactful feature)
            grad_sum = 1.0 / (abs(ind_grad[0]) + abs(n_ind_grad[0]))

            for sens in self.sens :
                grad_sum[ sens - 1 ] = 0

            probs = grad_sum / np.sum(grad_sum)
        probs = probs / probs.sum()
        if True in np.isnan(probs):
            probs = 1.0 / (self.input_shape) * np.ones(self.input_shape)

            for sens in self.sens :
                probs[sens - 1] = 0
            probs = probs/probs.sum()


        # randomly choose the feature for local perturbation
        index = np.random.choice(range(self.input_shape) , p=probs)
        local_cal_grad = np.zeros(self.input_shape)
        local_cal_grad[index] = 1.0
        x = clip(x + s * local_cal_grad, self.conf).astype("int")
        return x
                
#--------------------------------------
def m_instance( sample, sens_params, conf):
    index = []
    m_sample = []
    for sens in sens_params:
        index.append([i for i in range(conf.input_bounds[sens - 1][0], conf.input_bounds[sens - 1][1] + 1)])
      
    for ind in list(product(*index)):     
        temp = sample.copy()
        for i in range(len(sens_params)):
            temp[0][sens_params[i] - 1] = ind[i]
        m_sample.append(temp)
    return np.array(m_sample)

census_mapping = {
    
}

def m_instance_real_counterfactual(sample, sens_params, conf):
    # Grab the data configuration
    system_message = conf.system_message
    label_encoders = conf.label_encoders
    categorical_unique_values = conf.categorical_unique_values
    col_names = conf.feature_name

    # TODO: Delete later
    global counterfactual_number
    counterfactual_number += 1
    logging.info(f'M_INSTANCE CALL: {counterfactual_number}')
    logging.info(f'SAMPLE: {sample}')

    #TODO: Delete later, used for testing.
    # print('\n\nSAMPLE:', sample)
    # print('SENS_PARAM:', sens_params)
    # print('\n\nCONF:', conf, '\n\n')

    # Convert the sample data back to categorical data.
    decoded_sample = decode_sample(sample, label_encoders, categorical_unique_values, col_names)
    # print('\n\nDECODED SAMPLE:', decoded_sample)
    formatted_data = data_formatter(decoded_sample, sens_params, col_names)
    formatted_data = str(formatted_data)
    # print('\n\nFORMATTED DATA:', formatted_data)
    logging.info(f'DECODED SAMPLE FORMATTED: {formatted_data}')

    valid_response = False
    max_retries = 10
    tries = 0
    llm = 0
    error_msg = ""

    # Run the LLM and check the response.
    while not valid_response:
        tries += 1
        logging.info(f'LLM RUN: {tries}')

        if tries > max_retries:
            error_msg = (
                "The counterfactual you generated is missing one or more required features. "
                "Ensure that the generated counterfactual includes every feature exactly as specified in 'col_names'."
                )   
            print(f'\n\n{error_msg}\n\n')
            logging.critical(error_msg)
            return None

        # Add error to system message so the LLM can fix it.
        system_message = system_message + " " + error_msg

        if llm == 0:
            converted_response = llama31_8b_generator(system_message, 
                                            user_message=formatted_data, 
                                            col_names=col_names, 
                                            label_encoders=label_encoders, 
                                            categorical_unique_values=categorical_unique_values)
        elif llm == 1:
            converted_response = claude3_generator(system_message,
                                        user_message=formatted_data,
                                        col_names=col_names,
                                        label_encoders=label_encoders,
                                        categorical_unique_values=categorical_unique_values)
            
        logging.info(f'Generated Counterfactual: {converted_response}')
            
        # Verify that the generated response is valid.
        try:
            # Check that all the columns are included.
            valid_response = check_response(converted_response, col_names)
            if not valid_response:
                error_msg = (
                    "The counterfactual generated is missing required columns or has invalid formatting. "
                    "Please ensure all required data fields are included and properly formatted in the response."
                    )
                print(f'\n\n{error_msg}\n\n')
                logging.error(error_msg)
                continue

            # Encode the response
            encoded_response = encode_sample(converted_response, label_encoders, categorical_unique_values)
            # print('\n\nENCODED SAMPLE:', encoded_response)

            for sens_param in sens_params:
                sens_col_name = col_names[sens_param]

                original_value = str(decoded_sample[sens_param])
                counterfactual_value = str(converted_response[sens_col_name])

                # Use regex to compare the original counterfactual and the new one, make sure they are different
                max_errors = 2 # Allow for two similar characters
                escaped_original = regex.escape(original_value)
                fuzzy_pattern = f"({escaped_original}){{e<={max_errors}}}"
                match = regex.match(fuzzy_pattern, counterfactual_value)
                
                if match:
                    error_msg = (
                        f"The generated counterfactual didn't sufficiently change the sensitive attribute '{sens_col_name}'. "
                        f"Original value: '{original_value}', Counterfactual value: '{counterfactual_value}'. "
                        f"The values are too similar - please ensure they are substantially different."
                        )
                    print(f'\n\n{error_msg}\n\n')
                    logging.error(error_msg)
                    valid_response = False
                    break

        except Exception as e:
            error_msg = (
                f"Encoding the generated counterfactual failed due to an invalid format or value. "
                f"Please ensure all provided values match the expected categorical and numerical types. "
                f"Specific error: {e}"
                )
            print(f'\n\n{error_msg}\n\n')
            logging.error(error_msg)
            valid_response = False

    # Convert the original sample and the generated counterfactual into the correct types.
    og_sample = np.array(sample, dtype=np.float64)
    new_sample = np.array([np.array(list(encoded_response.values()), dtype=np.float64)])
    m_sample = np.array([og_sample, new_sample])
    # print('\n\nM SAMPLE:', m_sample, '\n\n')
    print('Finished generating Counter Factual Successfully.')
    logging.info(f'Finished generating Counter Factual Successfully. Tries: {tries}')

    return m_sample

def decode_sample(sample, label_encoders, categorical_unique_values, col_names):
    decoded_sample = list(sample[0])

    for indx, col in enumerate(col_names):
        if col in categorical_unique_values.keys():
            encoded_value = int(sample[0][indx])
            decoded_value = label_encoders[col].inverse_transform([encoded_value])[0]
            decoded_sample[indx] = decoded_value

    return decoded_sample

def encode_sample(sample, label_encoders, categorical_unique_values):
    # NOTE: Fails to encode a variable that it hasn't seen before.
    encode_sample = sample.copy()

    # Remove 'Counterfactual.request' if the LLM gives it
    if 'Counterfactual.request' in encode_sample.keys():
        del encode_sample['Counterfactual.request']

    for col, encoder in label_encoders.items():
        cat_value = encode_sample[col]
        closest_cat_value = [find_closest_regex_match(cat_value, categorical_unique_values[col])]
        # print('GENERATED CAT VALUE:', cat_value, '\t\tCLOSEST_CAT_VALUE:', closest_cat_value)
        encoded_value = encoder.transform(closest_cat_value)[0]
        encode_sample[col] = encoded_value

    return encode_sample

def find_closest_regex_match(sample, choices, max_errors = 10):
    # Escape the sample to treat special characters literally
    escaped_sample = regex.escape(sample)

    # Fuzzy Patter Format: (patter){e<=max_errors}
    fuzzy_pattern = f"({escaped_sample}){{e<={max_errors}}}"

    # Compile the pattern
    compiled_pattern = regex.compile(fuzzy_pattern)

    best_choice = None
    best_error_count = float("inf")

    for choice in choices:
        match = compiled_pattern.search(choice)
        if match:
            # fuzzy__counts -> (insertions, deletions, substitutions)
            insertions, deletions, substitutions = match.fuzzy_counts
            total_errors = insertions + deletions + substitutions
            if total_errors == 0:
                best_choice = choice
                return best_choice
            elif total_errors < best_error_count:
                best_error_count = total_errors
                best_choice = choice

    if best_choice is None:
        error_msg = f"Failed to find a closest match for: {sample}"
        logging.error(error_msg)
        raise Exception(error_msg)
    

    return best_choice

def data_formatter(sample, sens_params, col_names):
    map = {value: index for index, value in enumerate(col_names)}

    # Build a reverse mapping: index -> name
    index_to_name = {v: k for k, v in map.items()}

    # For each integer in sens_params, look up its string
    cf_list = [index_to_name[val] for val in sens_params]

    # Join into a single comma-separated string
    counterfactuals = ",".join(cf_list)


    formatted_data = dict()
    for category, indx in map.items():
        try:
            formatted_data[category] = int(sample[indx])
        except Exception:
            formatted_data[category] = sample[indx]
    
    formatted_data['Counterfactual.request'] = counterfactuals

    return formatted_data

def llama31_8b_generator(system_message, user_message, col_names, label_encoders, categorical_unique_values):
    # retries = 0
    # max_retries = 3
    # valid_response = False

    message = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    # Keep trying until all the column names appear in the response and all the responses can be converted.
    # while not valid_response:
    response = ollama.chat(model='llama3.1:8b', messages=message, format='json')
    converted_response = json.loads(response['message']['content'])
    
    return converted_response

def claude3_generator(system_message, user_message, col_names, label_encoders, categorical_unique_values):
    valid_response = False
    max_retries = 5
    retries = 0

    # Grab the key from the local file.
    keyfile = open('../../Counterfactual_data/claude_key.txt', 'r')
    api_key = keyfile.readline().rstrip()
    keyfile.close()
    client = anthropic.Anthropic(api_key=api_key)

    # Set up the user message.
    user_message = [{'role': 'user', 'content': user_message}]

    # while not valid_response:
    response = client.messages.create(
        model='claude-3-haiku-20240307',
        system=system_message,
        messages=user_message,
        max_tokens=2000,
    )
    try:
        converted_response = json.loads(response.content[0].text)
        # print('\n\nRESPONSE:', converted_response)
    except Exception as e:
        print("FAILED TO CONVERT: ", e)
        return None

    return converted_response

def check_response(converted_response, col_names, dictionary=True):
    valid = True
    columns = set(col_names)
    
    # Verify the response has every attribute above
    for column in columns:
        if dictionary and column not in converted_response.keys():
            valid = False
            break
        if not dictionary and column not in converted_response:
            valid = False
            break
    
    if not valid:
        print('RESPONSE FAILED VERIFICATION WITH:', column)
        return False
    return True

def global_sample_select(clus_dic, sens_params):
    leng = 0
    for key in clus_dic.keys():
        if key == 'Seed':
            continue
        if len(clus_dic[key]) > leng:
            leng = len(clus_dic[key])
            largest = key
    
    sample_ind = np.random.randint(len(clus_dic[largest]))
    n_sample_ind = np.random.randint(len(clus_dic[largest]))
    
    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] -1] = clus_dic[largest][sample_ind][i]
    # returns one sample of largest partition and its pair
    return np.array([sample]),clus_dic[largest][n_sample_ind]


def local_sample_select(clus_dic, sens_params):
      
    k_1 = min(list(clus_dic.keys())[1:])
    k_2 = max(list(clus_dic.keys())[1:])
    
    sample_ind = np.random.randint(len(clus_dic[k_1]))
    n_sample_ind = np.random.randint(len(clus_dic[k_2]))

    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i] -1] = clus_dic[k_1][sample_ind][i]
    return np.array([sample]),clus_dic[k_2][n_sample_ind]
    

def clustering(probs,m_sample, sens_params, epsillon):
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0][0]
    bins= np.arange(0, 1, epsillon )
    digitized = np.digitize(probs, bins) - 1
    for  k in range(len(digitized)):

        if digitized[k] not in cluster_dic.keys():        
            cluster_dic[digitized[k]]=[ [m_sample[k][0][j - 1] for j in sens_params]]
        else:
            cluster_dic[digitized[k]].append( [m_sample[k][0][j - 1] for j in sens_params])
    return cluster_dic 
    
def pred_prob(sess, x, preds, m_sample, input_shape):
        probs = model_prediction(sess, x, preds, np.array(m_sample).reshape(len(m_sample),
                                                                input_shape[1]))[:,1:2].reshape(len(m_sample))
        return probs

def sh_entropy(probs,bin_thresh, base=2 ):
    bins = np.arange(0, 1,bin_thresh )
    digitized = np.digitize(probs, bins)
    value,counts = np.unique(digitized, return_counts=True)   
    return entropy(counts, base=base)
     
    
def dnn_fair_testing(dataset, sens_params, model_path, cluster_num, 
                     max_global, max_local, max_iter, epsillon, timeout, RQ):
    """
    
    The implementation of DICE
    :param dataset: the name of testing dataset
    :param sensitive_param: the index of sensitive feature
    :param model_path: the path of testing model
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :param max_global: the maximum number of samples for global search
    :param max_local: the maximum number of samples for local search
    :param max_iter: the maximum iteration of global perturbation
    """
    # TODO: Return the census_data and bank_data to the original version.
    data = {"census":census_data2, "credit":credit_data, "bank":bank_data2, "compas":compas_data2, 
            "default": default_data2, "heart":heart_data2, "diabetes":diabetes_data, 
            "students":students_data, "meps15":meps15_data, "meps16":meps16_data, "lawschool":law_school_data2}
    # NOTE: Grabs the config of each dataset.
    # TODO: Return the bank to the original version.
    data_config = {"census":census, "credit":credit, "bank":bank, "compas":compas, "default":default,
                  "heart":heart , "diabetes":diabetes,"students":students, "meps15":meps15, "meps16":meps16, "lawschool":lawschool}
    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()

    tf.set_random_seed(1234)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.allow_soft_placement= True

    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)   

    # ! ERROR: The `saver.restore()` is currently causing an error because there is no "../models/lawschool/test.model"
    preds = model(x)
    saver = tf.train.Saver()
    model_path = model_path + dataset + "/test.model"
    saver.restore(sess, model_path)

    # construct the gradient graph
    grad_0 = gradient_graph(x, preds)

    # build the clustering model
    clf = cluster(dataset, cluster_num)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # store the result of fairness testing
   
    global max_k
    global start_time
    global max_k_time

    print(dataset, sens_params)
    RQ2_table = []
    RQ1_table = []
    for trial in range(1):
        print('Trial', trial)
        if  sess._closed:
            sess = tf.Session(config=config)
            sess = tf.Session(config=config)
            x = tf.placeholder(tf.float32, shape=input_shape)
            y = tf.placeholder(tf.float32, shape=(None, nb_classes))
            model = dnn(input_shape, nb_classes)   
            preds = model(x)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            grad_0 = gradient_graph(x, preds)
        global_inputs = set()
        tot_inputs =set()
        global_inputs_list = []
        local_inputs = set()
        local_inputs_list = []
        seed_num = 0          
        max_k_time = 0
        max_k_time_list = []
        init_k_list  = []
        max_k_list = []
        #-----------------------
        def evaluate_local(inp):

            """
            Evaluate whether the test input after local perturbation is an individual discriminatory instance
            :param inp: test input
            :return: whether it is an individual discriminatory instance
            """ 
            global max_k
            global max_k_time
            global start_time
            global time1
            result, K ,conf = check_for_error_condition(data_config[dataset], sess, x, preds, inp, 
                                               sens_params, input_shape, epsillon)    
            if K > max_k:
                max_k = K 
                max_k_time = time.time() - start_time

            dis_sample =copy.deepcopy(inp.astype('int').tolist())   
            for sens in sens_params:
                dis_sample[sens - 1] = 0    
            if tuple(dis_sample) not in global_inputs and\
                                    tuple(dis_sample) not in local_inputs:

                local_inputs.add(tuple(dis_sample))
                local_inputs_list.append(dis_sample + [time.time() - time1])

            return (-1 * result)

        # select the seed input for fairness testing
        inputs = seed_test_input(clusters, min(max_global, len(X)))
        global time1
        time1 = time.time()  
        for num in range(len(inputs)):
            
            #clear_output(wait=True)
            start_time = time.time()
            if time.time()-time1 > timeout:
                break 
            print('Input ',seed_num)
            index = inputs[num]
            sample = X[ index : index + 1]
            
            # start global perturbation
            for iter in range( max_iter + 1 ):  
                          
                if time.time()-time1 > timeout :
                    break
                
                # m_sample = m_instance( np.array(sample) , sens_params, data_config[dataset] )
                m_sample = m_instance_real_counterfactual(np.array(sample), sens_params, data_config[dataset])
                # print('M_SAMPLE TYPE: ', type(m_sample))
                # print('M_SAMPLE[0] TYPE: ', type(m_sample[0]))
                # print('M_SAMPLE[0][0] TYPE: ', type(m_sample[0][0]))
                # print('M_SAMPLE[0][0][0] TYPE: ', type(m_sample[0][0][0]))

                # TODO: Add a continue if nothing is returned from m_sample.
                if m_sample is None:
                    continue

                pred = pred_prob( sess, x, preds, m_sample , input_shape )
                clus_dic = clustering( pred, m_sample, sens_params, epsillon )

                if iter == 0:
                    init_k = len(clus_dic) - 1
                    max_k = init_k

                if len(clus_dic) - 1 > max_k:
                    max_k = len(clus_dic) - 1
                    max_k_time = round((time.time() - start_time),4)

                sample,n_values = global_sample_select( clus_dic, sens_params )
                dis_sample = sample.copy()
                for sens in sens_params:
                    dis_sample[0][sens  - 1] = 0


                if tuple(dis_sample[0].astype('int')) not in global_inputs and\
                                    tuple(dis_sample[0].astype('int')) not in local_inputs:
                    dis_flag = True
                    global_inputs.add(tuple(dis_sample[0].astype('int')))
                    global_inputs_list.append(dis_sample[0].astype('int').tolist())

                else:
                    dis_flag = False


                if dis_flag and (len(clus_dic)-1 >= 2):    

                    loc_x,n_values = local_sample_select(clus_dic ,sens_params )                              
                    minimizer = {"method": "L-BFGS-B"}
                    local_perturbation = Local_Perturbation(sess, grad_0, x, n_values, 
                                                            sens_params, input_shape[1], 
                                                            data_config[dataset])               
                    basinhopping(evaluate_local, loc_x, stepsize = 1.0, 
                                 take_step = local_perturbation, minimizer_kwargs = minimizer, 
                                 niter = max_local)


                if dis_flag :
                    global_inputs_list[-1] +=  [time.time() - time1]


                clus_dic = {}
                if iter == max_iter:
                    break

                #Making up n_sample
                n_sample = sample.copy()
                for i in range(len(sens_params)):
                    n_sample[0][sens_params[i] - 1] = n_values[i]                

                # global perturbation

                s_grad = sess.run(tf.sign(grad_0), feed_dict = {x: sample})
                n_grad = sess.run(tf.sign(grad_0), feed_dict = {x: n_sample})

                # find the feature with same impact
                if np.zeros(data_config[dataset].params).tolist() == s_grad[0].tolist():
                    g_diff = n_grad[0]
                elif np.zeros(data_config[dataset].params).tolist() == n_grad[0].tolist():
                    g_diff = s_grad[0]
                else:
                    g_diff = np.array(s_grad[0] == n_grad[0], dtype = float)                
                for sens in sens_params:
                    g_diff[sens - 1] = 0 

                cal_grad = s_grad * g_diff
                if np.zeros(input_shape[1]).tolist() == cal_grad.tolist()[0]:
                    index = np.random.randint(len(cal_grad[0]) - 1)
                    for i in range(len(sens_params) - 1, -1, -1):
                        if index == sens_params[i] - 1 :
                            index = index + 1

                    cal_grad[0][index]  = np.random.choice([1.0, -1.0])
                # NOTE: Here the sample is checked to make sure each feature is in range.
                sample[0] = clip(sample[0] + perturbation_size * cal_grad[0], data_config[dataset]).astype("int")

            seed_num += 1
            if max_k > 1:
                max_k_time_list.append(max_k_time)
                init_k_list.append(init_k)
                max_k_list.append(max_k)
            
        print('Search Done!')
        if RQ == 1:
            
            # create the folder for storing the fairness testing result
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + dataset + '/'):
                os.makedirs('../results/' + dataset + '/')
            if not os.path.exists('../results/' + dataset + '/DICE/'):
                os.makedirs('../results/' + dataset + '/DICE/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ1/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ1/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ1/' + ''.join(str(i) for i in sens_params)+'_10runs/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ1/' + ''.join(str(i) for i in sens_params)+'_10runs/')
            # storing the fairness testing result
            np.save('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_'+str(trial)+'.npy', 
                    np.array(global_inputs_list))
            np.save('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_'+str(trial)+'.npy', 
                    np.array(local_inputs_list))
            total_inputs = np.concatenate((local_inputs_list,global_inputs_list), axis=0)
            np.save('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_inputs_'+str(trial)+'.npy',
                    total_inputs)
            # RQ1 & RQ2

            local_sam = np.array(local_inputs_list).astype('int32')
            global_sam = np.array(global_inputs_list).astype('int32')
            # Storing result for RQ1 table
            print('Analyzing the search results....')
            continue

            with open('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(global_inputs_list)):
                    m_sample = m_instance( np.array([global_inputs_list[ind][:input_shape[1]]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[global_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            with open('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(local_inputs_list)):
                    m_sample = m_instance( np.array([local_inputs_list[ind][:input_shape[1]]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[local_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            df_l = pd.read_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv',header=None)  
            df_g = pd.read_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv',header=None)

            df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:,:input_shape[1]])
            df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:,:input_shape[1]])
            g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1] ,
                                                                                  np.array(sens_params)-1)), aggfunc=np.sum)
            l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_l.columns[:-1] ,
                                                                               np.array(sens_params)-1)), aggfunc=np.sum)

            g_time = g_pivot.index[np.where((g_pivot['label'] > 0) & (g_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            l_time = l_pivot.index[np.where((l_pivot['label'] > 0) & (l_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            tot_time = np.sort(np.concatenate((l_time, g_time), axis=0 ))

            g_dis = (len(m_sample) - g_pivot.loc[(g_pivot['label'] > 0) & \
                                                 (g_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()
            l_dis = (len(m_sample) - l_pivot.loc[(l_pivot['label'] > 0) & \
                                                 (l_pivot['label'] < len(m_sample))]['label'].to_numpy()).sum()


            tot_df = pd.DataFrame(total_inputs)
            tot_df.columns=[i for i in range(input_shape[1])] + ['time']

            k = []
            disc = []
            tot_df['sh_entropy'] = 0
            for sam_ind in range(total_inputs.shape[0]): 

                m_sample = m_instance( np.array([total_inputs[:,:input_shape[1]][sam_ind]]) , sens_params, data_config[dataset] )
                pred = pred_prob( sess, x, preds, m_sample , input_shape )
                clus_dic = clustering( pred, m_sample, sens_params, epsillon )
                tot_df.loc[[sam_ind], 'sh_entropy'] = sh_entropy(pred, epsillon)
                if pred.max() > 0.5 and  pred.min()< 0.5:
                    disc.append(1)
                else:
                    disc.append(0)
                k.append(len(clus_dic) - 1)

            tot_df['k'] = k
            tot_df['disc'] = disc
            tot_df['min_entropy'] = round(np.log2(tot_df['k'] ),2)
            tot_dis = tot_df.loc[tot_df['disc'] == 1]
            tot_dis.to_csv('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_disc_' + str(trial)+'.csv')
            # reseting the TF graph for the next round
            sess.close()
            tf.reset_default_graph()
            haighest_k = np.sort(tot_df['k'].unique())[-3:]
            if len(haighest_k)>2:
                IK1F = np.where(tot_df['k']==haighest_k[2])[0].shape[0]
                IK2F = np.where(tot_df['k']==haighest_k[1])[0].shape[0]
                IK3F = np.where(tot_df['k']==haighest_k[0])[0].shape[0]

            else:
                IK1F = np.where(tot_df['k']==haighest_k[1])[0].shape[0]
                IK2F = np.where(tot_df['k']==haighest_k[0])[0].shape[0]
                IK3F = 0
            print('Global ID RQ1 =', g_dis)
            print('local  ID RQ1  =',  l_dis)
            print('Total loc samples  = ', len(local_sam)) 
            print('Total glob samples = ', len(global_sam)) 
            print('Total ID = ',g_dis + l_dis)

            global_succ = round( g_dis / (len(global_sam) * \
                                 len(m_sample)) * 100,1)
            local_succ = round(l_dis  / (len(local_sam) * \
                                 len(m_sample)) * 100,1)


            row = [len(total_inputs)] + [np.mean(init_k_list), np.mean(max_k_list),np.mean(max_k_time_list)] + list(tot_dis[['min_entropy', 'sh_entropy']].mean())  + [IK1F, IK2F,IK3F] 
            RQ1_table.append(row)
            

        if RQ == 2:
            
            # create the folder for storing the fairness testing result
            if not os.path.exists('../results/'):
                os.makedirs('../results/')
            if not os.path.exists('../results/' + dataset + '/'):
                os.makedirs('../results/' + dataset + '/')
            if not os.path.exists('../results/' + dataset + '/DICE/'):
                os.makedirs('../results/' + dataset + '/DICE/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ2/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ2/')
            if not os.path.exists('../results/' + dataset + '/DICE/RQ2/' + ''.join(str(i) for i in sens_params)+'_10runs/'):
                os.makedirs('../results/' + dataset + '/DICE/RQ2/' + ''.join(str(i) for i in sens_params)+'_10runs/')
            # storing the fairness testing result
            np.save('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_'+str(trial)+'.npy', 
                    np.array(global_inputs_list))
            np.save('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_'+str(trial)+'.npy', 
                    np.array(local_inputs_list))
            total_inputs = np.concatenate((local_inputs_list,global_inputs_list), axis=0)
            np.save('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_inputs_'+str(trial)+'.npy',
                    total_inputs)
            # RQ1 & RQ2

            local_sam = np.array(local_inputs_list).astype('int32')
            global_sam = np.array(global_inputs_list).astype('int32')
            # Storing result for RQ1 table\
            print('Analyzing the search results....')
            continue

            with open('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(global_inputs_list)):
                    m_sample = m_instance( np.array([global_inputs_list[ind][:input_shape[1]]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[global_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            with open('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv', 'w') as f:
                writer = csv.writer(f)
                for ind in range(len(local_inputs_list)):
                    m_sample = m_instance( np.array([local_inputs_list[ind][:input_shape[1]]]) , sens_params, data_config[dataset] ) 
                    rows = m_sample.reshape((len(m_sample),input_shape[1]))
                    writer.writerows(np.append(rows,[[local_inputs_list[ind][-1]] for i in range(len(m_sample))],axis=1))

            df_l = pd.read_csv('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/local_inputs_90_'+str(trial)+'.csv',header=None)  
            df_g = pd.read_csv('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/global_inputs_90_'+str(trial)+'.csv',header=None)

            df_g['label'] = model_argmax(sess, x, preds, df_g.to_numpy()[:,:input_shape[1]])
            df_l['label'] = model_argmax(sess, x, preds, df_l.to_numpy()[:,:input_shape[1]])
            g_pivot = pd.pivot_table(df_g, values="label", index=list(np.setxor1d(df_g.columns[:-1] ,
                                                                                  np.array(sens_params)-1)), aggfunc=np.sum)
            l_pivot = pd.pivot_table(df_l, values="label", index=list(np.setxor1d(df_l.columns[:-1] ,
                                                                               np.array(sens_params)-1)), aggfunc=np.sum)

            g_time = g_pivot.index[np.where((g_pivot['label'] > 0) & (g_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            l_time = l_pivot.index[np.where((l_pivot['label'] > 0) & (l_pivot['label'] < len(m_sample)))[0]].get_level_values(input_shape[1]).values
            tot_time = np.sort(np.concatenate((l_time, g_time), axis=0 ))
            
            if len(tot_time) >= 1000:
                time_1000 = tot_time[999]
            else:
                time_1000 = np.nan
                
            print('Time to 1st ID',tot_time[0])   
            print('time to 1000 ID',time_1000)
            g_dis_adf = len(g_time)
            l_dis_adf = len(l_time)


            global_succ_adf = round((g_dis_adf / len(global_sam)) * 100, 1)
            local_succ_adf  = round((l_dis_adf / len(local_sam)) * 100, 1)

            tot_df = pd.DataFrame(total_inputs)
            tot_df.columns=[i for i in range(input_shape[1])] + ['time']
            disc = []
            for sam_ind in range(total_inputs.shape[0]): 

                m_sample = m_instance( np.array([total_inputs[:,:input_shape[1]][sam_ind]]) , sens_params, data_config[dataset] )
                pred = pred_prob( sess, x, preds, m_sample , input_shape )
                clus_dic = clustering( pred, m_sample, sens_params, epsillon )
                if pred.max() > 0.5 and  pred.min()< 0.5:
                    disc.append(1)
                else:
                    disc.append(0)


            tot_df['disc'] = disc
            tot_dis = tot_df.loc[tot_df['disc'] == 1]
            tot_dis.to_csv('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/total_disc_' + str(trial)+'.csv')
            # reseting the TF graph for the next round
            sess.close()
            tf.reset_default_graph()
            print('Total ID RQ2  = ',g_dis_adf + l_dis_adf)
            print('Global ID RQ2  = ',g_dis_adf)
            print('Local  ID RQ2  = ',l_dis_adf)



            RQ2_table.append([g_dis_adf + l_dis_adf ,local_succ_adf, tot_time[0], time_1000 ])

            print('Local search success rate  = ', local_succ_adf, '%')
            print('Global search success rate = ', global_succ_adf, '%')

    
    if RQ==1 :
        np.save('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/QID_RQ1_10runs.npy',
                RQ1_table)

        with open('../results/' + dataset + '/DICE/RQ1/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/RQ1_table.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['#I','K_I', 'K_F', 'T_KF','Q_inf',
                             'Q_1', 'IK1F', 'IK2F', 'IK3F'])
                writer.writerow(np.mean(RQ1_table,axis=0))
                writer.writerow(np.std(RQ1_table,axis=0))
    
    elif RQ==2:
        
        np.save('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/QID_RQ2_10runs.npy',
                RQ2_table)

        with open('../results/' + dataset + '/DICE/RQ2/'+ ''.join(str(i) for i in sens_params)+'_10runs'+'/RQ2_table.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['tot_adf_disc','local_succ_adf',
                                 'time_to_first','time_to_1000'])
                writer.writerow(np.mean(RQ2_table,axis=0))
                writer.writerow(np.std(RQ2_table,axis=0))

                
def main(argv=None):
    dataset = FLAGS.dataset
    sensitive_index = FLAGS.sens_params
    logging.basicConfig(filename=f"../results/{dataset}/DICE/RQ1/{sensitive_index}_10runs/logfile.log",
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s'
                        )

    dnn_fair_testing(dataset = FLAGS.dataset, 
                     sens_params = FLAGS.sens_params,
                     model_path  = FLAGS.model_path,
                     cluster_num = FLAGS.cluster_num,
                     max_global  = FLAGS.max_global,
                     max_local   = FLAGS.max_local,
                     max_iter    = FLAGS.max_iter,
                     epsillon    = FLAGS.epsillon,
                     timeout    = FLAGS.timeout,
                     RQ = FLAGS.RQ
                    )
    

if __name__ == '__main__':    
    sens_list = [int(i) for i in re.findall('[0-9]+', args.sensitive_index)]
    flags.DEFINE_string("dataset",args.dataset, "the name of dataset")
    flags.DEFINE_string('model_path', '../models/', 'the path for testing model')
    flags.DEFINE_integer('cluster_num', 4, 'the number of clusters to form as well as the number of centroids to generate')
    flags.DEFINE_integer('max_global', 1000, 'maximum number of samples for global search')
    flags.DEFINE_integer('max_local', 1000, 'maximum number of samples for local search')
    flags.DEFINE_integer('max_iter', 10, 'maximum iteration of global perturbation')
    flags.DEFINE_integer('timeout', args.timeout, 'search timeout')
    flags.DEFINE_integer('RQ', args.RQ, 'RQ')
    
    #if result for RQ1 table: set this to [9,8,1], if result for RQ2 table: set this one sensitive attribute each time 
    # e.g. for census dataset, set [9], [8], [1] for sex, race and age respectively
    flags.DEFINE_list('sens_params', sens_list, 'sensitive parameters index.1 for age, 9 for gender, 8 for race')
    flags.DEFINE_float('epsillon', 0.025, 'the value of epsillon for partitioning')
    tf.app.run()


