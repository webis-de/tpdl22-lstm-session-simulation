import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

step = pd.read_csv('C:/thesis-goettert/numpyarray/15nd_step.csv', header=0, sep='\t', encoding='utf-8')

step['action_length2'] = step['action_length2'].str.replace('[', '')
step['response'] = step['response'].str.replace('[', '')
step['request2_2'] = step['request2_2'].str.replace('[', '')
step['request2_5'] = step['request2_5'].str.replace('[', '')

step['action_length2'] = step['action_length2'].str.replace(']', '')
step['response'] = step['response'].str.replace(']', '')
step['request2_2'] = step['request2_2'].str.replace(']', '')
step['request2_5'] = step['request2_5'].str.replace(']', '')

step = step.apply(pd.to_numeric, errors='coerce')


#step['action_length2'] = step['action_length2'].apply(np.floor)
#step['response'] = step['response'].apply(np.floor)
#step['request2_2'] = step['request2_2'].apply(np.floor)
#step['request2_5'] = step['request2_5'].apply(np.floor)

step = step.round()
step = step.clip(lower= 0)

#namelist = ['action2', 'subaction2', 'action_length2', 'origin_action2', 'request2_1', 'request2_2', 'request2_3', 'request2_4', 'request2_5', 'request2_6', 'response2']
step = pd.DataFrame(step)      
step.to_csv('C:/thesis-goettert/numpyarray/15nd_step_fixed.csv', index = False, header=False, sep='\t', encoding='utf-8')