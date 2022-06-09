import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys

#pred_dataframe = pd.read_csv('C:/thesis-goettert/numpyarray/numpyarray2.csv', names=['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'])
#pred_dataframe = pred_dataframe.drop(['nummer1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'], axis=1)

pred_dataframe2 = pd.read_csv('C:/thesis-goettert/numpyarray/14nd_step_fixed.csv', sep='\t', names=['action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1'])


print(pred_dataframe2)

loaded_model = keras.models.load_model('C:/thesis-goettert/saved_model')

def pred_dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    return ds

pred_ds = pred_dataframe_to_dataset(pred_dataframe2)
pred_ds = pred_ds.batch(128)    

score = loaded_model.predict(pred_ds,verbose=2)
namelist = ['action2', 'subaction2', 'action_length2', 'origin_action2', 'request2_1', 'request2_2', 'request2_3', 'request2_4', 'request2_5', 'request2_6', 'response']
prediction_df = []
is_first_slice = True
for data_slice in score:   
    if(is_first_slice):
        for y in data_slice:
            prediction_df.append([np.argmax(y)])
        is_first_slice = False        
    else:        
        for idx, y in enumerate(data_slice):
            if(y.size == 1):
                prediction_df[idx].append(y)
            else:
                prediction_df[idx].append(np.argmax(y))

 
prediction_df = pd.DataFrame(prediction_df)      
prediction_df.to_csv('C:/thesis-goettert/numpyarray/15nd_step.csv',header= namelist, index = False, sep='\t', encoding='utf-8')