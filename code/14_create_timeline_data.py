import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys

step1 = pd.read_csv('C:/thesis-goettert/numpyarray/first_step.csv', names=['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'])
step1 = step1.drop(['nummer1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'], axis=1)
step1['session'] = step1.index
step1 = step1[['session','action1','action_length1','request1_4','request1_5','response1']]

step2 = pd.read_csv('C:/thesis-goettert/numpyarray/2nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step2 = step2.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step2['session'] = step2.index
step2 = step2[['session','action1','action_length1','request1_4','request1_5','response1']]

step3 = pd.read_csv('C:/thesis-goettert/numpyarray/3nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step3 = step3.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step3['session'] = step3.index
step3 = step3[['session','action1','action_length1','request1_4','request1_5','response1']]

step4 = pd.read_csv('C:/thesis-goettert/numpyarray/4nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step4 = step4.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step4['session'] = step4.index
step4 = step4[['session','action1','action_length1','request1_4','request1_5','response1']]

step5 = pd.read_csv('C:/thesis-goettert/numpyarray/5nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step5 = step5.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step5['session'] = step5.index
step5 = step5[['session','action1','action_length1','request1_4','request1_5','response1']]

step6 = pd.read_csv('C:/thesis-goettert/numpyarray/6nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step6 = step6.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step6['session'] = step6.index
step6 = step6[['session','action1','action_length1','request1_4','request1_5','response1']]

step7 = pd.read_csv('C:/thesis-goettert/numpyarray/7nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step7 = step7.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step7['session'] = step7.index
step7 = step7[['session','action1','action_length1','request1_4','request1_5','response1']]

step8 = pd.read_csv('C:/thesis-goettert/numpyarray/8nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step8 = step8.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step8['session'] = step8.index
step8 = step8[['session','action1','action_length1','request1_4','request1_5','response1']]

step9 = pd.read_csv('C:/thesis-goettert/numpyarray/9nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step9 = step9.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step9['session'] = step9.index
step9 = step9[['session','action1','action_length1','request1_4','request1_5','response1']]

step10 = pd.read_csv('C:/thesis-goettert/numpyarray/10nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step10 = step10.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step10['session'] = step10.index
step10 = step10[['session','action1','action_length1','request1_4','request1_5','response1']]

step11 = pd.read_csv('C:/thesis-goettert/numpyarray/11nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step11 = step11.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step11['session'] = step11.index
step11 = step11[['session','action1','action_length1','request1_4','request1_5','response1']]

step12 = pd.read_csv('C:/thesis-goettert/numpyarray/12nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step12 = step12.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step12['session'] = step12.index
step12 = step12[['session','action1','action_length1','request1_4','request1_5','response1']]

step13 = pd.read_csv('C:/thesis-goettert/numpyarray/13nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step13 = step13.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step13['session'] = step13.index
step13 = step13[['session','action1','action_length1','request1_4','request1_5','response1']]

step14 = pd.read_csv('C:/thesis-goettert/numpyarray/14nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step14 = step14.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step14['session'] = step14.index
step14 = step14[['session','action1','action_length1','request1_4','request1_5','response1']]

step15 = pd.read_csv('C:/thesis-goettert/numpyarray/15nd_step_fixed.csv', sep='\t', names=['action1', 'subaction1', 'action_length1', 'origin_action1', 'request1_1', 'request1_2', 'request1_3', 'request1_4', 'request1_5', 'request1_6', 'response1'])
step15 = step15.drop(['subaction1','origin_action1','request1_1','request1_2','request1_3','request1_6'], axis=1)
step15['session'] = step15.index
step15 = step15[['session','action1','action_length1','request1_4','request1_5','response1']]

concat_df = pd.DataFrame(columns=['session','action1','action_length1','request1_4','request1_5','response1'])

#b = len(step1.index)
b = 200
print(b)

i = 0 
while i < b:
    concat_df = concat_df.append(step1.iloc[[i]])
    concat_df = concat_df.append(step2.iloc[[i]])
    concat_df = concat_df.append(step3.iloc[[i]])
    concat_df = concat_df.append(step4.iloc[[i]])
    concat_df = concat_df.append(step5.iloc[[i]])
    concat_df = concat_df.append(step6.iloc[[i]])
    concat_df = concat_df.append(step7.iloc[[i]])
    concat_df = concat_df.append(step8.iloc[[i]])
    concat_df = concat_df.append(step9.iloc[[i]])
    concat_df = concat_df.append(step10.iloc[[i]])
    concat_df = concat_df.append(step11.iloc[[i]])
    concat_df = concat_df.append(step12.iloc[[i]])
    concat_df = concat_df.append(step13.iloc[[i]])
    concat_df = concat_df.append(step14.iloc[[i]])
    concat_df = concat_df.append(step15.iloc[[i]])
    i += 1

concat_df.to_csv('C:/thesis-goettert/numpyarray/session_length_15.csv', index = False, header=False, encoding='utf-8')   