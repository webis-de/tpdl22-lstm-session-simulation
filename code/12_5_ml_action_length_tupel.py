import pandas as pd
import numpy as np 
import tensorflow as tf
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
import numpy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense
import sys
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import keras.backend as K
from keras import regularizers


dataframe = pd.read_csv('C:/thesis-goettert/numpyarray/numpyarray2.csv', names=['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'])
dataframe = dataframe.drop(['nummer1','nummer2','action2','response2','subaction2','origin_action2','request2_1','request2_3','request2_2','request2_4','request2_5','request2_6'], axis=1)

print(dataframe.shape)
print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

labels = val_dataframe['action_length2']

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    cs = MinMaxScaler()
    dataframe = dataframe.copy()
    labels = dataframe.pop("action_length2")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
##print(train_ds)
#sys.exit()

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(128)
val_ds = val_ds.batch(128)

def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)
    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature    

# Categorical features encoded as integers
action1 = keras.Input(shape=(1,), name="action1", dtype="int64")
subaction1 = keras.Input(shape=(1,), name="subaction1", dtype="int64")
origin_action1 = keras.Input(shape=(1,), name="origin_action1", dtype="int64")
request1_1 = keras.Input(shape=(1,), name="request1_1", dtype="int64")
request1_3 = keras.Input(shape=(1,), name="request1_3", dtype="int64")
request1_4 = keras.Input(shape=(1,), name="request1_4", dtype="int64")
request1_6 = keras.Input(shape=(1,), name="request1_6", dtype="int64")



#numerical features
action_length1 = keras.Input(shape=(1,), name='action_length1')
response1 = keras.Input(shape=(1,), name='response1')
request1_2 = keras.Input(shape=(1,), name='request1_2')
request1_5 = keras.Input(shape=(1,), name='request1_5')


all_inputs = [action_length1,action1,subaction1,origin_action1,request1_1,request1_2,request1_3,request1_4,request1_5,request1_6,response1]

# Integer categorical features
action1_encoded = encode_categorical_feature(action1, "action1", train_ds, False)
subaction1_encoded = encode_categorical_feature(subaction1, "subaction1", train_ds, False)
origin_action1_encoded = encode_categorical_feature(origin_action1, "origin_action1", train_ds, False)
request1_1_encoded = encode_categorical_feature(request1_1, "request1_1", train_ds, False)
request1_3_encoded = encode_categorical_feature(request1_3, "request1_3", train_ds, False)
request1_4_encoded = encode_categorical_feature(request1_4, "request1_4", train_ds, False)
request1_6_encoded = encode_categorical_feature(request1_6, "request1_6", train_ds, False)



#Numerical features
action_length1_encoded = encode_numerical_feature(action_length1, "action_length1", train_ds)
response1_encoded = encode_numerical_feature(response1, "response1", train_ds)
request1_2_encoded = encode_numerical_feature(request1_2, "request1_2", train_ds)
request1_5_encoded = encode_numerical_feature(request1_5, "request1_5", train_ds)


all_features = layers.concatenate(
    [
        action_length1_encoded,
        action1_encoded,
        subaction1_encoded,
        origin_action1_encoded,
        request1_1_encoded,
        request1_2_encoded,
        request1_3_encoded,
        request1_4_encoded,
        request1_5_encoded,
        request1_6_encoded,
        response1_encoded,
        ])



x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = keras.layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(all_features)

model = keras.Model(all_inputs, output)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=  tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics='mse'
                        )

model.fit(train_ds, epochs=10, validation_data=val_ds)

score = model.predict(val_ds,verbose=2)
    
prediction_df = [] 
is_first_slice = True   
for idx, y in enumerate(score):
    prediction_df.append(y)

label_list = labels.tolist()
label_list = label_list[:200]
pred_list = prediction_df
pred_list = pred_list[:200]


a = plt.axes(aspect='equal')
plt.scatter(label_list, pred_list)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 200]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()

#confusion_matrix = tf.math.confusion_matrix(label_list,prediction_df)
#confusion_matrix2 = confusion_matrix.numpy()

#df_cm = pd.DataFrame(confusion_matrix2, index = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27']],
 #                 columns = [i for i in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27']])
#plt.figure(figsize = (24,8))
#sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
#plt.show()     

#prediction_df = pd.DataFrame(prediction_df)      
#prediction_df.to_csv('C:/thesis-goettert/classification/predictions.csv',header= 'prediction', index = False, sep='\t', encoding='utf-8')