import pandas as pd
import numpy as np 
import tensorflow as tf
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
import sys
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

dataframe = pd.read_csv('C:/thesis-goettert/numpyarray/numpyarray2troll.csv', names=['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2'])
dataframe = dataframe.drop(['nummer1','nummer2'], axis=1)

print(dataframe.shape)
print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()

    labels = dataframe.pop("action2")
    labels2 = dataframe.pop('subaction2')
    labels3 = dataframe.pop('action_length2')
    labels4 = dataframe.pop('origin_action2')
    labels5 = dataframe.pop('request2_1')
    labels6 = dataframe.pop('request2_2')
    labels7 = dataframe.pop('request2_3')
    labels8 = dataframe.pop('request2_4')
    labels9 = dataframe.pop('request2_5')
    labels10 = dataframe.pop('request2_6')
    labels11 = dataframe.pop('response2')

    labels = keras.utils.to_categorical(labels, num_classes=29)
    labels2 = keras.utils.to_categorical(labels2, num_classes=58)
    labels4 = keras.utils.to_categorical(labels4, num_classes=29)
    labels5 = keras.utils.to_categorical(labels5, num_classes=5)
    labels7 = keras.utils.to_categorical(labels7, num_classes=2)
    labels8 = keras.utils.to_categorical(labels8, num_classes=2)
    labels10 = keras.utils.to_categorical(labels10, num_classes=3)

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), (labels, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10, labels11)))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

val_dataframe.to_csv('C:/thesis-goettert/numpyarray/multiple_outputs_predictions_data.csv', sep='\t', encoding='utf-8', index= False)
train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target1:", y)
    #print("Target2:", z)


train_ds = train_ds.batch(100)
val_ds = val_ds.batch(100)

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
   

x = layers.Dense(128, activation="relu")(all_features)
x = layers.Dense(64, activation="relu")(x)

output1 = keras.layers.Dense(29, activation="softmax", name='action2')(x)
output2 = keras.layers.Dense(58, activation="softmax", name='subaction2')(x)
output3 = keras.layers.Dense(1, activation="linear", name='action_length2')(x)
output4 = keras.layers.Dense(29, activation="softmax", name='origin_action2')(x)
output5 = keras.layers.Dense(5, activation="softmax", name='request2_1')(x)
output6 = keras.layers.Dense(1, activation="linear", name='request2_2')(x)
output7 = keras.layers.Dense(2, activation="softmax", name='request2_3')(x)
output8 = keras.layers.Dense(2, activation="softmax", name='request2_4')(x)
output9 = keras.layers.Dense(1, activation="linear", name='request2_5')(x)
output10 = keras.layers.Dense(3, activation="softmax", name='request2_6')(x)
output11 = keras.layers.Dense(1, activation="linear", name='response2')(x)

encoder = keras.Model( inputs = all_inputs, outputs = [output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, output11], name="encoder")

# Let's plot 
keras.utils.plot_model(
    encoder
)

encoder.compile(
    loss={      'action2': tf.keras.losses.CategoricalCrossentropy(),
                'subaction2': tf.keras.losses.CategoricalCrossentropy(),
                'action_length2': tf.keras.losses.MeanSquaredError(),
                'origin_action2': tf.keras.losses.CategoricalCrossentropy(),
                'request2_1': tf.keras.losses.CategoricalCrossentropy(),
                'request2_2': tf.keras.losses.MeanSquaredError(),
                'request2_3': tf.keras.losses.CategoricalCrossentropy(),
                'request2_4': tf.keras.losses.CategoricalCrossentropy(),
                'request2_5': tf.keras.losses.MeanSquaredError(),
                'request2_6': tf.keras.losses.CategoricalCrossentropy(),
                'response2': tf.keras.losses.MeanSquaredError()},
    optimizer=  tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics={   'action2': 'accuracy',
                'subaction2': 'accuracy',
                'action_length2': 'mse',
                'origin_action2': 'accuracy',
                'request2_1': 'accuracy',
                'request2_2': 'mse',
                'request2_3': 'accuracy',
                'request2_4': 'accuracy',
                'request2_5': 'mse',
                'request2_6': 'accuracy',
                'response2': 'mse'},
                        )


# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(encoder, show_shapes=True, rankdir="LR", to_file='C:/thesis-goettert/numpyarray/plotmodel_multiple_outputs.png')


encoder.fit(train_ds, epochs=3, verbose=2, validation_data=val_ds) 

score = encoder.predict(val_ds,verbose=0)  
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
                print(data_slice)
                prediction_df[idx].append(y)
            else:
                prediction_df[idx].append(np.argmax(y))

 
prediction_df = pd.DataFrame(prediction_df)      
prediction_df.to_csv('C:/thesis-goettert/numpyarray/multiple_outputs_prediction_answer.csv',header= namelist, index = False, sep='\t', encoding='utf-8')