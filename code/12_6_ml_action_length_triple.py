import pandas as pd
import numpy as np 
import tensorflow as tf
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
import sys
from matplotlib import pyplot as plt

dataframe = pd.read_csv('C:/thesis-goettert/numpyarray/numpyarray3.csv', names=['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','request1_4','request1_5','request1_6','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','request2_4','request2_5','request2_6','response2','nummer3','action_length3','action3','subaction3','origin_action3','request3_1','request3_2','request3_3','request3_4','request3_5','request3_6','response3'])
dataframe = dataframe.drop(['nummer1','nummer2','nummer3','subaction3','action3','origin_action3','request3_1','request3_2','request3_3','request3_4','request3_5','request3_6','response3'], axis=1)

print(dataframe.shape)
print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)
labels = val_dataframe['action_length3']

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("action_length3")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(128)
val_ds = val_ds.batch(128)

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

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
action2 = keras.Input(shape=(1,), name="action2", dtype="int64")
subaction2 = keras.Input(shape=(1,), name="subaction2", dtype="int64")
origin_action2 = keras.Input(shape=(1,), name="origin_action2", dtype="int64")
request2_1 = keras.Input(shape=(1,), name="request2_1", dtype="int64")
request2_3 = keras.Input(shape=(1,), name="request2_3", dtype="int64")


#numerical features
action_length1 = keras.Input(shape=(1,), name='action_length1')
request1_2 = keras.Input(shape=(1,), name='request1_2')
response1 = keras.Input(shape=(1,), name='response1')
action_length2 = keras.Input(shape=(1,), name='action_length2')
response2 = keras.Input(shape=(1,), name='response2')
request2_2 = keras.Input(shape=(1,), name='request2_2')

all_inputs = [action_length1,action1,subaction1,origin_action1,request1_1,request1_2,request1_3,response1,action_length2,action2,subaction2,origin_action2,request2_1,request2_2,request2_3,response2]

# Integer categorical features
action1_encoded = encode_categorical_feature(action1, "action1", train_ds, False)
subaction1_encoded = encode_categorical_feature(subaction1, "subaction1", train_ds, False)
origin_action1_encoded = encode_categorical_feature(origin_action1, "origin_action1", train_ds, False)
request1_1_encoded = encode_categorical_feature(request1_1, "request1_1", train_ds, False)
request1_3_encoded = encode_categorical_feature(request1_3, "request1_3", train_ds, False)
action2_encoded = encode_categorical_feature(action2, "action2", train_ds, False)
subaction2_encoded = encode_categorical_feature(subaction2, "subaction2", train_ds, False)
origin_action2_encoded = encode_categorical_feature(origin_action2, "origin_action2", train_ds, False)
request2_1_encoded = encode_categorical_feature(request2_1, "request2_1", train_ds, False)
request2_3_encoded = encode_categorical_feature(request2_3, "request2_3", train_ds, False)


#Numerical features
action_length1_encoded = encode_numerical_feature(action_length1, "action_length1", train_ds)
response1_encoded = encode_numerical_feature(response1, "response1", train_ds)
request1_2_encoded = encode_numerical_feature(request1_2, "request1_2", train_ds)
action_length2_encoded = encode_numerical_feature(action_length2, "action_length2", train_ds)
response2_encoded = encode_numerical_feature(response2, "response2", train_ds)
request2_2_encoded = encode_numerical_feature(request2_2, "request2_2", train_ds)

all_features = layers.concatenate(
    [
        action_length1_encoded,
        action1_encoded,
        subaction1_encoded,
        origin_action1_encoded,
        request1_1_encoded,
        request1_2_encoded,
        request1_3_encoded,
        response1_encoded,
        action_length2_encoded,
        action2_encoded,
        subaction2_encoded,
        origin_action2_encoded,
        request2_1_encoded,
        request2_2_encoded,
        request2_3_encoded,
        response2_encoded
        ])

x = layers.Dense(128, activation='relu')(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = keras.layers.Dense(1, activation='linear')(all_features)

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
pred_list = prediction_df


a = plt.axes(aspect='equal')
plt.scatter(label_list, pred_list)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 200]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()