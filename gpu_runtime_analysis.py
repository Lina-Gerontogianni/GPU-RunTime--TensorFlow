## Importing the basic libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

## Making NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

## Importing libraries for the ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# for reproducibility purpose 
from numpy.random import seed
seed(42)
tf.random.set_seed(42)
sk.utils.check_random_state(42)


## Importing the dataset
os.chdir('.../kaggle_datasets/GPU-runtime_dataset')
dataset = pd.read_csv('sgemm_product.csv')
dataset['GPU_avg_run'] = dataset[dataset.columns[-4:]].mean(axis=1)
dataset = dataset.drop(dataset.iloc[:,-5:-1], axis=1)

## Separating features X from target variable y (GPU runtime)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Building the ANN
dnn = Sequential(
    [layers.Dense(65,kernel_initializer='normal', activation = 'relu'),
     layers.Dense(65,kernel_initializer='normal', activation = 'relu'),
     layers.Dense(50,kernel_initializer='normal', activation = 'relu'),
     layers.Dense(40,kernel_initializer='normal', activation = 'relu'),
     layers.Dense(30,kernel_initializer='normal', activation = 'relu'),
     layers.Dense(1)
     ])

## Compiling the ANN
dnn.compile(optimizer = keras.optimizers.Adam(10e-4), loss='mean_squared_error')


## Training the ANN
dnn_training = dnn.fit(X_train, y_train, batch_size = 100, epochs =200,
                       validation_split = 0.2)

## Plotting the loss functions
def plot_loss(history):
  plt.ylim(top=1000)
  plt.plot(history.history['loss'], label='training_loss')
  plt.plot(history.history['val_loss'], label='validation_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Training error')
  plt.legend()
  plt.grid(True)
  
plot_loss(dnn_training)

## Predicting the GPU runtime for the test data
y_pred = dnn.predict(X_test)

## Calculating the Mean Squared Error of the predicted test values
mean_squared_error(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1))

## Plotting predicted vs true GPU values
plt.figure(figsize=(10, 10))
plt.xlabel('true') 
plt.ylabel('predicted')
plt.title('Test data: GPU running time')
plt.scatter(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1), s=0.3)
plt.plot(y_test.reshape(len(y_test),1),y_test.reshape(len(y_test),1),
         color='red', linewidth=0.5)
plt.show()
