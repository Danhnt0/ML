import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load data
df = pd.read_csv('data/weather_normal.csv')

X = df.iloc[:, 0:10].values
y = df.iloc[:, 8].values

print(X.shape)
print(y.shape)


# shift y prediction by 1-5 hours and stack into a numpy array
label = []
n = -5
for i in range(1,6):

  y_shifted = np.roll(y, -i)
  label.append(y_shifted)
label = np.array(label).T


X = X[:n, :]
label = label[:n, :]

x_std = np.std(X,axis =  0)
x_mean = np.mean(X,axis = 0)

y_mean = np.mean(label)
y_std = np.std(label)

print(x_std)
print(x_mean)

# print(X[:10,:])
# print(label[:10,:])

X = (X - x_mean)/x_std

label = (label - y_mean) / y_std

# get 5 hours of data for each training sample and flatten
X = [X[i:i+5].ravel() for i in range(len(X)-4)]
# print(X[0])

# remove first 4 samples of label
label = label[4:, :]

# print(label[0])

X = np.array(X)
label = np.array(label)



# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2,random_state=0)


# reshape input to fit LSTM model
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# created generator to get 5 hours of data for each training sample
# def generator(X,y, step):
#   while True:
#     for i in range(0, len(X), step):
#       yield X[i:i+step].ravel(), y[i:i+step]



# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True, activation='tanh'),
  tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.LSTM(128, activation='tanh'),
  tf.keras.layers.Dropout(0.2),


  tf.keras.layers.Dense(256, activation="linear"),
  tf.keras.layers.Dense(128, activation="linear"),
  tf.keras.layers.Dense(128, activation="linear"),
  tf.keras.layers.Dense(5, activation="linear")
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train model

history = model.fit(X_train_lstm, y_train, epochs=70, batch_size= 128, validation_split=0.2)

# save model
model.save('Model/model_temp.h5')

# predict on test data
y_pred = model.predict(X_test_lstm)

# print some predictions
for i in range(10):
  print('Actual: ', y_test[i])
  print('Predicted: ', y_pred[i])
  print('------------------')



loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()