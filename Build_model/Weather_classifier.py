import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('data/weather_normal.csv')
df = df.dropna()


X = df.iloc[: , 0:10].values
y = df.iloc[: , 9].values

# y value constant [fog,rainy,cloudy,sunny]
n = 5
y = np.roll(y, -n)

X = X[:-n, :]
y = y[:-n]

for i in range(len(y)):
    if y[i] == 'Rainy':
        y[i] = 2
    elif y[i] == 'Cloudy':
        y[i] = 0
    elif y[i] == 'Sunny':
        y[i] = 1

# one hot encode y value

def one_hot_encode(y):
    one_hot = np.zeros((len(y), 3))
    for i in range(len(y)):
        one_hot[i, int(y[i])] = 1
    return one_hot


y = one_hot_encode(y)



X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# normalize data stardardization

x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)

X = (X - x_mean) / x_std


X = [X[i:i+5].ravel() for i in range(len(X)-4)]
# print(X[0])

# remove first 4 samples of label
y = y[4:, :]

X = np.array(X)
y = np.array(y)
# split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# build model

model = tf.keras.Sequential([
  tf.keras.layers.LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True, activation='tanh'),
  tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.LSTM(64, activation='tanh'),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_lstm, y_train, epochs=25, batch_size= 128, validation_split = 0.2)


# evaluate model
model.evaluate(X_test_lstm, y_test)

# save model

model.save('model/predict_weather5.h5')