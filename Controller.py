import tensorflow as tf
import numpy as np
import warnings
import GetCurrent

warnings.filterwarnings("ignore")


x_mean = np.array([7.74973094e+01, 2.80202640e+00, 1.42257474e+02, 2.46415703e-01,
       1.00962442e+03, 3.00062242e+02, 6.81242020e+01, 1.14995508e+01,
       2.97592885e+02,1.24065860e+00])
x_std = np.array([ 15.88368275,   1.40884157, 101.64481521,   0.99006853,
         7.03036941,   8.3542265 ,  27.69297933,   6.92203511,
         5.75967504, 0.75557027])


input = np.array(GetCurrent.input_data)
# print(input)

# # reverse input
# input = input[::-1]
# print(input.ravel() )


for i in range(9):
    # check if input nan 
    if np.isnan(input[0][i]):
        input[0][i] = x_mean[i]

def predict_temp():
    model = tf.keras.models.load_model('Model/model_temp.h5')
   
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = y_pred*5.759 + 24.44
    y_pred = np.round(y_pred,1, out=None)
    print(y_pred)
    return y_pred[0]



def get_icon_url(weather):
    if weather == 2:
        return 'url(:/Img/Image/icons8-rain-48.png);'
    elif weather == 0:
        return 'url(:/Img/Image/icons8-partly-cloudy-day-48.png);'
    elif weather == 1:
        return 'url(:/Img/Image/icons8-sun-48.png);'
    
def predict_weather_1h():
    model = tf.keras.models.load_model('Model/predict_weather1.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)



def predict_weather_2h():
    model = tf.keras.models.load_model('Model/predict_weather2.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_3h():
    model = tf.keras.models.load_model('Model/predict_weather3.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_4h():
    model = tf.keras.models.load_model('Model/predict_weather4.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_5h():
    model = tf.keras.models.load_model('Model/predict_weather5.h5')
    X = (input- x_mean)/x_std
    X = X.ravel()
    X = np.reshape(X, (1,50,1))
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)



    
