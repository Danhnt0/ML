import requests
import datetime


weather_map = {
    0:["fog","haze","smoke","dust","mist","foggy","hazy","smoky","dusty","misty"],
    0:["cloudy","overcast clouds","clouds","cloud","partly cloudy","mostly cloudy"],
    1:["clear sky","clear","sunny","sun","broken clouds","sky","scattered clouds","few clouds","mostly sunny","partly sunny","partly sun","mostly clear"],
    2:["rain","rainy","drizzle","drizzling","shower","showers","thunderstorm","thunderstorms","thunder","storm","stormy","light rain","light showers","light drizzle","light thunderstorm","light thunderstorms","light thunder","light storm","light stormy","heavy rain","heavy showers","heavy drizzle","heavy thunderstorm","heavy thunderstorms","heavy thunder","heavy storm","heavy stormy"],

}

def get_weather(weather):
    for key, value in weather_map.items():
        if weather in value:
            return key

url = 'http://dataservice.accuweather.com/currentconditions/v1/353412/historical?apikey=%099GfDMXluwADxMLMKrjy36x2ynAgNOvEv&details=true'

response = requests.get(url)
data = response.json()

# get data humidity,wind_speed,wind_dir,precip,pressure,app_temp,cloud,hour,month,temp
# get data from 5h before

input_data = []

data_now = []

for i in range(4, -1, -1):
    humidity = data[i]['RelativeHumidity']
    wind_speed = data[i]['Wind']['Speed']['Metric']['Value']
    wind_dir = data[i]['Wind']['Direction']['Degrees']
    precip = data[i]['Precip1hr']['Metric']['Value']
    pressure = data[i]['Pressure']['Metric']['Value']
    app_temp = data[i]['RealFeelTemperature']['Metric']['Value'] + 273.15
    cloud = data[i]['CloudCover']
    hour = datetime.datetime.strptime(data[i]['LocalObservationDateTime'], '%Y-%m-%dT%H:%M:%S%z').hour
    temp = data[i]['Temperature']['Metric']['Value'] + 273.15
    weather = data[i]['WeatherText']
    if i == 0:
        UV = data[i]['UVIndex']
        visibility = data[i]['Visibility']['Metric']['Value']
        data_now = [humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp,get_weather(weather.lower()),UV,visibility]

    input_data.append([humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp,get_weather(weather.lower())])


