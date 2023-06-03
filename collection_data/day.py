import csv
import requests

# api_key = "7c0a239224e7429daddafb9b9de536c5"
api_key = "5e27f851fd764810be5b08f89ab34e3e"
# api_key ="b57f5d5ab486430db1d389157fd6ee1e"

city_name = "Hanoi"

start_date = "2004-05-01"
step = "604800"# 7 days
end_date = "2005-05-01"

url = f"https://api.weatherbit.io/v2.0/history/daily?city=Hanoi&country=VN&start_date={start_date}&end_date={end_date}&key={api_key}"

response = requests.get(url)
print(response)

data = response.json()
with open('data/weather_date.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    #writer.writerow([ 'Datetime','Date', 'Temperature', 'Humidity', 'Temp_min', 'Temp_max', 'Pressure', 'Wind_speed', 'Clouds','Precip'])
    for item in data['data']:
        # Extract the date, temperature, and humidity from the JSON data
        date = item['ts']
        wind_speed = item['wind_spd']
        humidity = item['rh']
        temp_min = item['min_temp']
        temp_max = item['max_temp']
        pressure = item['pres']
        temp = item['temp']
        clouds = item['clouds']
        datetime = item['datetime']
        precip = item['precip']
        writer.writerow([datetime,date, temp, humidity, temp_min, temp_max, pressure, wind_speed, clouds,precip])


