import pandas as pd

df = pd.read_csv('data/weather_normal.csv')

# check sum of each unique value

print(df['weather'].value_counts())

