import requests
import json

url = "http://localhost:20000/api/v1/charts"

response = requests.get(url)
charts = response.json()["charts"]

for chart in charts:
    print(chart)