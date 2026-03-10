import requests

url = "http://localhost:20000/api/v1/data"

params = {
    "chart": "system.cpu",
    "after": -60,
    "format": "json"
}

r = requests.get(url, params=params)

print(r.json())