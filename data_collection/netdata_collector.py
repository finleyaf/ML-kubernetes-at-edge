import requests
import time
import csv
import os

BASE_URL = "http://localhost:20000"

nodes = [
    "k3s-control",
    "k3s-worker-2",
    "k3s-worker-3"
]

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "dataset", "dataset.csv")
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

file = open(dataset_path, "w", newline="")
writer = csv.writer(file)

writer.writerow([
    "timestamp",
    "node",
    "cpu_user",
    "cpu_system",
    "cpu_iowait",
    "ram_used",
    "net_received",
    "net_sent",
    "load1"
])

while True:

    timestamp = int(time.time())

    for node in nodes:

        try:
            cpu = requests.get(
                f"{BASE_URL}/host/{node}/api/v1/data",
                params={
                    "chart":"system.cpu",
                    "after":-1,
                    "points":1,
                    "options":"percentage"
                },
                timeout=5
            ).json()

            ram = requests.get(
                f"{BASE_URL}/host/{node}/api/v1/data",
                params={
                    "chart":"system.ram",
                    "after":-1,
                    "points":1
                },
                timeout=5
            ).json()

            net = requests.get(
                f"{BASE_URL}/host/{node}/api/v1/data",
                params={
                    "chart":"system.net",
                    "after":-1,
                    "points":1
                },
                timeout=5
            ).json()

            load = requests.get(
                f"{BASE_URL}/host/{node}/api/v1/data",
                params={
                    "chart":"system.load",
                    "after":-1,
                    "points":1
                },
                timeout=5
            ).json()

            cpu_row = cpu["data"][0]
            ram_row = ram["data"][0]
            net_row = net["data"][0]
            load_row = load["data"][0]

            writer.writerow([
                timestamp,
                node,
                cpu_row[6],
                cpu_row[7],
                cpu_row[9],
                ram_row[1],
                abs(net_row[1]),
                abs(net_row[2]),
                load_row[1]
            ])

            print(f"{node} collected")

        except Exception as e:
            print(f"{node} skipped: {e}")

    file.flush()

    time.sleep(1)