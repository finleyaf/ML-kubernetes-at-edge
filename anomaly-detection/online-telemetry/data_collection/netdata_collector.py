import argparse
import csv
import os
import time

import requests

DEFAULT_NODES = ["k3s-control", "k3s-worker-2", "k3s-worker-3"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:20000", help="Netdata base URL")
    parser.add_argument("--output", help="Output CSV path (defaults to dataset/dataset.csv)")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, help="Node hostnames to collect")
    return parser.parse_args()


def resolve_output_path(custom_output):
    if custom_output:
        return os.path.abspath(custom_output)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "dataset", "dataset.csv")


def main():
    args = parse_args()
    dataset_path = resolve_output_path(args.output)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with open(dataset_path, "w", newline="") as file:
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

            for node in args.nodes:
                try:
                    cpu = requests.get(
                        f"{args.base_url}/host/{node}/api/v1/data",
                        params={
                            "chart": "system.cpu",
                            "after": -1,
                            "points": 1,
                            "options": "percentage"
                        },
                        timeout=5
                    ).json()

                    ram = requests.get(
                        f"{args.base_url}/host/{node}/api/v1/data",
                        params={
                            "chart": "system.ram",
                            "after": -1,
                            "points": 1
                        },
                        timeout=5
                    ).json()

                    net = requests.get(
                        f"{args.base_url}/host/{node}/api/v1/data",
                        params={
                            "chart": "system.net",
                            "after": -1,
                            "points": 1
                        },
                        timeout=5
                    ).json()

                    load = requests.get(
                        f"{args.base_url}/host/{node}/api/v1/data",
                        params={
                            "chart": "system.load",
                            "after": -1,
                            "points": 1
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
            time.sleep(args.interval)


if __name__ == "__main__":
    main()