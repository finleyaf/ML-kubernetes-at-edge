import argparse
import csv
import os
import time

import requests

DEFAULT_NODES = [
    "k3s-control",
    "k3s-worker-2",
    "k3s-worker-3",
    "k3s-worker-4",
    "raspberrypi",
]
REQUEST_TIMEOUT_S = 15
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY_S = 1
CSV_HEADER = [
    "timestamp",
    "node",
    "cpu_user",
    "cpu_system",
    "cpu_iowait",
    "ram_used",
    "net_received",
    "net_sent",
    "load1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:20000", help="Netdata base URL")
    parser.add_argument("--output", help="Output CSV path (defaults to dataset/dataset.csv)")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, help="Node hostnames to collect")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples to collect before exiting (0 means run continuously)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of overwriting it",
    )
    return parser.parse_args()


def resolve_output_path(custom_output):
    if custom_output:
        return os.path.abspath(custom_output)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "dataset", "dataset.csv")


def fetch_chart(base_url, node, chart, params):
    url = f"{base_url}/host/{node}/api/v1/data"
    last_error = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params={"chart": chart, **params},
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError):
                body = exc.response.text.strip() if exc.response is not None else ""
                last_error = RuntimeError(
                    f"HTTP {exc.response.status_code if exc.response is not None else 'error'} for chart {chart}: "
                    f"{body[:160] or '<empty response>'}"
                )
            elif isinstance(exc, ValueError):
                try:
                    body = response.text.strip()  # type: ignore[name-defined]
                except Exception:
                    body = ""
                last_error = RuntimeError(
                    f"non-JSON response for chart {chart}: {body[:160] or '<empty response>'}"
                )
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_RETRY_DELAY_S)
    raise RuntimeError(f"request failed after {REQUEST_RETRIES} attempts: {last_error}")


def write_header_if_needed(writer, dataset_path, append):
    if append and os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        return
    writer.writerow(CSV_HEADER)


def collect_sample(writer, base_url, nodes):
    timestamp = int(time.time())

    for node in nodes:
        try:
            cpu = fetch_chart(
                base_url,
                node,
                "system.cpu",
                {
                    "after": -1,
                    "points": 1,
                    "options": "percentage",
                },
            )

            ram = fetch_chart(
                base_url,
                node,
                "system.ram",
                {
                    "after": -1,
                    "points": 1,
                },
            )

            net = fetch_chart(
                base_url,
                node,
                "system.net",
                {
                    "after": -1,
                    "points": 1,
                },
            )

            load = fetch_chart(
                base_url,
                node,
                "system.load",
                {
                    "after": -1,
                    "points": 1,
                },
            )

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
                load_row[1],
            ])

            print(f"{node} collected")

        except Exception as e:
            print(f"{node} skipped: {e}")


def main():
    args = parse_args()
    dataset_path = resolve_output_path(args.output)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    mode = "a" if args.append else "w"
    with open(dataset_path, mode, newline="") as file:
        writer = csv.writer(file)
        write_header_if_needed(writer, dataset_path, args.append)

        samples_collected = 0
        while True:
            collect_sample(writer, args.base_url, args.nodes)
            file.flush()

            samples_collected += 1
            if args.samples > 0 and samples_collected >= args.samples:
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()