"""Small load test of the FastAPI backend: N requests at a given concurrency, latency percentiles and errors.

    uvicorn api:app --port 8000 &   then   python3 bench_api.py --url http://127.0.0.1:8000 --n 60 --c 6
"""
import argparse, concurrent.futures as cf, json, statistics, time, urllib.request

MSGS = ["someone stole my phone on the bus", "the police won't register my FIR", "my husband beats me",
        "what is the punishment for murder", "IPC 420", "my landlord won't return my deposit",
        "a guy keeps following me home", "someone hacked my upi and took money", "my brother was arrested last night",
        "my neighbour's dog keeps barking all night"]


def one(url, msg):
    body = json.dumps({"message": msg, "turns": [], "last_kind": None}).encode()
    req = urllib.request.Request(url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            ok = r.status == 200 and "reply" in json.loads(r.read())
    except Exception:
        ok = False
    return (time.perf_counter() - t) * 1000, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--c", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(a.c) as ex:
        res = list(ex.map(lambda i: one(a.url, MSGS[i % len(MSGS)]), range(a.n)))
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    print(f"{a.n} requests, concurrency {a.c}: errors {sum(not r[1] for r in res)}, median {statistics.median(lat):.0f} ms, "
          f"p95 {lat[int(.95 * len(lat)) - 1]:.0f} ms, max {lat[-1]:.0f} ms, throughput {a.n / wall:.1f} req/s")


if __name__ == "__main__":
    main()
