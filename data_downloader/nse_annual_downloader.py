import os
import time
import zipfile
import requests

BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com",
    "Connection": "keep-alive"
}

def create_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(BASE, timeout=10)   # mandatory cookie
    time.sleep(1)
    return s

def download_annual_reports(symbol):
    outdir = f"data/nse/{symbol}"
    os.makedirs(outdir, exist_ok=True)

    session = create_session()
    api = f"{BASE}/api/annual-reports?index=cm&symbol={symbol}"
    resp = session.get(api, timeout=15)

    payload = resp.json()
    rows = payload.get("data", [])

    if not rows:
        print(f"[NO DATA] {symbol}")
        return

    for row in rows:
        url = row.get("fileName")
        if not url:
            continue

        fname = os.path.basename(url)
        path = os.path.join(outdir, fname)

        if os.path.exists(path):
            continue

        print(f"Downloading: {fname}")
        r = session.get(url, timeout=30)

        with open(path, "wb") as f:
            f.write(r.content)

        # Auto-extract ZIPs
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as z:
                    z.extractall(outdir)
                os.remove(path)
            except Exception as e:
                print(f"ZIP extract failed: {fname}")

        time.sleep(1)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols_file",
        required=True,
        help="Path to sector symbol file (e.g. symbols/finance.txt)"
    )
    args = parser.parse_args()

    with open(args.symbols_file) as f:
        symbols = [line.strip().upper() for line in f if line.strip()]

    for sym in symbols:
        print(f"\n=== Processing {sym} ===")
        download_annual_reports(sym)

