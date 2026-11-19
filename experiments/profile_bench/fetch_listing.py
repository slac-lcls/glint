"""Download every .h5 a CXIDB directory listing links to, for a deposit with no tar or checksum list.

The listing page links each file on portal.nersc.gov and gives its size to 0.1 GiB. The portal sends
no Content-Length, so a file is renamed from .part only when curl finished cleanly, its size agrees
with the listing, and h5py can open it and find its frames. Complete files are skipped, so re-running
after a failure fetches only what is missing.

    python fetch_listing.py https://cxidb.org/data/45/ProK-RT/ OUTDIR [-j 4] [--match 'run[0-9]+-1\.h5$']
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

import h5py

UNIT = {"KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30, "TiB": 2 ** 40}


def fetch(url, listed, outdir, tries=6):
    out = os.path.join(outdir, url.rsplit("/", 1)[1])
    if os.path.exists(out):
        return f"have {out}"
    part = out + ".part"
    t0 = time.time()
    for attempt in range(tries):
        rc = subprocess.run(["curl", "-fsSL", "--connect-timeout", "60", "--speed-limit", "100000",
                             "--speed-time", "300", "-o", part, url]).returncode
        size = os.path.getsize(part) if os.path.exists(part) else 0
        if rc == 0 and abs(size - listed) <= 0.06 * UNIT["GiB"]:
            with h5py.File(part, "r") as f:
                n = sum(k.startswith("tag-") for k in f)
            if not n:
                raise RuntimeError(f"{url}: no frames")
            os.replace(part, out)
            dt = time.time() - t0
            return f"got {out} {size / 1e9:.2f} GB, {n} frames, in {dt:.0f} s ({size / 1e6 / max(dt, 1):.0f} MB/s)"
        print(f"retry {url}: curl {rc}, {size} bytes, listed {listed}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{url}: incomplete after {tries} tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("outdir")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--match", default="", help="only files whose name matches this regex")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    page = subprocess.run(["curl", "-fsSL", "-m", "120", a.listing], capture_output=True, text=True, check=True).stdout
    files = {u.replace("http://", "https://"): float(v) * UNIT[w]
             for u, v, w in re.findall(r'href="([^"]+\.h5)"[^>]*>[^<]*\(([\d.]+)([KMGT]iB)\)', page)}
    files = {u: s for u, s in files.items() if re.search(a.match, u.rsplit("/", 1)[1])}
    print(f"{len(files)} files, {sum(files.values()) / 2 ** 30:.0f} GiB", flush=True)
    if not files:
        raise SystemExit("no files in the listing")
    failed = 0
    with cf.ThreadPoolExecutor(a.j) as pool:
        for fut in cf.as_completed([pool.submit(fetch, u, s, a.outdir) for u, s in files.items()]):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print(f"FAILED {e}", flush=True)
    print(f"done, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
