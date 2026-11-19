"""Stream-extract an uncompressed tar over HTTP, resuming at the last whole member on a dropped
connection. CXIDB deposits its large sets as single plain tars (cxidb_33_DatasetA.h5.tar is 243 GB);
this extracts members as they arrive, so no second copy of the archive is ever on disk.

A member's header starts on a 512-byte boundary, and a byte range that starts at a header is itself
a valid tar stream. So the resume point is the header after the last member that was written out
whole. Members land as <name>.part and are renamed only when complete, so anything without the
suffix is whole, and members already on disk with the right size are skipped.

    python fetch_tar.py URL OUTDIR [--max-members N] [--state OUTDIR/.offset]
"""
import argparse
import os
import sys
import tarfile
import time
import subprocess


class stream_from:
    """curl's stdout from byte ``offset`` on. curl, as CXIDB recommends for its big files."""

    def __init__(self, url, offset):
        self.p = subprocess.Popen(["curl", "-sfL", "--speed-time", "120", "--speed-limit", "1000",
                                   "-r", f"{offset}-", url], stdout=subprocess.PIPE)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.p.kill()
        self.p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("outdir")
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument("--tries", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    state = os.path.join(a.outdir, ".offset")
    offset = int(open(state).read()) if os.path.exists(state) else 0
    done = 0
    for attempt in range(a.tries):
        try:
            with stream_from(a.url, offset) as s:
                tf = tarfile.open(fileobj=s.p.stdout, mode="r|")
                for m in tf:
                    nxt = offset + m.offset_data + ((m.size + 511) // 512) * 512
                    if not m.isfile():
                        continue
                    name = os.path.basename(m.name)
                    dest = os.path.join(a.outdir, name)
                    if os.path.exists(dest) and os.path.getsize(dest) == m.size:
                        print(f"skip {name}", flush=True)
                    else:
                        src = tf.extractfile(m)
                        t0 = time.time()
                        with open(dest + ".part", "wb") as f:
                            while True:
                                b = src.read(1 << 24)
                                if not b:
                                    break
                                f.write(b)
                        if os.path.getsize(dest + ".part") != m.size:
                            raise IOError(f"{name}: short read")
                        os.replace(dest + ".part", dest)
                        dt = time.time() - t0
                        print(f"got {name} {m.size / 1e9:.2f} GB in {dt:.0f} s "
                              f"({m.size / 1e6 / max(dt, 1e-9):.0f} MB/s)", flush=True)
                    # the next member's header, relative to the original archive
                    with open(state, "w") as f:
                        f.write(str(nxt))
                    done += 1
                    if a.max_members and done >= a.max_members:
                        print("max-members reached", flush=True)
                        return 0
                # a curl that died on a member boundary looks like a clean end of archive to tarfile
                if s.p.wait() != 0:
                    raise IOError(f"curl exited {s.p.returncode}")
            print("archive complete", flush=True)
            return 0
        except Exception as e:                       # dropped connection, timeout, short read
            offset = int(open(state).read()) if os.path.exists(state) else 0
            print(f"retry {attempt + 1} from byte {offset}: {e!r}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
