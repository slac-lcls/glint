"""Minimal LCLS elog query (no lute import): Kerberos-authed GET to the lgbk API.
Characterizes an experiment: description/sample, run count, run-table params."""
import sys

import requests

exp = sys.argv[1] if len(sys.argv) > 1 else "mfx101343025"
base = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"

headers, auth = {}, None
try:
    from krtc import KerberosTicket
    headers = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
except Exception as e:
    print(f"krtc unavailable ({e}); trying requests_kerberos")
    from requests_kerberos import OPTIONAL, HTTPKerberosAuth
    auth = HTTPKerberosAuth(mutual_authentication=OPTIONAL)

for ep in [f"{exp}/ws/info", f"{exp}/ws/runs", f"{exp}/ws/run_table_parameters"]:
    try:
        r = requests.get(f"{base}/{ep}", headers=headers, auth=auth, timeout=30)
        j = r.json()
        v = j.get("value", j) if isinstance(j, dict) else j
        print(f"\n===== {ep}  [{r.status_code}] =====")
        if ep.endswith("/runs") and isinstance(v, list):
            print(f"total runs: {len(v)}")
            if v:
                print("first run entry:", str(v[0])[:400])
                print("last run entry: ", str(v[-1])[:400])
        else:
            print(str(v)[:1200])
    except Exception as e:
        print(f"\n===== {ep}  EXC {type(e).__name__}: {e}")
