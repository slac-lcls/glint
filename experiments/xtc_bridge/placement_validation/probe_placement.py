"""glint#256 validation: glint_xtc_mpi's placement check under a real `srun --mpi=pmix` launch.

Imports mpi4py FIRST, as glint_xtc_mpi.main() does, then runs the wrapper's own agree_on_placement on
every rank. Each rank also starts a child process (as the --psana 2 reader is started) and asks which
GPU the child's CuPy device 0 is, so the parent's identity can be compared with the child's.
"""
import json
import os
import socket
import subprocess
import sys

from mpi4py import MPI


def libcuda_mapped():
    with open("/proc/self/maps") as f:
        return any("libcuda.so" in line for line in f)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
libcuda_after_mpi_import = libcuda_mapped()
sys.path.insert(0, os.path.join(os.environ["GLINT_ROOT"], "experiments", "xtc_bridge"))
import glint_xtc_mpi as W  # noqa: E402

cvd_before = os.environ.get("CUDA_VISIBLE_DEVICES")
env_before = dict(os.environ)
args = W.build_parser().parse_args(["--exp", "probe", "--run", "1", "--zdist", "0.1"] + sys.argv[1:])
problems, records = W.agree_on_placement(comm, args, os.environ)
env_unchanged = dict(os.environ) == env_before
code = "import cupy; print('PCI ' + cupy.cuda.Device(0).pci_bus_id.upper())"
ch = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
err = ch.stderr.strip().splitlines()
child = ch.stdout.strip() or ("child error: " + (err[-1] if err else "no output"))
mine = next(r for r in records if r["rank"] == rank)
info = {"rank": rank, "host": socket.gethostname(), "cvd": cvd_before, "env_unchanged": env_unchanged,
        "status": mine["status"], "key": mine["key"], "detail": mine["detail"], "child": child,
        "child_matches_parent": child == mine["key"], "libcuda_mapped_after_mpi_import": libcuda_after_mpi_import}
allinfo = comm.gather(info, root=0)
if rank == 0:
    print("ARGS", sys.argv[1:], "| GLINT_GPUS_PER_NODE", os.environ.get("GLINT_GPUS_PER_NODE"))
    print("\n".join(W.placement_summary(records)))
    for p in problems:
        print("PROBLEM", p)
    print("VERDICT", "STOP" if problems else "RUN")
    for i in allinfo:
        print("RANK", json.dumps(i))
