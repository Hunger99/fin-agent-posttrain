"""Hold one free GPU for our job while it starts (research/log/2026-11-01-0010.md, 11-02 07:5x UTC): allocates HOLD_GB of
memory on the GPU this process sees (CUDA_VISIBLE_DEVICES=<one GPU>), then checks every second and frees it and exits as
soon as another process of ours (same user) appears on that GPU, or after MAX_S seconds.
Usage: CUDA_VISIBLE_DEVICES=<gpu> HOLD_GB=36 MAX_S=600 python -B gpu_holder.py <physical gpu index>"""
import getpass, os, subprocess, sys, time
import torch

gpu, me, user = sys.argv[1], os.getpid(), getpass.getuser()
SMI = "/usr/bin/nvidia-smi"  # the real one, never a stub on PATH: a holder must not land on another user's GPU (11-02 08:2x UTC)


def foreign_or_short():
    """True when the GPU has a process of another user, or less free memory than the hold plus 4 GiB."""
    u = subprocess.run([SMI, "-i", gpu, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    if not u:
        return True
    uid, free_mib = [v.strip() for v in u.split(",")]
    out = subprocess.run([SMI, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        a, _, b = line.replace(" ", "").partition(",")
        if a == uid and b.isdigit() and int(b) != me:
            if subprocess.run(["ps", "-o", "user=", "-p", b], capture_output=True, text=True).stdout.strip() != user:
                return True
    return float(free_mib) < (float(os.environ.get("HOLD_GB", "36")) + 4) * 1024


if foreign_or_short():
    sys.exit(0)
x = torch.empty(int(float(os.environ.get("HOLD_GB", "36")) * 2**30), dtype=torch.uint8, device="cuda")
uuid = subprocess.run([SMI, "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
t0 = time.time()
while time.time() - t0 < float(os.environ.get("MAX_S", "600")):
    out = subprocess.run([SMI, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    pids = [int(p) for u, p in (l.replace(" ", "").split(",") for l in out.splitlines() if "," in l) if u == uuid and int(p) != me]
    ours = [p for p in pids if subprocess.run(["ps", "-o", "user=", "-p", str(p)], capture_output=True, text=True).stdout.strip() == user]
    if ours:
        break
    time.sleep(1)
del x
torch.cuda.empty_cache()
