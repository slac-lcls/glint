import numpy as np, sys
d = np.load("truth.npz")
Itrue = {tuple(int(x) for x in k): float(v) for k, v in zip(d["keys"], d["vals"])}
def laue4m(t):
    h, k, l = t
    return max([(h,k,l),(-k,h,l),(-h,-k,l),(k,-h,l),(-h,-k,-l),(k,-h,-l),(h,k,-l),(-k,h,-l)])
xs, ys = [], []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 4: continue
    try: h, k, l, I = int(p[0]), int(p[1]), int(p[2]), float(p[3])
    except ValueError: continue
    c = laue4m((h, k, l))
    if c in Itrue: xs.append(I); ys.append(Itrue[c])
xs, ys = np.array(xs), np.array(ys)
cc = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
print(f"  {sys.argv[1]}: CC(merged I, ground truth) = {cc:.3f}   over {len(xs)} reflections")
