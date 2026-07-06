import numpy as np
mt = np.load("truth.npz")["modes"]; a = []
for line in open("asn.txt"):
    for tok in line.split()[::-1]:
        try: a.append(int(float(tok))); break
        except ValueError: pass
a = np.array(a[:len(mt)])
print("%.1f" % (100 * max((a == mt).mean(), (a != mt).mean())) if len(a) == len(mt) else "NA")
