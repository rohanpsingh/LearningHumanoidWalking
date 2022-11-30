import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv)<2:
    print("Usage: python plot_logs.py log1/train.txt [log2/train.txt ...]")
    sys.exit(-1)

fns = sys.argv[1:]

fig, ax = plt.subplots(3, 1)

for fn in fns:
    with open(fn, "r") as f:
        lines = [list(map(float, l.strip().split(','))) for l in f.readlines()[1:]]
    data = np.array(lines)
    s = os.path.split(os.path.dirname(fn))[-1]

    ax[0].plot(data[:,0], label="{}(mean={})".format(s, repr(np.round(np.mean(data[:,0]), 2))))
    ax[1].plot(data[:,1], label="{}(mean={})".format(s, repr(np.round(np.mean(data[:,1]), 2))))
    ax[2].plot(data[:,0]/data[:,1], label="{}(mean={})".format(s, repr(np.round(np.mean(data[:,0]/data[:,1]), 2))))

ax[0].set_ylabel('returns')
ax[1].set_ylabel('ep-lengths')
ax[2].set_ylabel('per-step-returns')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
