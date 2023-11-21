import matplotlib.pyplot as plt
import numpy as np
from SquareHoleSolver import SquareHoleSolver

# global font size for plots
plt.rc("font", size=16)

# N = np.array([16, 32, 64])
N = np.array([16, 32, 64, 128, 256])
ks = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
R = 0.4
kSim = np.zeros((N.shape[0], ks.shape[0]))
err = np.zeros((N.shape[0], ks.shape[0]))

# analytic result (provided by Yang)
vp = np.pi * R**2
alpha = 1
ks = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
Kanal = ks* (R**2*(1+vp)-2*(vp-1)*ks-3*R*(vp+1)*alpha*np.sqrt(ks)) / \
                (R**2*(1-vp)+2*(vp+1)*ks+3*R*(vp-1)*alpha*np.sqrt(ks))
Kanal = ks * (1+vp) / (1-vp)

for j, k in enumerate(ks):
    for i, n in enumerate(N):
        tmp = SquareHoleSolver(N=n, R=R, ks=k, timer=True)
        u, p = tmp.solver()
        kSim[i, j] = tmp.macroPerm(u)
        err[i, j] = np.abs(kSim[i, j]-Kanal[j])/(Kanal[j])
    kSim[:, j] /= ks[j]

fig, axes = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(14)
for j in range(ks.shape[0]):
    axes.plot(N, kSim[:, j], marker="x", label=f"ks: {str(ks[j])}")
axes.axhline(Kanal[-1], linewidth=2, label="Analytic")
axes.set_xlabel("N")
axes.set_ylabel("K/ks")
axes.set_xticks(N)
# axes.set_xlim([N[0], N[-1]])
axes.legend(loc="upper right", ncol=3, fancybox=True)
plt.tight_layout()
plt.savefig("figs/fig1")

fig, axes = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(14)
for j in range(ks.shape[0]):
    axes.plot(N, err[:, j], marker="x", label=f"ks: {str(ks[j])}")
axes.set_xlabel("N")
axes.set_ylabel("Relative error")
axes.set_xticks(N)
# axes.set_xlim([N[0], N[-1]])
axes.legend(loc="upper right", ncol=3, fancybox=True)
plt.tight_layout()
plt.savefig("figs/fig2")

plt.show()