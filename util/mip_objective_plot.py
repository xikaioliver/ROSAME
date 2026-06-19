# import numpy as np
# import matplotlib.pyplot as plt
#
# # Domain labels (2-line for the two BW variants)
# tick_labels = [
#     "BW\n(MNIST)",
#     "Gripper",
#     "Logistics",
#     "BW\n(Synth)",
#     "Hanoi",
#     "8-puzzle",
# ]
# x = np.arange(len(tick_labels))
#
# # === Fill these from your table ===
# # For each domain: State-only (S), State+Action (SA), State+Action+Model (SAM)
#
# # Example for the first three domains (you will extend to all six):
# # NOTE: numbers below are from your table; extend to BW(Synth), Hanoi, 8-puzzle.
# S_err   = np.array([7,   0,   0,   0, 0, 0], dtype=float)
# SA_err  = np.array([0,   0,   0,   0, 0, 0], dtype=float)
# SAM_err = np.array([0,   0,   0,   0, 0, 0], dtype=float)
#
# S_agree   = np.array([0.879, 0.978, 0.983, 0.964, 0.940, 0.985])
# SA_agree  = np.array([0.967, 0.978, 0.983, 0.964, 0.940, 0.985])
# SAM_agree = np.array([0.964, 0.978, 0.983, 0.964, 0.940, 0.985])
#
# S_sacc   = np.array([93.76, 100.00, 99.86, 99.29, 98.55, 99.77])   # %
# SA_sacc  = np.array([99.05,  99.85, 99.91, 99.29, 98.55, 99.77])
# SAM_sacc = np.array([97.81, 100.00, 99.89, 99.29, 98.55, 99.77])
#
# S_aacc   = np.array([56.83, 100.00, 99.67, 88.67, 81.40, 92.60])   # %
# SA_aacc  = np.array([88.33,  99.22, 99.56, 88.67, 81.40, 92.60])
# SAM_aacc = np.array([85.33, 100.00, 99.56, 88.67, 81.40, 92.60])
#
# # === Deltas relative to State-only baseline ===
# # Two series: SA - S, SAM - S
#
# dSA_err   = SA_err   - S_err
# dSAM_err  = SAM_err  - S_err
#
# dSA_agree  = SA_agree  - S_agree
# dSAM_agree = SAM_agree - S_agree
#
# dSA_sacc  = SA_sacc  - S_sacc
# dSAM_sacc = SAM_sacc - S_sacc
#
# dSA_aacc  = SA_aacc  - S_aacc
# dSAM_aacc = SAM_aacc - S_aacc
#
# plt.rcParams.update({
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
# })
#
# fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2))
# (ax1, ax2), (ax3, ax4) = axes
#
# width = 0.35  # bar width
#
# def plot_delta(ax, dSA, dSAM, title, ylabel):
#     ax.bar(x - width/2, dSA,  width, label="State+Action")
#     ax.bar(x + width/2, dSAM, width, label="State+Action+Model")
#     ax.set_title(title, pad=2)
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(x)
#     ax.set_xticklabels(tick_labels, rotation=45, ha="right")
#     ax.axhline(0, linewidth=0.8)
#     ax.grid(axis="y", linestyle=":", linewidth=0.5)
#
#
# # Order: Error, Agreement, State Acc, Action Acc
# plot_delta(ax1, dSA_err,   dSAM_err,   "Error count",     r"$\Delta$ Err")
# plot_delta(ax2, dSA_agree, dSAM_agree, "Agreement",       r"$\Delta$ Agree")
# plot_delta(ax3, dSA_sacc,  dSAM_sacc,  "State accuracy",  r"$\Delta$ State Acc (pp)")
# plot_delta(ax4, dSA_aacc,  dSAM_aacc,  "Action accuracy", r"$\Delta$ Action Acc (pp)")
#
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=2,
#            bbox_to_anchor=(0.5, 1.03), fontsize=8)
#
# fig.suptitle("Effect of MIP objective configuration across domains", fontsize=10)
# fig.tight_layout()
#
# plt.savefig("mip_objective_ablation_deltas.pdf", bbox_inches="tight")
# plt.savefig("mip_objective_ablation_deltas.png", dpi=300, bbox_inches="tight")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# OPTIONAL but recommended
mpl.rcParams['font.family'] = 'DejaVu Sans'

# ------------- Domains (clean compact notation) -------------
tick_labels = [
    "BW-M",    # Blocksworld (MNIST grid)
    "Grip",    # Gripper
    "Log",     # Logistics
    "BW-S",    # Blocksworld (Synthesized)
    "Hanoi",   # Hanoi
    "8-puz",    # 8-puzzle
]
x = np.arange(len(tick_labels))

# ------------- Fill from your table below (order must match) -------------
# Example data from your table sections (please fill missing values)

# If some configs aren't available for a domain, use np.nan
S_err   = np.array([7, 0, 0, 8, 0, 0], dtype=float)
SA_err  = np.array([6, 0, 0, 6, 0, 0], dtype=float)
SAM_err = np.array([0, 0, 0, 0, 0, 0], dtype=float)

S_agree   = np.array([0.876, 0.978, 0.983, 0.820, 0.932, 0.985])
SA_agree  = np.array([0.883, 0.978, 0.983, 0.861, 0.936, 0.985])
SAM_agree = np.array([0.977, 0.978, 0.983, 0.976, 0.940, 0.985])

S_sacc   = np.array([90.79, 100,   99.86, 91.79, 98.18, 99.90])  # %
SA_sacc  = np.array([91.62, 99.85, 99.91, 93.54, 98.43, 99.79])
SAM_sacc = np.array([97.81, 100,   99.89, 99.29, 98.55, 99.77])

S_aacc   = np.array([56.00, 100,   99.67, 52.67, 75.60, 97.60])
SA_aacc  = np.array([60.00, 99.22, 99.56, 63.00, 78.40, 94.40])
SAM_aacc = np.array([85.33, 100,   99.56, 88.67, 81.40, 92.60])

eps = 0.1  # small "visual" bar

plot_S_err  = S_err.copy()
plot_SA_err = SA_err.copy()
plot_SAM_err= SAM_err.copy()

plot_S_err[plot_S_err == 0]   = eps
plot_SA_err[plot_SA_err == 0] = eps
plot_SAM_err[plot_SAM_err == 0] = eps

# --------------------------------------------------------------

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2))
(ax1, ax2), (ax3, ax4) = axes

width = 0.22  # slightly narrower since 3 bars

def grouped(ax, ys, title, ylabel, ylim=None):
    S, SA, SAM = ys
    ax.bar(x - width, S,   width, label="State")
    ax.bar(x,         SA,  width, label="State+Action")
    ax.bar(x + width, SAM, width, label="State+Action+Model (default)")
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    # ---> your request: rotate ticks
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

# ORDER: Error, Agreement, State Acc, Action Acc
grouped(ax1, (plot_S_err, plot_SA_err, plot_SAM_err),
        "Error count", "Err", ylim=(-0.5, 9))

grouped(ax2, (S_agree, SA_agree, SAM_agree),
        "Agreement", "Agree", ylim=(0.8, 1.0))

grouped(ax3, (S_sacc, SA_sacc, SAM_sacc),
        "State accuracy", "State Acc (%)", ylim=(90, 102))

grouped(ax4, (S_aacc, SA_aacc, SAM_aacc),
        "Action accuracy", "Action Acc (%)", ylim=(50, 105))

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.05), fontsize=10)

fig.tight_layout()

plt.savefig("mip_objective_ablation_grouped.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
