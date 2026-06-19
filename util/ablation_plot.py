import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# OPTIONAL but recommended
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Domains (short labels to save horizontal space)
domains = [
    "BW-M",    # Blocksworld (MNIST grid)
    "Grip",    # Gripper
    "Log",     # Logistics
    "BW-S",    # Blocksworld (Synthesized)
    "Hanoi",   # Hanoi
    "8-puz",    # 8-puzzle
]
x = np.arange(len(domains))

# Raw numbers from your table
# Order: [BW-M, Gripper, Logistics, BW-S, Hanoi, 8-puzzle]
err_wo   = np.array([10, 6, 0, 4, 0, 0], dtype=float)
err_w    = np.array([0,  0, 0, 0, 0, 0], dtype=float)

agree_wo = np.array([0.784, 0.724, 0.979, 0.899, 0.926, 0.985])
agree_w  = np.array([0.977, 0.978, 0.983, 0.976, 0.940, 0.985])

sacc_wo  = np.array([89.22, 86.22, 99.93, 93.90, 97.15, 99.90])
sacc_w   = np.array([97.81, 100,   99.89, 99.29, 98.55, 99.77])

aacc_wo  = np.array([13.67, 7.6, 99.67, 66.67, 57.60, 97.40])
aacc_w   = np.array([85.33, 100, 99.56, 88.67, 81.40, 92.60])

# Deltas: with MIP minus without MIP
delta_err   = err_w   - err_wo              # negative is good
delta_agree = agree_w - agree_wo            # higher is better
delta_sacc  = sacc_w  - sacc_wo             # percentage points
delta_aacc  = aacc_w  - aacc_wo             # percentage points

# --- Plot ---
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.2))  # small footprint
(ax1, ax2), (ax3, ax4) = axes

bar_kwargs = dict(width=0.6)

ax1.bar(x, delta_err, **bar_kwargs)
ax1.set_title("Error count", fontsize=10, pad=4)
ax1.set_ylabel(r"Δ Err", fontsize=10, )
ax1.set_xticks(x)
ax1.set_xticklabels(domains, rotation=45, ha="right")
ax1.axhline(0, linewidth=0.8)

# 2) Agreement (top-right)
ax2.bar(x, delta_agree, **bar_kwargs)
ax2.set_title("Agreement", fontsize=10, pad=4)
ax2.set_ylabel(r"Δ Agree", fontsize=10, )
ax2.set_xticks(x)
ax2.set_xticklabels(domains, rotation=45, ha="right")
ax2.axhline(0, linewidth=0.8)

# 3) State accuracy (bottom-left)
ax3.bar(x, delta_sacc, **bar_kwargs)
ax3.set_title("State accuracy", fontsize=10, pad=4)
ax3.set_ylabel(r"Δ State Acc (%)", fontsize=10, )
ax3.set_xticks(x)
ax3.set_xticklabels(domains, rotation=45, ha="right")
ax3.axhline(0, linewidth=0.8)

# 4) Action accuracy (bottom-right)
ax4.bar(x, delta_aacc, **bar_kwargs)
ax4.set_title("Action accuracy", fontsize=10, pad=4)
ax4.set_ylabel(r"Δ Action Acc (%)", fontsize=10, )
ax4.set_xticks(x)
ax4.set_xticklabels(domains, rotation=45, ha="right")
ax4.axhline(0, linewidth=0.8)

fig.tight_layout()

# Save to file for LaTeX
plt.savefig("mip_ablation_deltas.pdf", bbox_inches="tight", pad_inches=0)
plt.show()









# import numpy as np
# import matplotlib.pyplot as plt
#
# # Domains (short labels)
# domains = [
#     "BW-M",   # Blocksworld (MNIST grid)
#     "Grip",   # Gripper
#     "Log",    # Logistics
#     "BW-S",   # Blocksworld (Synthesized)
#     "Hanoi",  # Hanoi
#     "8puz",   # 8-puzzle
# ]
# x = np.arange(len(domains))
#
# # Raw numbers from your table
# # Order: [BW-M, Gripper, Logistics, BW-S, Hanoi, 8-puzzle]
# err_wo   = np.array([10,   6, 0, 4, 0, 0], dtype=float)
# err_w    = np.array([ 0,   0, 0, 0, 0, 0], dtype=float)
#
# agree_wo = np.array([0.792, 0.724, 0.979, 0.905, 0.926, 0.985])
# agree_w  = np.array([0.964, 0.978, 0.983, 0.964, 0.940, 0.985])
#
# sacc_wo  = np.array([89.22, 86.22, 99.93, 93.90, 97.15, 99.90])
# sacc_w   = np.array([97.81,100.00, 99.89, 99.29, 98.55, 99.77])
#
# aacc_wo  = np.array([13.67,  7.60, 99.67, 66.67, 57.60, 97.40])
# aacc_w   = np.array([85.33,100.00, 99.56, 88.67, 81.40, 92.60])
#
# # Plot settings
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
# def grouped_bars(ax, y_wo, y_w, title, ylabel):
#     ax.bar(x - width/2, y_wo, width, label="w/o MIP")
#     ax.bar(x + width/2, y_w,  width, label="with MIP")
#     ax.set_title(title, pad=2)
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(x)
#     ax.set_xticklabels(domains, rotation=45, ha="right")
#     ax.grid(axis="y", linestyle=":", linewidth=0.5)
#
# # 1) Err
# grouped_bars(ax1, err_wo, err_w, "Error count", "Err")
#
# # 2) Agreement
# grouped_bars(ax2, agree_wo, agree_w, "Agreement", "Agree")
#
# # 3) State accuracy
# grouped_bars(ax3, sacc_wo, sacc_w, "State accuracy", "State Acc (%)")
#
# # 4) Action accuracy
# grouped_bars(ax4, aacc_wo, aacc_w, "Action accuracy", "Action Acc (%)")
#
# # Put a single legend for all subplots
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=8)
#
# fig.suptitle("Effect of MIP correction across domains", fontsize=10)
# fig.tight_layout()
#
# # Save for LaTeX
# plt.savefig("mip_ablation_grouped.pdf", bbox_inches="tight")
# plt.savefig("mip_ablation_grouped.png", dpi=300, bbox_inches="tight")
# plt.show()
