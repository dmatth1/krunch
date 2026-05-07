#!/usr/bin/env python3
"""Render the README throughput chart to assets/throughput.png.

Measured points: 1× and 4× A10G (g5.xlarge). Extrapolated dotted lines
project linear scaling through 8, 16, 32, 64 instances based on the
slope through the two measurements. Re-run after new measurements land.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# (instances, compress_kbps, decompress_kbps)
MEASURED = [
    (1, 180,  75),
    (4, 696, 261),
]
EXTRAPOLATED_X = [8, 16, 32, 64]

OUT = Path(__file__).resolve().parents[1] / "assets" / "throughput.png"

# Linear fit y = a*N + b through the two measured points (KB/s)
def fit(s):
    a = (s[1] - s[0]) / (MEASURED[1][0] - MEASURED[0][0])
    b = s[0] - a * MEASURED[0][0]
    return a, b

# Convert to MB/s for plotting
KB_PER_MB = 1024.0
cmp_meas_mb = [m[1] / KB_PER_MB for m in MEASURED]
dec_meas_mb = [m[2] / KB_PER_MB for m in MEASURED]
m_x = [m[0] for m in MEASURED]

a_c, b_c = fit([m[1] for m in MEASURED])
a_d, b_d = fit([m[2] for m in MEASURED])
cmp_ext_mb = [(a_c * x + b_c) / KB_PER_MB for x in EXTRAPOLATED_X]
dec_ext_mb = [(a_d * x + b_d) / KB_PER_MB for x in EXTRAPOLATED_X]

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

C_BLUE = "#1f77b4"
C_RED  = "#d62728"

ax.plot(m_x, cmp_meas_mb, "o-", color=C_BLUE, linewidth=2.2, markersize=9,
        label="compress (measured)")
ax.plot(m_x, dec_meas_mb, "s-", color=C_RED, linewidth=2.2, markersize=9,
        label="decompress (measured)")

ax.plot([m_x[-1], *EXTRAPOLATED_X], [cmp_meas_mb[-1], *cmp_ext_mb],
        ":", color=C_BLUE, linewidth=2,
        label="compress (extrapolated)")
ax.plot([m_x[-1], *EXTRAPOLATED_X], [dec_meas_mb[-1], *dec_ext_mb],
        ":", color=C_RED, linewidth=2,
        label="decompress (extrapolated)")

def fmt(v):
    return f"{v:.2f}" if v < 1 else f"{v:.1f}"

# All compress labels go ABOVE the curve; all decompress labels go BELOW.
# Keeps the line + label well separated regardless of point position.
for x, y in zip(m_x + EXTRAPOLATED_X, cmp_meas_mb + cmp_ext_mb):
    ax.annotate(fmt(y), (x, y), textcoords="offset points",
                xytext=(0, 10), fontsize=9, color=C_BLUE,
                ha="center", va="bottom")
for x, y in zip(m_x + EXTRAPOLATED_X, dec_meas_mb + dec_ext_mb):
    ax.annotate(fmt(y), (x, y), textcoords="offset points",
                xytext=(0, -12), fontsize=9, color=C_RED,
                ha="center", va="top")

all_x = m_x + EXTRAPOLATED_X
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(all_x)
ax.set_xticklabels([str(x) for x in all_x], fontsize=10)
ax.tick_params(axis="x", which="minor", bottom=False)

# Cleaner y-axis: only major decade ticks, no minor ticks/gridlines
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(mticker.NullLocator())

ax.set_xlabel("Number of A10G instances")
ax.set_ylabel("Aggregate throughput (MB/s)")
ax.set_title("Krunch throughput vs fleet size  (100 MB WildChat-English, real-work)")

# Major-only horizontal gridlines, no vertical
ax.grid(True, axis="y", which="major", alpha=0.3)
ax.grid(False, axis="x")

ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

# Headroom so top labels don't kiss the frame
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin * 0.7, ymax * 1.4)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
