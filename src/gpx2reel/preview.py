"""Quick 2D preview of a route: map coloured by day + elevation profile."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .storyboard import DAY_COLORS  # noqa: E402
from .world import projection_of  # noqa: E402


def render_preview(route: dict, out: Path, day_colors: list[str] | None = None) -> Path:
    colors = day_colors or DAY_COLORS
    proj = projection_of(route)

    fig = plt.figure(figsize=(7, 9), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1])

    base = route["totals"]["min_ele_m"] or 0
    for d in route["days"]:
        c = colors[(d["day"] - 1) % len(colors)]
        first = True
        for s in d["segments"]:
            pts = np.array([[q[0], q[1]] for q in s["points"]])
            x, y = proj.to_xy(pts[:, 0], pts[:, 1])
            ax.plot(x / 1000, y / 1000, color=c, lw=2, solid_capstyle="round",
                    label=f"Day {d['day']} · {d['distance_km']:.0f} km" if first else None)
            if first:
                ax.annotate(str(d["day"]), (x[0] / 1000, y[0] / 1000), fontsize=9, weight="bold",
                            ha="center", va="center", color="white",
                            bbox=dict(boxstyle="circle,pad=0.25", fc=c, ec="none"))
                first = False
            ele = np.array([np.nan if q[2] is None else q[2] for q in s["points"]], float)
            km = np.array([q[4] for q in s["points"]])
            axp.plot(km, ele, color=c, lw=1.5)
            axp.fill_between(km, base, ele, color=c, alpha=0.15)

    for g in route["gaps"]:
        a, b = g["from"], g["to"]
        x, y = proj.to_xy(np.array([a[0], b[0]]), np.array([a[1], b[1]]))
        ax.plot(x / 1000, y / 1000, color="#666", lw=1.5, ls=(0, (4, 3)))

    t = route["totals"]
    ax.set_title(f"{t['days']} days · {t['distance_km']:.0f} km · {t['ascent_m']} m climbed", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("km")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=False)
    axp.set_xlabel("km from start")
    axp.set_ylabel("elevation, m")
    axp.grid(alpha=0.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
