"""Figure helpers. All read from cached results, so redrawing is instant."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGS, DATASETS

METRIC = {"direction": ("circ_mae_deg", "circular MAE (deg)"),
          "speed": ("mae", "MAE (m/s)"),
          "acceleration": ("mae", "MAE (m/s^2)")}
# `obj` is drawn faint and explicitly flagged: its mask is built from the disk
# trajectory, i.e. from the label, and the tracker-only oracle beats every encoder
# probe -- so its curve is NOT interpretable as evidence about the encoder. Drawn
# subordinate so the figure cannot be misread as "obj is the best pooling".
POOL_STYLE = {"tmean": ("-", "tab:blue", "uniform mean", 1.8, 1.0),
              "sal": ("-", "tab:green", "top-8 salient (leak-free)", 1.8, 1.0),
              "obj": (":", "tab:red", "tracker-selected (LEAKS LABEL - not interpretable)", 1.0, 0.45)}


def save(fig, name: str) -> "Path":
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def plot_layer_sweep(sweeps: dict, dataset: str, baseline: dict | None = None):
    """sweeps: {pooling: rows}. One panel for R^2, one for the native metric."""
    key, label = METRIC[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pool, rows in sweeps.items():
        ls, c, nm, lw, al = POOL_STYLE[pool]
        L = [r["layer"] for r in rows]
        axes[0].plot(L, [r["r2"] for r in rows], ls, color=c, label=nm, lw=lw, alpha=al)
        axes[1].plot(L, [r[key] for r in rows], ls, color=c, label=nm, lw=lw, alpha=al)
    if baseline is not None:
        lbl = "tracker-only oracle (ceiling)"
        axes[0].axhline(baseline["r2"], color="grey", ls="-.", lw=1, label=lbl)
        axes[1].axhline(baseline[key], color="grey", ls="-.", lw=1, label=lbl)
    axes[0].set_ylabel("$R^2$ (held-out values)")
    axes[1].set_ylabel(label)
    for a in axes:
        a.set_xlabel("residual-stream layer  (25 = post-LayerNorm)")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{dataset}: layerwise probe performance")
    return save(fig, f"layer_sweep_{dataset}")


def plot_decile(rows: list[dict], dataset: str, layer: int):
    """Where the representation runs out of resolution."""
    key, label = METRIC[dataset]
    r = next(x for x in rows if x["layer"] == layer)
    fig, ax = plt.subplots(figsize=(6, 4))
    cen = [c for c in r["decile_centre"] if c is not None]
    err = [e for e, c in zip(r["decile_err"], r["decile_centre"]) if c is not None]
    ax.plot(cen, err, "o-")
    ax.set_xlabel(f"{dataset} label (decile centre)")
    ax.set_ylabel(label)
    ax.set_title(f"{dataset}: error by label decile (layer {layer})")
    ax.grid(alpha=.3)
    return save(fig, f"decile_{dataset}_L{layer}")


def plot_nullspace(res: dict, dataset: str, layer: int):
    """INLP vs random-subspace control. Part 1, step 2."""
    key, label = METRIC[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for nm, c in [("inlp", "tab:red"), ("random", "tab:grey")]:
        h = res[nm]
        d = [r["dims_removed"] for r in h]
        axes[0].plot(d, [r["r2"] for r in h], "o-", color=c, ms=3,
                     label="probe subspace" if nm == "inlp" else "random control")
        axes[1].plot(d, [r[key] for r in h], "o-", color=c, ms=3,
                     label="probe subspace" if nm == "inlp" else "random control")
    axes[0].set_ylabel("$R^2$")
    axes[1].set_ylabel(label)
    for a in axes:
        a.set_xlabel("dimensions removed")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{dataset}: iterative nullspace probing (layer {layer})")
    return save(fig, f"nullspace_{dataset}_L{layer}")


def plot_manifold(man, values, dataset: str, title_extra: str = ""):
    """Three views of the fitted manifold.

    Left:   centroids in the CLIP-PCA basis (the space the spline is fitted in).
            Dominated by within-value nuisance variance -- start position, and for
            `direction` also the speed/acceleration regime -- so the structure looks
            messy even when it is clean.
    Middle: the same centroids re-projected onto their OWN top-2 PCs, which removes
            within-value variance and shows the shape itself.
    Right:  cumulative explained variance, i.e. how far from 2-D the structure is.
    """
    import numpy as np
    cmap = "twilight" if dataset == "direction" else "viridis"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    C = man.centroids_
    u = np.linspace(0, 1, 500)
    crv = man.curve(u)

    sc = axes[0].scatter(C[:, 0], C[:, 1], c=man.values_, cmap=cmap, s=28, zorder=3)
    axes[0].plot(crv[:, 0], crv[:, 1], "-", color="k", lw=1, alpha=.6, zorder=2)
    axes[0].set_xlabel("PC1 (of all clips)"); axes[0].set_ylabel("PC2")
    axes[0].set_title(f"{dataset}: centroids in clip-PCA {title_extra}")

    # re-project centroids onto their own principal axes
    Cc = C - C.mean(0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    Z = Cc @ Vt[:2].T
    Kc = (crv - C.mean(0)) @ Vt[:2].T
    sc2 = axes[1].scatter(Z[:, 0], Z[:, 1], c=man.values_, cmap=cmap, s=30, zorder=3)
    axes[1].plot(Kc[:, 0], Kc[:, 1], "-", color="k", lw=1, alpha=.6, zorder=2)
    axes[1].set_xlabel("centroid-PC1"); axes[1].set_ylabel("centroid-PC2")
    axes[1].set_title("centroids in their own PCs")
    axes[1].set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc2, ax=axes[1], label=dataset, fraction=0.046, pad=0.04)

    evr_c = np.cumsum((np.linalg.svd(Cc, compute_uv=False) ** 2))
    evr_c = evr_c / evr_c[-1]
    axes[2].plot(np.arange(1, len(man.evr_) + 1), np.cumsum(man.evr_), "o-", ms=3,
                 label="all clips")
    axes[2].plot(np.arange(1, len(evr_c) + 1), evr_c, "s-", ms=3, label="centroids only")
    axes[2].axhline(0.9, color="grey", ls=":", lw=1)
    axes[2].set_xlabel("PCA component"); axes[2].set_ylabel("cumulative explained var")
    axes[2].set_title("is the structure really 2-D?")
    axes[2].grid(alpha=.3); axes[2].legend(fontsize=8)
    return save(fig, f"manifold_{dataset}")


def plot_causal(res: dict, dataset: str, layer: int, readout_layers: list[int],
                Ks: list[int]):
    """How far a mid-layer intervention survives as the network keeps computing."""
    key, label = METRIC[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("all", "salient")):
        for K, c in zip(Ks, ["tab:blue", "tab:orange", "tab:purple"]):
            r = res.get((mode, K))
            if r is None:
                continue
            y = [_get(r, R)["steered_vs_target"][key] for R in readout_layers]
            ax.plot([R - layer for R in readout_layers], y, "o-", color=c, label=f"K={K}")
        r0 = res.get((mode, Ks[-1]))
        if r0 is not None:
            fl = np.mean([_get(r0, R)["control_vs_target"][key] for R in readout_layers])
            ax.axhline(fl, color="black", ls="-.", lw=1, label="floor (unsteered)")
        ax.set_xlabel("blocks downstream of the intervention")
        ax.set_title(f"inject: {mode}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel(label)
    fig.suptitle(f"{dataset}: does a layer-{layer} edit survive propagation?")
    return save(fig, f"causal_{dataset}_L{layer}")


def plot_causal_path(res: dict, dataset: str, layer: int, readout: int,
                     ts=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """E3: accuracy and certainty along the manifold path vs the straight chord."""
    key, label = METRIC[dataset]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, c in (("manifold", "tab:green"), ("linear", "tab:red")):
        axes[0].plot(ts, [res[(mode, t)]["err_vs_path"][key] for t in ts], "o-",
                     color=c, label=mode)
        axes[1].plot(ts, [res[(mode, t)]["certainty"] for t in ts], "o-",
                     color=c, label=mode)
    axes[0].set_ylabel(f"{label} vs the value the path is at")
    axes[1].set_ylabel("readout certainty  ||(sin, cos)||" if dataset == "direction"
                       else "readout spread (std)")
    for a in axes:
        a.set_xlabel("t  (0 = unsteered, 1 = target)")
        a.grid(alpha=.3); a.legend(fontsize=9)
    fig.suptitle(f"{dataset}: steering along the manifold vs straight through "
                 f"(edit at L{layer}, read at L{readout})\n"
                 f"endpoints coincide by construction — only the route differs")
    return save(fig, f"causal_path_{dataset}_L{layer}")


def plot_harmonics(spec, noise, layer_spec, pc_rows, dataset="direction"):
    """Three panels: harmonic spectrum vs noise, depth profile, and per-PC assignment."""
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    M = np.arange(1, 13)
    axes[0].bar(M - 0.18, 100 * spec[1:13], width=.36, label="centroid signal")
    axes[0].bar(M + 0.18, 100 * noise[1:13], width=.36, color="grey", label="noise floor")
    axes[0].set_yscale("log")
    axes[0].set_xticks(M)
    axes[0].set_xlabel("Fourier harmonic $m$ (in $\\theta$)")
    axes[0].set_ylabel("% of centroid variance")
    axes[0].set_title("only m = 1, 2, 4 clear the noise")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3, axis="y")

    for m, c in ((1, "tab:blue"), (2, "tab:red"), (4, "tab:green")):
        axes[1].plot(sorted(layer_spec), [100 * layer_spec[L][m] for L in sorted(layer_spec)],
                     "o-", color=c, label=f"m={m}")
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("% of centroid variance")
    axes[1].set_title("the axis code (m=2) grows with depth")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

    pcs = [r["pc"] for r in pc_rows]
    axes[2].bar(pcs, [100 * r["evr"] for r in pc_rows],
                color=["tab:blue" if r["m"] == 1 else "tab:red" if r["m"] == 2
                       else "tab:green" if r["m"] == 4 else "lightgrey" for r in pc_rows])
    for r in pc_rows[:5]:
        axes[2].text(r["pc"], 100 * r["evr"] + 1.5, f"m={r['m']}\n$R^2$={r['r2']:.2f}",
                     ha="center", fontsize=7)
    axes[2].set_xlabel("centroid PC"); axes[2].set_ylabel("% variance")
    axes[2].set_title("each PC is one pure harmonic")
    axes[2].set_ylim(0, 62); axes[2].grid(alpha=.3, axis="y")
    fig.suptitle(f"{dataset}: the ring is a Fourier ladder m ∈ {{1, 2, 4}}, "
                 f"where m≥2 is an axis (180°-invariant) code")
    return save(fig, f"harmonics_{dataset}")


def plot_retention(series: dict, positions: list[str], fname="retention_direction"):
    """Intended-readout alignment as an edit propagates, one line per method."""
    import numpy as np
    x = np.arange(len(positions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, (align, dnorm, col) in series.items():
        axes[0].plot(x, align, "o-", color=col, ms=4, label=name)
        axes[1].plot(x, dnorm, "o-", color=col, ms=4, label=name)
    for ax, lab in ((axes[0], "alignment  ⟨pred, unit(intended)⟩"),
                    (axes[1], "surviving ‖delta‖ (relative)")):
        ax.set_xticks(x[::3]); ax.set_xticklabels(positions[::3], rotation=45, ha="right")
        ax.set_ylabel(lab); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].axhline(0, color="black", lw=.8)
    fig.suptitle("direction: what survives as the edit propagates "
                 "(injected at L16, read through L20)")
    return save(fig, fname)


def plot_leakage(cells: dict, dims=(4, 8, 16), fname="leakage_matrix"):
    """3x3 leakage matrix (two cells not estimable) plus the dimension trend.

    `cells` maps (steer, read) -> {dim: ratio}.
    """
    import numpy as np
    VARS = ["direction", "speed", "acceleration"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    show_dim = 8
    Mx = np.full((3, 3), np.nan)
    for i, sv in enumerate(VARS):
        for j, rv in enumerate(VARS):
            if i == j:
                continue
            Mx[i, j] = cells.get((sv, rv), {}).get(show_dim, np.nan)
    im = axes[0].imshow(np.ma.masked_invalid(Mx), cmap="RdYlGn_r", vmin=0, vmax=2.2)
    for i in range(3):
        for j in range(3):
            if i == j:
                axes[0].text(j, i, "on-target", ha="center", va="center", fontsize=8,
                             color="black", style="italic")
            elif np.isnan(Mx[i, j]):
                axes[0].text(j, i, "not\nestimable", ha="center", va="center",
                             fontsize=8, color="dimgrey")
            else:
                axes[0].text(j, i, f"{Mx[i, j]:.2f}×", ha="center", va="center",
                             fontsize=11, weight="bold")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels([v[:5] for v in VARS])
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(VARS)
    axes[0].set_xlabel("readout watched"); axes[0].set_ylabel("variable steered")
    axes[0].set_title(f"drift ÷ that probe's own MAE  (subspace dim {show_dim})")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="1.0 = drift equals probe noise")

    for (sv, rv), d in cells.items():
        axes[1].plot(list(d), list(d.values()), "o-", label=f"{sv[:5]} → {rv[:5]}")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].text(4.2, 1.03, "drift = probe's own noise", fontsize=8)
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(list(dims))
    axes[1].set_xticklabels([str(d) for d in dims])
    axes[1].set_xlabel("steering subspace dimension")
    axes[1].set_ylabel("drift ÷ baseline MAE")
    axes[1].set_title("specificity degrades as the subspace grows")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    fig.suptitle("cross-variable steering leakage: speed↔acceleration is not estimable "
                 "(no clip has both non-zero)")
    return save(fig, fname)


def plot_ghost_state(d: dict, fname="ghost_state"):
    """The 'ghost state': what stays intact along the linear direction chord.

    Top  — status badges for speed / axis / direction at five points on the path.
    Bottom — the measurements those badges summarise, on the SAME x-axis so the
    badges sit directly above the numbers they encode.
    """
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    t = np.array(d["t"])
    dirc, axc = np.array(d["dir_cert"]), np.array(d["axis_cert"])
    spd = np.array(d["spd_read"])
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]
    at = lambda arr, m: arr[int(np.argmin(abs(t - m)))]
    XLIM = (-0.04, 1.04)

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.30,
                          left=0.19, right=0.90, top=0.90, bottom=0.09)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    OK, WEAK, NULL = "#2e8b57", "#d99a00", "#c0392b"

    def status(v):
        return ("OK", OK) if v > 0.6 else (("WEAK", WEAK) if v > 0.3 else ("NULL", NULL))

    tracks = [("SPEED", None), ("AXIS  (m = 2)", axc), ("DIRECTION  (m = 1)", dirc)]

    ax0.set_xlim(*XLIM)
    ax0.set_ylim(-0.95, 3.5)
    ax0.axis("off")
    for yi, (name, arr) in enumerate(tracks):
        y = 2 - yi
        ax0.text(-0.07, y, name, ha="right", va="center", fontsize=11.5,
                 weight="bold", clip_on=False)
        ax0.plot([0, 1], [y, y], color="0.88", lw=2, zorder=0)
        for m in marks:
            lab, col = ("OK", OK) if arr is None else status(at(arr, m))
            ax0.add_patch(FancyBboxPatch((m - 0.055, y - 0.20), 0.110, 0.40,
                                         boxstyle="round,pad=0.014", linewidth=0,
                                         facecolor=col, zorder=2))
            ax0.text(m, y, lab, ha="center", va="center", color="white",
                     fontsize=10, weight="bold", zorder=3)
    for m in marks:
        ax0.text(m, -0.62, f"t = {m:g}", ha="center", fontsize=10.5)
    ax0.axvspan(0.435, 0.565, ymin=0.10, ymax=0.80, color=NULL, alpha=0.08, zorder=0)
    ax0.annotate("GHOST  STATE\nspeed and axis intact — direction annihilated",
                 xy=(0.5, 3.08), ha="center", va="center", fontsize=12, weight="bold",
                 color=NULL,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#fdf0ee", ec=NULL, lw=1.5))
    ax0.plot([0.5, 0.5], [2.76, 2.28], color=NULL, lw=1.2, ls=":", zorder=1)
    ax0.set_title("steering direction along the straight chord: what survives at each step",
                  fontsize=13, pad=12)

    ax1.axvspan(0.435, 0.565, color=NULL, alpha=0.08)
    ax1.plot(t, axc, "o-", color=OK, lw=2.3, ms=5.5, label="axis (m = 2) certainty")
    ax1.plot(t, dirc, "o-", color=NULL, lw=2.3, ms=5.5, label="direction (m = 1) certainty")
    ax1.axhline(0, color="0.6", lw=.8)
    ax1.annotate(f"{at(dirc, 0.5):.2f}", xy=(0.5, at(dirc, 0.5)), xytext=(0.5, -0.02),
                 ha="center", fontsize=11.5, weight="bold", color=NULL,
                 arrowprops=dict(arrowstyle="-", color=NULL, lw=0))
    ax1.set_ylim(-0.12, 1.42)
    ax1.set_xlim(*XLIM)
    ax1.set_xticks(marks)
    ax1.set_xlabel("t    (0 = source clip,  1 = the opposite direction)", fontsize=11)
    ax1.set_ylabel("readout certainty  ‖(sin, cos)‖")
    ax1.legend(loc="upper left", fontsize=9.5, frameon=False, ncol=2)
    ax1.grid(alpha=.25)

    ax2 = ax1.twinx()
    ax2.plot(t, spd, "s--", color="tab:blue", lw=1.7, ms=4.5, label="speed readout")
    ax2.set_ylim(0, 4.55)
    ax2.set_ylabel("speed readout (m/s)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax2.text(0.5, 2.55, f"speed flat: {spd[0]:.2f} → {spd[-1]:.2f} m/s   "
                        f"(label range 0.25 – 4.0)",
             ha="center", fontsize=9.5, color="tab:blue")
    return save(fig, fname)
