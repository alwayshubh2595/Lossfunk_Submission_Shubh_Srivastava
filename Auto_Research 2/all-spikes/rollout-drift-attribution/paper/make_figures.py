#!/usr/bin/env python3
"""Paper figures. Fixed identity encoding across all figures:
none = gray (#6b7280, x marker), moment = blue (#2a78d6, o), sigreg = orange (#eb6834, s).
Colors from the validated dataviz reference palette (non-adjacent slots, CVD-safe);
identity is never color-alone (markers + hatching + direct labels)."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

C = {"none": "#6b7280", "moment": "#2a78d6", "sigreg": "#eb6834"}
M = {"none": "x", "moment": "o", "sigreg": "s"}
H = {"none": "///", "moment": None, "sigreg": "\\\\\\"}
ENV_LABEL = {
    "stable-linear-det": "Stable linear (det.)",
    "ou": "OU (stationary)",
    "lorenz": "Lorenz",
    "pointmass-state": "Point mass (state)",
    "pointmass": "Point mass (visual)",
}
PROTO_LABEL = {"1step": "1-step TF", "bptt": "BPTT", "hstep-tf": "H-step TF"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200,
})


def load_rows():
    rows = list(csv.DictReader(open(RESULTS / "analysis" / "metrics_flat.csv")))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def fig1_collapse_matrix(rows):
    """Headline: collapse probability by env x (reg, protocol). Sequential blue."""
    envs = ["stable-linear-det", "ou", "lorenz", "pointmass-state", "pointmass"]
    conds = [("none", "1step"), ("none", "bptt"),
             ("moment", "1step"), ("moment", "bptt"), ("moment", "hstep-tf"),
             ("sigreg", "1step"), ("sigreg", "bptt"), ("sigreg", "hstep-tf")]
    grid = np.full((len(envs), len(conds)), np.nan)
    for i, env in enumerate(envs):
        for j, (reg, proto) in enumerate(conds):
            vals = [r["collapsed"] for r in rows
                    if r["env"] == env and r["regularizer"] == reg and r["protocol"] == proto]
            if vals:
                grid[i, j] = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#2a78d6"])
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(envs)):
        for j in range(len(conds)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color="#9aa1ab")
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > 0.6 else "#1a1a1a", fontsize=7.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{r}\n{PROTO_LABEL[p]}" for r, p in conds], fontsize=6.2)
    for j in (1.5, 4.5):
        ax.axvline(j, color="#1a1a1a", lw=0.9)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([ENV_LABEL[e] for e in envs], fontsize=7.5)
    ax.set_title("Collapse probability by environment and condition")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="collapse prob.")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_collapse_matrix.pdf")
    plt.close(fig)


def fig2_lorenz_interaction(rows):
    """Lorenz: primary metric by protocol x regularizer (only fully-viable env)."""
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    protos = ["1step", "bptt"]
    xs = np.arange(len(protos))
    for reg in ["moment", "sigreg"]:
        means, cis = [], []
        for proto in protos:
            vals = [r["integrated_probe_open_loop_error"] for r in rows
                    if r["env"] == "lorenz" and r["regularizer"] == reg
                    and r["protocol"] == proto and r["collapsed"] == 0.0]
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(xs, means, yerr=cis, color=C[reg], marker=M[reg], ms=5,
                    lw=1.8, capsize=3, label=reg)
        ax.annotate(f"{means[0]:.0f}", (0, means[0]), textcoords="offset points",
                    xytext=(-2, 7), ha="right", fontsize=7, color=C[reg])
        ax.annotate(f"{means[1]:.0f}", (1, means[1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=C[reg])
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([PROTO_LABEL[p] for p in protos])
    ax.set_ylabel("Integrated probe-space error (log)")
    ax.set_title("Lorenz: regularizer matters\nonly under 1-step training")
    ax.legend(frameon=False)
    ax.set_xlim(-0.35, 1.35)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lorenz_interaction.pdf")
    plt.close(fig)


def fig3_horizon_curves(rows):
    """Per-horizon probe error curves, Lorenz viable cells."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    horizons = [1, 5, 10, 25, 50]
    ls = {"1step": "-", "bptt": "--"}
    for reg in ["moment", "sigreg"]:
        for proto in ["1step", "bptt"]:
            sel = [r for r in rows if r["env"] == "lorenz" and r["regularizer"] == reg
                   and r["protocol"] == proto and r["collapsed"] == 0.0]
            ys = [np.mean([r[f"probe_open_loop_error_{h}"] for r in sel]) for h in horizons]
            ax.plot(horizons, ys, color=C[reg], linestyle=ls[proto], marker=M[reg],
                    ms=3.5, lw=1.5, label=f"{reg}, {PROTO_LABEL[proto]}")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout horizon $H$")
    ax.set_ylabel("Probe-space open-loop error")
    ax.set_title("Lorenz: error growth along rollouts")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lorenz_horizons.pdf")
    plt.close(fig)


def fig4_pointmass_bars(rows):
    """Viable point-mass cells (moment only): protocol effect, state + visual."""
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), sharey=False)
    for ax, env in zip(axes, ["pointmass-state", "pointmass"]):
        protos = ["1step", "bptt"] + (["hstep-tf"] if env == "pointmass" else [])
        means, cis = [], []
        for proto in protos:
            vals = [r["integrated_probe_open_loop_error"] for r in rows
                    if r["env"] == env and r["regularizer"] == "moment"
                    and r["protocol"] == proto and r["collapsed"] == 0.0]
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        xs = np.arange(len(protos))
        bars = ax.bar(xs, means, yerr=cis, width=0.55, color=C["moment"],
                      capsize=3, edgecolor="white", linewidth=1)
        for x, m_ in zip(xs, means):
            ax.annotate(f"{m_:.1f}", (x, m_), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([PROTO_LABEL[p] for p in protos], fontsize=7)
        ax.set_title(ENV_LABEL[env], fontsize=8.5)
        ax.margins(y=0.18)
    axes[0].set_ylabel("Integrated probe-space error")
    fig.suptitle("Point mass, viable cells (moment regularizer)", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_pointmass_protocol.pdf")
    plt.close(fig)


def fig5_swap(rows_unused):
    """Swap 2x2 on OU: flat outcome."""
    import glob
    cells = defaultdict(list)
    for p in sorted(glob.glob(str(RESULTS / "swap_runs" / "*" / "metrics.json"))):
        m = json.load(open(p))
        enc = "1-step enc." if "1step__seed" in m.get("load_encoder_from", "") else "BPTT enc."
        dyn = PROTO_LABEL[m["protocol"]]
        cells[(enc, dyn)].append(m["integrated_probe_open_loop_error"])
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    encs = ["1-step enc.", "BPTT enc."]
    dyns = ["1-step TF", "BPTT"]
    width = 0.32
    for k, dyn in enumerate(dyns):
        xs = np.arange(len(encs)) + (k - 0.5) * width
        means = [np.mean(cells[(e, dyn)]) for e in encs]
        cis = [1.96 * np.std(cells[(e, dyn)], ddof=1) / np.sqrt(len(cells[(e, dyn)])) for e in encs]
        ax.bar(xs, means, width=width * 0.94, yerr=cis, capsize=3,
               color="#2a78d6" if k == 0 else "#eb6834",
               hatch=None if k == 0 else "\\\\\\", edgecolor="white", linewidth=1,
               label=f"dyn: {dyn}")
    ax.set_xticks(np.arange(len(encs)))
    ax.set_xticklabels(encs)
    ax.set_ylabel("Integrated probe-space error")
    ax.set_title("OU encoder swap: flat outcome\n(all cells rank-collapsed)")
    ax.legend(frameon=False, fontsize=7)
    ax.margins(y=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_swap.pdf")
    plt.close(fig)


def fig6_effdim_vs_probe(rows):
    """Appendix: what 'collapse' means -- eff dim vs probe R2 scatter."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for reg in ["none", "moment", "sigreg"]:
        sel = [r for r in rows if r["regularizer"] == reg]
        ax.scatter([r["effective_dim"] for r in sel], [r["linear_probe_r2"] for r in sel],
                   c=C[reg], marker=M[reg], s=14, alpha=0.65, label=reg, linewidths=0.8)
    ax.axhline(0.5, color="#9aa1ab", lw=0.8, ls=":")
    ax.axvline(2.0, color="#9aa1ab", lw=0.8, ls=":")
    ax.set_xlabel("Effective latent dimension")
    ax.set_ylabel("Linear probe $R^2$")
    ax.set_title("Two collapse axes: rank vs. decodability")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_effdim_probe.pdf")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    fig1_collapse_matrix(rows)
    fig2_lorenz_interaction(rows)
    fig3_horizon_curves(rows)
    fig4_pointmass_bars(rows)
    fig5_swap(rows)
    fig6_effdim_vs_probe(rows)
    print("figures written to", OUT)


def fig7_lambda_frontier():
    """Appendix: lambda audit -- collapse status and val MSE across the pilot grid."""
    rep = json.load(open(RESULTS / "lambda_selection_report.json"))
    combos = [k for k in sorted(rep) if not k.startswith("stable")]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.4), sharex=True)
    for ax, key in zip(axes.flat, combos):
        env, reg = key.split(":")
        cands = rep[key]["candidates"]
        lams = [c["lambda"] for c in cands]
        mses = [max(c["val_one_step_mse"], 1e-10) for c in cands]
        cols = ["#e34948" if c["collapsed"] else "#1baf7a" for c in cands]
        mks = ["x" if c["collapsed"] else "o" for c in cands]
        for l, m_, co, mk in zip(lams, mses, cols, mks):
            ax.scatter(l, m_, c=co, marker=mk, s=26, linewidths=1.4, zorder=3)
        sel = rep[key]["selected_lambda"]
        ax.axvline(sel, color="#2a78d6", lw=0.9, ls="--", zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ENV_LABEL[env]}\n{reg}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[1, 0].set_xlabel("$\\lambda$"); axes[1, 0].set_ylabel("val 1-step MSE")
    handles = [plt.Line2D([], [], color="#1baf7a", marker="o", ls="", label="survived pilot"),
               plt.Line2D([], [], color="#e34948", marker="x", ls="", label="collapsed pilot"),
               plt.Line2D([], [], color="#2a78d6", ls="--", label="selected $\\lambda$")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(OUT / "fig7_lambda_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


fig7_lambda_frontier()
