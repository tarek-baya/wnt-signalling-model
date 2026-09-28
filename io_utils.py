# =============================================================================
# io_utils.py
# Checkpoint, plotting, and output utilities for the wntMain.py
# =============================================================================

import os
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
import dune.common.pickle   #type:ignore
from dune.common import comm   #type:ignore
from ufl import inner, grad
from dune.fem import integrate
from config import cfg
import inspect
from functools import partial
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

# =============================================================================
# 1. CHECKPOINT SAVE
# =============================================================================
def save_checkpoint(gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h,
    actTime, dt_val, steps, visCounter, accepted_steps, prev_refineTol, prev_coarsenTol,
    z, M_R, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_const, checkpoint_path):
    state = { # Pack all needed states into a dictionary
        "actTime": actTime,
        "dt": dt_val,
        "steps": steps,
        "visCounter": visCounter,
        "accepted_steps": accepted_steps,
        "prev_refineTol": prev_refineTol,
        "prev_coarsenTol": prev_coarsenTol,
        "z": z.tolist(),
        "M_R": M_R,
        "V_b": float(V_b_const.value),
        "Wb_time_list": Wb_time_list,
        "Wb_vals": Wb_vals,
        "Vb_time_list": Vb_time_list,
        "Vb_vals": Vb_vals,
        "cfg_snapshot": cfg.snapshot(),
    }
    # dune.common.pickle.dump triggers MPI collectives internally. All ranks must call it simultaneously
    # If only rank 0 calls it, other ranks skip ahead causing a deadlock in the next gridAdapt (I think specifcally for refinement not coarsening)
    # Explanation taken with a pinch of salt! I figured this out via "trial and error" so no idea what is happening behind scenes. Might want to check for yourself
    buf = io.BytesIO() # RAM buffer: temporary holding place which pickle writes into it
    dune.common.pickle.dump([gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, state], buf)
    rank_checkpoint_path = f"{checkpoint_path}.rank{comm.rank}"
    with open(rank_checkpoint_path, "wb") as f:
        f.write(buf.getvalue())
    comm.barrier() # Ranks wait until all ranks have finished writing before continuing

    meta_path = f"{checkpoint_path}.meta.json"
    if comm.rank == 0:
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                old_meta = json.load(f)
            old_ranks = old_meta.get("saved_with_ranks", 0)
            for old_rank in range(comm.size, old_ranks):
                orphan_path = f"{checkpoint_path}.rank{old_rank}"
                if os.path.exists(orphan_path):
                    os.remove(orphan_path)
        with open(meta_path, "w") as f:
            json.dump({"saved_with_ranks": comm.size}, f)
    comm.barrier()
    print(f" • Checkpoint saved: actTime={actTime:.4e}, step={steps}")

def save_experiment_record(cfg, plots_folder, vtk_folder, checkpoint_path, g_L_func_src, build_ic_func, mesh_type, load_mode, job_id, job_name, working_dir, filename=None):
    """
    Saves a record of all experiment parameters to a .txt file.
    Call once at the start of each run, before the time loop.
    """
    if comm.rank != 0:
        return
    if filename is None:
        filename = os.path.join(working_dir, f"experiment_record_{cfg.Experiment}_{cfg.SubExperiment}_{job_id}.txt")

    from datetime import datetime
    import textwrap

    def row(label, val):
        f.write(f"{label}: {val}\n")

    with open(filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT RECORD\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        row("Experiment", cfg.Experiment)
        row("Sub-Experiment", cfg.SubExperiment)
        row("Experiment Description", cfg.DescriptionExperiment)
        row("Aim of Experiment", cfg.AimExperiment)
        row("load_mode", load_mode)
        row("job_id", job_id)
        row("job_name", job_name)
        row("working_dir", working_dir)
        row("plots_folder", plots_folder)
        row("vtk_folder", vtk_folder)
        row("checkpoint_path", checkpoint_path)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("MESH AND DOMAIN\n")
        f.write("-" * 70 + "\n")
        row("mesh_type", mesh_type)
        row("bbox_margin", cfg.bbox_margin)
        row("pOrder", cfg.pOrder)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("TIME DISCRETISATION\n")
        f.write("-" * 70 + "\n")
        row("finalT", cfg.finalT)
        row("dt_initial", cfg.dt.value)
        row("use_adaptive_dt", cfg.use_adaptive_dt)
        row("target_change", cfg.target_change)
        row("safety_factor", cfg.safety_factor)
        row("dt_shrink_rate", cfg.dt_shrink_rate)
        row("dt_growth_rate", cfg.dt_growth_rate)
        row("max_change_allowed", cfg.max_change_allowed)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("SOLVERS\n")
        f.write("-" * 70 + "\n")
        row("solver_verbose", cfg.solver_verbose)
        row("newtonTol", cfg.newtonTol)
        row("newtonMaxIt", cfg.newtonMaxIt)
        row("linear_method", cfg.linear_method)
        row("preconditioner", cfg.preconditioner)
        row("linearMaxIt", cfg.linearMaxIt)
        row("picardTol", cfg.picardTol)
        row("picardMaxIt", cfg.picardMaxIt)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("AMR\n")
        f.write("-" * 70 + "\n")
        row("use_adaptive_mesh", cfg.use_adaptive_mesh)
        row("minLevel", cfg.minLevel)
        row("maxLevel", cfg.maxLevel)
        row("raw_ref_frac", cfg.raw_ref_frac)
        row("raw_crs_frac", cfg.raw_crs_frac)
        row("max_tol_change", cfg.max_tol_change)
        row("adaptFreq", cfg.adaptFreq)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("ODE AND INITIAL VALUES\n")
        f.write("-" * 70 + "\n")
        row("W_b_init", cfg.W_b_init)
        row("V_b_init", cfg.V_b_init)
        f.write("g_L_func source:\n")
        f.write(textwrap.dedent(inspect.getsource(g_L_func_src)))
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("INITIAL CONDITIONS (source of build_initial_conditions)\n")
        f.write("-" * 70 + "\n")
        f.write(textwrap.dedent(inspect.getsource(build_ic_func)))
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("PHYSICAL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        row("beta_bump", cfg.beta_bump)
        row("phi_bar", cfg.phi_bar)
        row("gamma", cfg.gamma)
        row("K_phi", cfg.K_phi)
        row("K_eps", cfg.K_eps)
        row("m_0", cfg.m_0)
        row("N_p", cfg.N_p)
        row("X_p", cfg.X_p)
        row("Lambda", cfg.Lambda)
        row("psiP_star", cfg.psiP_star)
        row("Pi_a_val", cfg.Pi_a_val)
        row("Pi_p_val", cfg.Pi_p_val)
        row("eps_star", cfg.eps_star)
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("REACTION RATE CONSTANTS\n")
        f.write("-" * 70 + "\n")
        for name, val in [
            ("k_ads", cfg.k_ads), ("k_des", cfg.k_des), ("k_on", cfg.k_on),
            ("k_off", cfg.k_off), ("k_phs", cfg.k_phs), ("k_dph", cfg.k_dph),
            ("k_int", cfg.k_int), ("k_phi", cfg.k_phi),
        ]:
            row(name, val)
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF RECORD\n")
        f.write("=" * 70 + "\n")

    print(f" • Experiment record saved: {filename}")
    
# =============================================================================
# 2. PLOTTING FUNCTION
# =============================================================================
def fmt_tick_vals(v):
    if v == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(v))))
    coeff = v / 10**exp
    # If exponent is 0, 1, or -1 just write the number plainly
    if exp in (0, 1, -1):
        val = v if v == int(v) else round(v, 4)
        return r"$" + str(val) + r"$"
    if abs(coeff - 1.0) < 0.05:
        return r"$10^{" + str(exp) + r"}$"
    else:
        coeff_str = f"{coeff:.1f}".rstrip('0').rstrip('.')
        return r"$" + coeff_str + r"{\times}10^{" + str(exp) + r"}$"

def plot_timeseries(time_array, data_array, ylabel, title, filename=None, color="#2166ac", round_data=False, x_limits=None, y_limits=None):
    """
    Creates and saves a high-quality time-series figure.
    """
    # Only rank 0 does any plotting -- prevents memory exhaustion on all other ranks
    if comm.rank != 0:
        return

    if filename is not None:
        base = os.path.splitext(filename)[0]
        save_results_json(
            {"time": list(time_array), "data": list(data_array), "ylabel": ylabel,
            "title": title, "color": color, "x_limits": x_limits, "y_limits": y_limits,},
            base + "_data.json")

    FONTSIZE_LABEL = 7
    FONTSIZE_TITLE = 8
    FONTSIZE_TICK = 6

    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    plot_data = np.round(data_array, 10) if round_data else data_array

    ax.plot(time_array, plot_data, linewidth=1.2, color=color, solid_capstyle="round", zorder=3)
    ax.scatter(time_array, plot_data, s=0, color=color, zorder=0, linewidths=0)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", which="major", direction="in", length=3, width=0.6, labelsize=FONTSIZE_TICK, pad=3)
    ax.tick_params(axis="both", which="minor", direction="in", length=1.5, width=0.5)
    ax.grid(True, which="major", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Time $[\mathrm{s}]$", fontsize=FONTSIZE_LABEL, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL, labelpad=4)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, pad=4)

    if x_limits is not None:
        ax.set_xlim(*x_limits)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        data_range = plot_data.max() - plot_data.min()
        if data_range > 0:
            margin = 0.05 * data_range
            ax.set_ylim(plot_data.min() - margin, plot_data.max() + margin)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_tick_vals(v) if v != 0 else r"$0$"))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_tick_vals(v) if v != 0 else r"$0$"))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    fig.tight_layout(pad=0.5)

    if filename is not None:
        base = os.path.splitext(filename)[0]
        for ext in [".png", ".pdf", ".svg"]:
            fig.savefig(base + ext, bbox_inches="tight", facecolor="white", dpi=300)
    else:
        print(" • No filename provided, plot not saved.")

    plt.close(fig)

# =============================================================================
# 3. JSON SAVE / LOAD
# =============================================================================
def save_results_json(results_dict, filename):
    if comm.rank != 0:
        return
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)

def load_results_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
# =============================================================================
# 4. COMPUTE ERRORS FUNCTIONS
# =============================================================================    
def picard_H1_error(psiF_m1, psiA_m1, psiP_m1, psiF_m, psiA_m, psiP_m, gridView):
    """Compute the absolute and relative H1(Gamma)^3 Picard iteration errors."""
    dF = psiF_m1 - psiF_m
    dA = psiA_m1 - psiA_m
    dP = psiP_m1 - psiP_m

    err_abs_sq = float(integrate(
        dF**2 + inner(grad(dF), grad(dF))
        + dA**2 + inner(grad(dA), grad(dA))
        + dP**2 + inner(grad(dP), grad(dP)),
        gridView, order=10,))

    norm_new_sq = float(integrate(
        psiF_m1**2 + inner(grad(psiF_m1), grad(psiF_m1))
        + psiA_m1**2 + inner(grad(psiA_m1), grad(psiA_m1))
        + psiP_m1**2 + inner(grad(psiP_m1), grad(psiP_m1)),
        gridView, order=10,))

    err_abs = np.sqrt(max(err_abs_sq, 0.0))
    norm_new = np.sqrt(max(norm_new_sq, 0.0))
    err_rel = err_abs / max(1.0, norm_new)

    return err_abs, err_rel, norm_new

def picard_max_iterations(err_abs_first, tol):
    """Compute the maximum Picard iteration count from the first absolute correction."""
    if err_abs_first <= tol:
        return 1

    return 1 + int(np.ceil(np.log(err_abs_first / tol) / np.log(2.0)))
