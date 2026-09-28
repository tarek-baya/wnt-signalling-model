# =============================================================================
# ode_solver.py
# ODE solver and algebraic constraint functions for the wntMain.py.
# =============================================================================

from scipy.integrate import solve_ivp
from dune.fem import integrate
from dune.common import comm   #type:ignore
from config import cfg
from functools import partial
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

# =============================================================================
# 1. ODE AND ALGEBRAIC CONSTRAINT FUNCTIONS
# =============================================================================
def ode_rhs(t, z, psiF_h, psiA_h, gridView, vol_ext, g_L_func):
    """Computes the RHS of the ODE for extracellular Wnt W_b."""
    W_b = z[0]
    int_psiF = float(integrate(psiF_h, gridView, order=10))
    int_psiA = float(integrate(psiA_h, gridView, order=10))
    dWb_dt = (1.0 / vol_ext) * (cfg.k_off * int_psiA - cfg.k_on * W_b * int_psiF + g_L_func(t))
    return [dWb_dt]

def solve_ode(t_start, t_end, z, psiF_h, psiA_h, gridView, vol_ext, g_L_func):
    """Solve the W_b ODE from t_start to t_end via scipy solve_ivp (Radau implicit method)."""
    sol = solve_ivp(
        fun=lambda t, z: ode_rhs(t, z, psiF_h, psiA_h, gridView, vol_ext, g_L_func),
        t_span=(t_start, t_end), y0=z,method="Radau",  # Implicit method
        rtol=1e-8, atol=1e-10,)
    if not sol.success: # raise error if solving failed
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.y[:, -1]  # Return solution at t_end

def compute_V_b(psiF_h, psiA_h, psiP_h, gridView, M_R, vol_in):
    """Compute intracellular receptor concentration V_b from total mass conservation:
        M_R = (vol_in * V_b) + int(psiF) + int(psiA) + int(psiP)."""
    int_psiF = float(integrate(psiF_h, gridView, order=10))
    int_psiA = float(integrate(psiA_h, gridView, order=10))
    int_psiP = float(integrate(psiP_h, gridView, order=10))
    return (M_R - int_psiF - int_psiA - int_psiP) / vol_in

def update_pde_coefficients(z, psiF_h, psiA_h, psiP_h, gridView, W_b_const, V_b_const, M_R, vol_in):
    """Update the nonlocal ODE-PDE coupling coefficients W_b and V_b after each accepted time step."""
    W_b_const.value = float(z[0])
    V_b_const.value = compute_V_b(psiF_h, psiA_h, psiP_h, gridView, M_R, vol_in)