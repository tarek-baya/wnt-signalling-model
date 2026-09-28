    # =============================================================================
    # 11. PHI BLOCK PARAMETERS
    # =============================================================================
    phi_bar = -0.3   # Target lipid composition
    gamma = 3   # Lipid-to-receptor stoichiometry
    K_phi = 1.0 / eps_star   # K_phi: double-well prefactor
    K_eps = eps_star   # K_eps: Laplace prefactor
    m_0 = 1  # phi Mobility prefactor

    # =============================================================================
    # 12. PSI_P BLOCK PARAMETERS
    # =============================================================================
    N_p = (xi_p) / (k_B * theta)   # N_p: coupling of phi to mu_p
    X_p = (chi_p) / (k_B * theta)   # X_p: self-interaction of psiP
    Lambda = (lmbd) / (k_B * theta * (L_star)**2)   # Lambda: psiP gradient penalty
    psiP_star = 0.1  # Confinement threshold

    # =============================================================================
    # 13. MU_PHI COUPLING PARAMETERS
    # =============================================================================
    Pi_a_val = (xi_a * L_star) / (alph * sgm)   # Pi_a: coupling of psiA to mu_phi
    Pi_p_val = (xi_p * L_star) / (alph * sgm)   # Pi_p: coupling of psiP to mu_phi

    # =============================================================================
    # 14. ADAPTIVE TIME-STEPPING PARAMETERS
    # =============================================================================
    target_change = 0.03  # Target change for solutions. If more --> shrink dt ... If less --> grow dt
    target_change_phi = 0.06
    safety_factor = 1  # A conservative factor: always multiply the new determined dt by -- just to make it smaller
    max_amr_allowed = 0.35  # If mesh changes more than +0.35% time step will be reduced
    max_change_allowed = 0.3  # If any solution changes max|_h - _n| > 0.3 steps rejected and time step will be reduced
    dt_reduce_after_change_reject = 0.4  # If step rejected due to large change, dt will be reduced by 60%
    dt_reduce_after_picard_reject = 0.05  # If step rejected due to Picard divergence change, dt will be reduced by 95%
    dt_reduce_after_newton_reject = 0.05  # Same for Newton loop
    dt_reduce_after_amr_change = 0.95  # Same for after AMR
    dt_shrink_rate = 0.25  # dt cannot shrink more than 25% per step (natural time stepping only -- does not apply to rejections)
    dt_growth_rate = 1.5  # dt cannot grow more than 50% per step (natural time stepping only)

    # =============================================================================
    # 15. ADAPTIVE MESH REFINEMENT PARAMETERS
    # =============================================================================
    minLevel = 0  # Smallest level of adaptation
    maxLevel = 3  # Largest level of adaptation
    max_tol_change = 0.5  # Max shift in adapt tolerances per step (dynamic tolerances are used). Prevents sudden mesh changes
    raw_ref_frac = 0.6   # Fraction of global max indicator used for refinement tolerance
    raw_crs_frac = 0.3  # Fraction of global max indicator used for coarsening tolerance

    # =============================================================================
    # 16. UPDATE METHOD: recomputes all derived quantities after base parameter changes
    # Call cfg.update() after changing any base parameter in wntMain
    # =============================================================================
    def update(self):
        # --- Derived from eps, L_star, t_star ---
        self.eps_star = self.eps / self.L_star
        self.D_star = self.L_star**2 / self.t_star

        # --- Derived from eps_star ---
        self.K_phi = 1.0 / self.eps_star
        self.K_eps = self.eps_star

        # --- Derived from xi_p, chi_p, ell_p, lmbd, k_B, theta, L_star ---
        self.lmbd = (self.ell_p**2) * self.chi_p
        self.N_p = (self.xi_p) / (self.k_B * self.theta)
        self.X_p = (self.chi_p) / (self.k_B * self.theta)
        self.Lambda = (self.lmbd) / (self.k_B * self.theta * (self.L_star)**2)

        # --- Derived from xi_a, xi_p, L_star, alph, sgm ---
        self.Pi_a_val = (self.xi_a * self.L_star) / (self.alph * self.sgm)
        self.Pi_p_val = (self.xi_p * self.L_star) / (self.alph * self.sgm)

        print(" • cfg.update() called: all derived quantities recomputed.")

    def snapshot(self):
        result = {}
        for name in dir(self):
            if name.startswith("_") or name == "update" or name == "snapshot":
                continue
            val = getattr(self, name)
            if callable(val):
                continue
            if hasattr(val, "value"):
                val = val.value
            try:
                json.dumps(val)
            except (TypeError, ValueError):
                continue
            result[name] = val
        return result


def setup_output(outputFlag, vtk_folder, plots_folder, working_dir, visTime, visMaxSteps):
    if outputFlag:
        print(f" -- VTK files saved every {visTime} units of physical time or every {visMaxSteps} accepted steps, whichever occurs first. Check Directory: {working_dir}")
    else:
        if comm.rank == 0:
            if os.path.exists(vtk_folder):
                shutil.rmtree(vtk_folder)   # Remove vtk folder created earlier
            if os.path.exists(plots_folder):
                shutil.rmtree(plots_folder)  # Remove plots folder
        comm.barrier()
        print(" -- Plots and VTK files Will NOT Be Saved. Folders removed.")

def setup_checkpoint(checkpoint_flag, checkpoint_dir, checkpointFreq, load_mode):
    if checkpoint_flag:
        print(f" -- Every {checkpointFreq} Steps Checkpoint Will Be Saved. Check Directory: {checkpoint_dir}")
    else:
        if comm.rank == 0:
            if os.path.exists(checkpoint_dir) and load_mode == "fresh":
                shutil.rmtree(checkpoint_dir)  # Remove checkpoint folder
        comm.barrier()
        print(" -- Checkpoint Will NOT Be Saved. Folder removed.")
        if load_mode in ("resume", "ic_only"):
            print("   -- WARNING: Resuming from checkpoint but checkpoint saving is disabled. New progress will not be saved.")

def setup_amr(use_adaptive_mesh, adaptFreq):
    if use_adaptive_mesh:
        print(f" -- Every {adaptFreq} Step Adaptive Mesh Refinement: Used")
    else:
        print(" -- Adaptive Mesh Refinement: Not Used")

def setup_adaptive_dt(use_adaptive_dt):
    if use_adaptive_dt:
        print(" -- Adaptive Time Stepping: Used")
    else:
        print(" -- Adaptive Time Stepping: Not Used")

cfg = Params()