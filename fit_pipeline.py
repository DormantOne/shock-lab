"""
Fit Pipeline v3 — With Tipping Points
=======================================
Uses the v3 reservoir core with learned multi-channel tipping points.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

from reservoir_core import (
    INPUTS, TARGETS, INPUT_NAMES, TARGET_NAMES, ALL_NAMES,
    N_INPUT, N_TARGET, N_SOURCE, N_PARAMS, N_WEIGHTS, N_TIP_PARAMS, K_TIPS,
    OBSERVABLE_TS, OBSERVABLE_TS_IDX,
    normalize, denormalize, pack_params, unpack_params,
    get_bounds, random_params, prune_weights,
    analyze_hidden, analyze_tipping_points, to_gdmt_json, _sigmoid,
)

VASO_MAP = {
    "none": 0, "norepinephrine": 1, "vasopressin": 2,
    "dopamine": 3, "dobutamine": 4, "phenylephrine": 5,
}
MECH_MAP = {"none": 0, "IABP": 1, "impella": 2, "ECMO": 3}

FAST_DT = 1.0
FAST_STEPS = 24
FAST_TARGET_STEPS = [6, 12, 24]


def fast_forward_batch(flat_params, X_input, X_init):
    """Optimized forward sim with tipping points."""
    W, eq, spring, thresh, steep, mag = unpack_params(flat_params)
    N = X_input.shape[0]

    traj = np.zeros((N, N_TARGET, FAST_STEPS + 1))
    state = X_init.copy()
    traj[:, :, 0] = state

    sd = spring[None, :] * FAST_DT
    eq_b = eq[None, :]

    # Pre-extract per-channel tipping params as (N_TARGET,) slices
    tip_thresh = [thresh[:, k] for k in range(K_TIPS)]  # list of (N_TARGET,)
    tip_steep = [steep[:, k] for k in range(K_TIPS)]
    tip_mag = [mag[:, k] for k in range(K_TIPS)]

    sig_buf = np.empty_like(state)  # reusable buffer

    for step in range(FAST_STEPS):
        # Linear spring
        full = np.concatenate([X_input, state], axis=1)
        eff_eq = eq_b + full @ W

        # Tipping forces — loop over K channels (K=3, so loop is trivial)
        tip_force = np.zeros_like(state)
        for k in range(K_TIPS):
            np.subtract(state, tip_thresh[k], out=sig_buf)
            np.multiply(sig_buf, tip_steep[k], out=sig_buf)
            np.clip(sig_buf, -20, 20, out=sig_buf)
            np.negative(sig_buf, out=sig_buf)
            np.exp(sig_buf, out=sig_buf)
            np.add(sig_buf, 1.0, out=sig_buf)
            np.divide(1.0, sig_buf, out=sig_buf)  # sig_buf is now sigmoid
            tip_force += tip_mag[k] * sig_buf

        state = state + (eff_eq - state) * sd + tip_force * FAST_DT
        np.clip(state, 0, 1, out=state)
        traj[:, :, step + 1] = state

    return traj


def load_data(csv_path):
    """Load CSV and prepare normalized arrays."""
    df = pd.read_csv(csv_path)
    N = len(df)

    X_input = np.zeros((N, N_INPUT))
    X_input[:, 0] = df["vasopressor_primary"].map(VASO_MAP).fillna(0).values / INPUTS["vaso"]["hi"]
    X_input[:, 1] = df["fluid_vol_ml_per_kg"].values / INPUTS["fluid_vol"]["hi"]
    X_input[:, 2] = df["mechanical_support"].map(MECH_MAP).fillna(0).values / INPUTS["mech"]["hi"]
    X_input[:, 3] = (df["age"].values - 18) / 81.0

    X_init = np.zeros((N, N_TARGET))
    col_map = {"map": "map_0h", "hr": "hr_0h", "lactate": "lactate_0h",
               "creatinine": "creatinine", "wbc": "wbc", "procal": "procalcitonin"}
    for i, k in enumerate(TARGET_NAMES):
        t = TARGETS[k]
        if k in col_map:
            X_init[:, i] = (df[col_map[k]].values - t["lo"]) / (t["hi"] - t["lo"])
        else:
            X_init[:, i] = 0.5
    X_init = np.clip(X_init, 0, 1)

    Y_targets = np.zeros((N, len(OBSERVABLE_TS), len(FAST_TARGET_STEPS)))
    ts_cols = {
        "map": ["map_6h", "map_12h", "map_24h"],
        "hr": ["hr_6h", "hr_12h", "hr_24h"],
        "lactate": ["lactate_6h", "lactate_12h", "lactate_24h"],
    }
    for oi, obs_key in enumerate(OBSERVABLE_TS):
        t = TARGETS[obs_key]
        for ti, col in enumerate(ts_cols[obs_key]):
            Y_targets[:, oi, ti] = (df[col].values - t["lo"]) / (t["hi"] - t["lo"])
    Y_targets = np.clip(Y_targets, 0, 1)

    return X_input, X_init, Y_targets, df


# --- Fitting State ---
fit_state = {
    "status": "idle", "progress": 0, "loss_history": [],
    "current_loss": None, "best_loss": None, "elapsed": 0,
    "result": None, "n_evals": 0, "phase": "",
}

_eval_count = 0
_start_time = 0
_best_loss_seen = float("inf")


def _tracked_loss(flat_params, X_input, X_init, Y_targets):
    """Loss with progress tracking."""
    global _eval_count, _best_loss_seen

    traj = fast_forward_batch(flat_params, X_input, X_init)
    pred = traj[:, OBSERVABLE_TS_IDX, :][:, :, FAST_TARGET_STEPS]
    mse = float(np.mean((pred - Y_targets) ** 2))

    # Regularization: L2 on weights + L1 on tipping magnitudes (encourage sparsity)
    W = flat_params[:N_WEIGHTS]
    reg_w = 0.001 * float(np.mean(W ** 2))
    # Extract tipping magnitudes
    tip_start = N_WEIGHTS + N_TARGET + N_TARGET + N_TARGET * K_TIPS + N_TARGET * K_TIPS
    tip_mags = flat_params[tip_start:tip_start + N_TARGET * K_TIPS]
    reg_tip = 0.003 * float(np.mean(np.abs(tip_mags)))  # L1 sparsity on tipping

    loss = mse + reg_w + reg_tip

    _eval_count += 1
    if loss < _best_loss_seen:
        _best_loss_seen = loss

    if _eval_count % 3 == 0:
        elapsed = time.time() - _start_time
        estimated_iters = _eval_count / (N_PARAMS + 1)
        fit_state["n_evals"] = _eval_count
        fit_state["current_loss"] = round(loss, 6)
        fit_state["best_loss"] = round(_best_loss_seen, 6)
        fit_state["elapsed"] = round(elapsed, 1)
        fit_state["progress"] = min(95, int(estimated_iters * 0.6))
        fit_state["phase"] = "eval %d (~iter %d)" % (_eval_count, int(estimated_iters))

    if _eval_count % 40 == 0:
        fit_state["loss_history"].append(round(_best_loss_seen, 6))

    return loss


def _plain_loss(flat_params, X_input, X_init, Y_targets):
    """Loss without tracking."""
    traj = fast_forward_batch(flat_params, X_input, X_init)
    pred = traj[:, OBSERVABLE_TS_IDX, :][:, :, FAST_TARGET_STEPS]
    mse = float(np.mean((pred - Y_targets) ** 2))
    W = flat_params[:N_WEIGHTS]
    tip_start = N_WEIGHTS + N_TARGET + N_TARGET + N_TARGET * K_TIPS + N_TARGET * K_TIPS
    tip_mags = flat_params[tip_start:tip_start + N_TARGET * K_TIPS]
    return mse + 0.001 * float(np.mean(W ** 2)) + 0.003 * float(np.mean(np.abs(tip_mags)))


def run_fit(csv_path, method="L-BFGS-B", max_iter=150):
    """Main fitting entry point."""
    global _eval_count, _start_time, _best_loss_seen

    fit_state.clear()
    fit_state.update({
        "status": "running", "progress": 0, "loss_history": [],
        "current_loss": None, "best_loss": None, "elapsed": 0,
        "result": None, "n_evals": 0, "phase": "loading data",
    })
    _eval_count = 0
    _best_loss_seen = float("inf")

    try:
        X_input, X_init, Y_targets, df = load_data(csv_path)
        n_patients = X_input.shape[0]
        fit_state["phase"] = "testing seeds"

        best_params = None
        best_init_loss = float("inf")
        for seed in range(5):
            p0 = random_params(seed=seed * 7 + 13)
            l0 = _plain_loss(p0, X_input, X_init, Y_targets)
            fit_state["phase"] = "seed %d/5 (loss=%.4f)" % (seed + 1, l0)
            fit_state["current_loss"] = round(l0, 6)
            if l0 < best_init_loss:
                best_init_loss = l0
                best_params = p0

        _best_loss_seen = best_init_loss
        fit_state["loss_history"].append(round(best_init_loss, 6))
        fit_state["best_loss"] = round(best_init_loss, 6)
        _start_time = time.time()
        bounds = get_bounds()

        # Phase 1: Fast fit on subsample (200 patients)
        fit_state["phase"] = "phase 1: fast fit (200 patients)"
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(n_patients, min(200, n_patients), replace=False)
        X_in_sub = X_input[sub_idx]
        X_init_sub = X_init[sub_idx]
        Y_sub = Y_targets[sub_idx]

        result1 = minimize(
            _tracked_loss, best_params,
            args=(X_in_sub, X_init_sub, Y_sub),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-7, "disp": False},
        )

        # Phase 2: Refine on full data (shorter)
        fit_state["phase"] = "phase 2: refine (all patients)"
        result2 = minimize(
            _tracked_loss, result1.x,
            args=(X_input, X_init, Y_targets),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max(max_iter // 4, 10), "ftol": 1e-8, "disp": False},
        )

        fitted_params = result2.x
        final_loss = float(result2.fun)
        elapsed = time.time() - _start_time

        fit_state["loss_history"].append(round(final_loss, 6))
        fit_state["phase"] = "analyzing"

        # Post-process
        pruned_params, connections = prune_weights(fitted_params, threshold=0.12)
        hidden_analysis = analyze_hidden(fitted_params, threshold=0.12)
        tipping_analysis = analyze_tipping_points(fitted_params, mag_threshold=0.03)
        gdmt_json = to_gdmt_json(fitted_params, connections, hidden_analysis, tipping_analysis)

        # Per-variable errors
        traj = fast_forward_batch(fitted_params, X_input, X_init)
        var_errors = {}
        for oi, obs_key in enumerate(OBSERVABLE_TS):
            pred = traj[:, OBSERVABLE_TS_IDX[oi], :][:, FAST_TARGET_STEPS]
            actual = Y_targets[:, oi, :]
            t_def = TARGETS[obs_key]
            pred_real = pred * (t_def["hi"] - t_def["lo"]) + t_def["lo"]
            actual_real = actual * (t_def["hi"] - t_def["lo"]) + t_def["lo"]
            rmse = float(np.sqrt(np.mean((pred_real - actual_real) ** 2)))
            mae = float(np.mean(np.abs(pred_real - actual_real)))
            var_errors[obs_key] = {"rmse": round(rmse, 2), "mae": round(mae, 2), "unit": t_def.get("unit", "")}

        fit_result = {
            "status": "done",
            "n_patients": n_patients,
            "n_params": N_PARAMS,
            "n_tipping_params": N_TIP_PARAMS,
            "initial_loss": round(float(best_init_loss), 6),
            "final_loss": round(final_loss, 6),
            "improvement": round(float(best_init_loss - final_loss), 6),
            "improvement_pct": round((1 - final_loss / max(best_init_loss, 1e-9)) * 100, 1),
            "elapsed_s": round(elapsed, 1),
            "n_evals_total": _eval_count,
            "n_iterations": _eval_count // max(N_PARAMS, 1),
            "n_connections_total": N_SOURCE * N_TARGET,
            "n_connections_pruned": len(connections),
            "connections": connections[:30],
            "hidden_analysis": hidden_analysis,
            "tipping_points": tipping_analysis,
            "n_active_tipping": len(tipping_analysis),
            "var_errors": var_errors,
            "loss_history": fit_state["loss_history"],
            "gdmt_json": gdmt_json,
            "_fitted_params": fitted_params.tolist(),
        }

        fit_state["status"] = "done"
        fit_state["progress"] = 100
        fit_state["result"] = fit_result
        fit_state["phase"] = "complete"

        return fit_result

    except Exception as e:
        import traceback
        traceback.print_exc()
        fit_state["status"] = "error"
        fit_state["phase"] = "error: " + str(e)
        fit_state["result"] = {"error": str(e)}
        raise
