"""
Reservoir Core v3 — With Learned Multi-Channel Tipping Points
==============================================================
Each target reservoir can have K tipping points (default K=3).
Each tipping point has 3 learned parameters:
  - threshold (0-1 normalized): WHERE the edge is
  - steepness (signed): HOW sharp + WHICH direction
      positive steepness → activates when state ABOVE threshold
      negative steepness → activates when state BELOW threshold
  - magnitude (signed): WHAT HAPPENS when activated
      positive → pushes state UP (additional positive force)
      negative → pushes state DOWN (destabilizing collapse)

The combinations cover all physiological tipping phenomena:
  - steepness<0, magnitude<0: "below threshold, collapse" (autoregulatory failure)
  - steepness>0, magnitude>0: "above threshold, cascade up" (lactate spiral)
  - steepness>0, magnitude<0: "above threshold, suppress" (negative feedback)
  - steepness<0, magnitude>0: "below threshold, compensate" (rescue reflex)

Multiple tipping points per reservoir create multi-channel edges:
  e.g., MAP might learn thresholds at ~65 (autoregulation), ~50 (organ failure),
        and ~80 (no effect → pruned)
"""

import numpy as np
from collections import OrderedDict

# --- Reservoir Definitions ---

INPUTS = OrderedDict([
    ("vaso",      {"label": "Vasopressor",  "lo": 0, "hi": 5,  "color": "#ff6644"}),
    ("fluid_vol", {"label": "Fluid Volume", "lo": 0, "hi": 50, "color": "#60a5fa"}),
    ("mech",      {"label": "Mech Support", "lo": 0, "hi": 3,  "color": "#c4b5fd"}),
    ("age_n",     {"label": "Age",          "lo": 0, "hi": 1,  "color": "#94a3b8"}),
])

TARGETS = OrderedDict([
    ("map",       {"label": "MAP",           "unit": "mmHg",  "lo": 40,  "hi": 200, "color": "#fb923c", "group": "phys"}),
    ("hr",        {"label": "HR",            "unit": "bpm",   "lo": 30,  "hi": 180, "color": "#f472b6", "group": "phys"}),
    ("lactate",   {"label": "Lactate",       "unit": "mmol/L","lo": 0.3, "hi": 15,  "color": "#fbbf24", "group": "phys"}),
    ("creatinine",{"label": "Creatinine",    "unit": "mg/dL", "lo": 0.3, "hi": 6,   "color": "#fca5a5", "group": "lab"}),
    ("wbc",       {"label": "WBC",           "unit": "K/uL",  "lo": 0.5, "hi": 40,  "color": "#86efac", "group": "lab"}),
    ("procal",    {"label": "Procalcitonin", "unit": "ng/mL", "lo": 0,   "hi": 25,  "color": "#fdba74", "group": "lab"}),
    ("h0",        {"label": "Hidden-0",      "unit": "",      "lo": 0,   "hi": 1,   "color": "#818cf8", "group": "hidden"}),
    ("h1",        {"label": "Hidden-1",      "unit": "",      "lo": 0,   "hi": 1,   "color": "#a78bfa", "group": "hidden"}),
    ("h2",        {"label": "Hidden-2",      "unit": "",      "lo": 0,   "hi": 1,   "color": "#c084fc", "group": "hidden"}),
])

INPUT_NAMES = list(INPUTS.keys())
TARGET_NAMES = list(TARGETS.keys())
ALL_NAMES = INPUT_NAMES + TARGET_NAMES
N_INPUT = len(INPUT_NAMES)
N_TARGET = len(TARGET_NAMES)
N_SOURCE = N_INPUT + N_TARGET
N_HIDDEN = sum(1 for t in TARGETS.values() if t.get("group") == "hidden")
OBSERVABLE_TS = ["map", "hr", "lactate"]
OBSERVABLE_TS_IDX = [TARGET_NAMES.index(k) for k in OBSERVABLE_TS]

# Tipping point config
K_TIPS = 3  # tipping points per reservoir

# Time config
DT = 0.5
N_STEPS = 48

# --- Parameter Layout ---
# weights:     N_SOURCE * N_TARGET
# eq:          N_TARGET
# spring:      N_TARGET
# tip_thresh:  N_TARGET * K_TIPS
# tip_steep:   N_TARGET * K_TIPS
# tip_mag:     N_TARGET * K_TIPS
N_WEIGHTS = N_SOURCE * N_TARGET
N_TIP_PARAMS = N_TARGET * K_TIPS * 3  # threshold + steepness + magnitude per tip
N_PARAMS = N_WEIGHTS + N_TARGET + N_TARGET + N_TIP_PARAMS


def normalize(val, lo, hi):
    return (val - lo) / (hi - lo) if hi > lo else 0.0


def denormalize(nval, lo, hi):
    return nval * (hi - lo) + lo


def pack_params(weights, eq, spring, tip_thresh, tip_steep, tip_mag):
    """Flatten all parameters into 1D array."""
    return np.concatenate([
        weights.ravel(), eq, spring,
        tip_thresh.ravel(), tip_steep.ravel(), tip_mag.ravel()
    ])


def unpack_params(flat):
    """Unpack 1D array into components."""
    idx = 0
    w = flat[idx:idx + N_WEIGHTS].reshape(N_SOURCE, N_TARGET); idx += N_WEIGHTS
    eq = flat[idx:idx + N_TARGET]; idx += N_TARGET
    spring = flat[idx:idx + N_TARGET]; idx += N_TARGET
    nk = N_TARGET * K_TIPS
    thresh = flat[idx:idx + nk].reshape(N_TARGET, K_TIPS); idx += nk
    steep = flat[idx:idx + nk].reshape(N_TARGET, K_TIPS); idx += nk
    mag = flat[idx:idx + nk].reshape(N_TARGET, K_TIPS); idx += nk
    return w, eq, spring, thresh, steep, mag


def get_bounds():
    """Parameter bounds for optimizer."""
    bounds = []
    bounds += [(-3.0, 3.0)] * N_WEIGHTS      # weights
    bounds += [(0.0, 1.0)] * N_TARGET          # eq
    bounds += [(0.005, 0.4)] * N_TARGET        # spring
    bounds += [(0.05, 0.95)] * (N_TARGET * K_TIPS)  # thresholds
    bounds += [(-25.0, 25.0)] * (N_TARGET * K_TIPS)  # steepness (signed!)
    bounds += [(-0.5, 0.5)] * (N_TARGET * K_TIPS)    # magnitude (signed!)
    return bounds


def random_params(seed=42):
    """Random initial parameters."""
    rng = np.random.RandomState(seed)
    weights = rng.randn(N_SOURCE, N_TARGET) * 0.1
    eq = rng.uniform(0.2, 0.8, N_TARGET)
    spring = rng.uniform(0.05, 0.2, N_TARGET)
    # Start tipping points spread across the range with small magnitude
    # so they don't dominate early — optimizer activates them if needed
    thresh = rng.uniform(0.2, 0.8, (N_TARGET, K_TIPS))
    steep = rng.uniform(-5, 5, (N_TARGET, K_TIPS))
    mag = rng.uniform(-0.05, 0.05, (N_TARGET, K_TIPS))  # start small!
    return pack_params(weights, eq, spring, thresh, steep, mag)


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def forward_batch(flat_params, X_input, X_init, n_steps=N_STEPS, dt=DT):
    """
    Vectorized forward simulation with tipping points.

    Physics per timestep:
      1. eff_eq = base_eq + Σ(source × weight)           [linear spring]
      2. tip_force = Σ_k magnitude_k × σ(steep_k × (state - thresh_k))  [nonlinear tips]
      3. state += (eff_eq - state) × spring × dt + tip_force × dt
      4. clamp [0, 1]
    """
    W, eq, spring, thresh, steep, mag = unpack_params(flat_params)
    N = X_input.shape[0]

    traj = np.zeros((N, N_TARGET, n_steps + 1))
    state = X_init.copy()
    traj[:, :, 0] = state

    sd = spring[None, :] * dt

    for step in range(n_steps):
        # Linear: spring toward shifted equilibrium
        full = np.concatenate([X_input, state], axis=1)  # (N, N_SOURCE)
        eff_eq = eq[None, :] + full @ W  # (N, N_TARGET)
        linear_delta = (eff_eq - state) * sd

        # Nonlinear: tipping point forces
        # state is (N, N_TARGET), thresh is (N_TARGET, K_TIPS)
        # We need (N, N_TARGET, K_TIPS) for the sigmoid
        state_exp = state[:, :, None]  # (N, N_TARGET, 1)
        thresh_exp = thresh[None, :, :]  # (1, N_TARGET, K_TIPS)
        steep_exp = steep[None, :, :]
        mag_exp = mag[None, :, :]

        # σ(steepness × (state - threshold))
        sig_input = steep_exp * (state_exp - thresh_exp)  # (N, N_TARGET, K_TIPS)
        sig_val = _sigmoid(sig_input)  # (N, N_TARGET, K_TIPS)

        # Sum tipping forces across K channels
        tip_force = np.sum(mag_exp * sig_val, axis=2)  # (N, N_TARGET)

        # Update
        state = state + linear_delta + tip_force * dt
        np.clip(state, 0, 1, out=state)
        traj[:, :, step + 1] = state

    return traj


def forward_single(flat_params, input_vals, init_vals, n_steps=24, dt=1.0):
    """Forward sim for a single patient in REAL space."""
    x_in = np.array([[normalize(input_vals.get(k, 0), INPUTS[k]["lo"], INPUTS[k]["hi"])
                       for k in INPUT_NAMES]])
    x_init = np.array([[normalize(init_vals.get(k, TARGETS[k].get("lo", 0)),
                                   TARGETS[k]["lo"], TARGETS[k]["hi"])
                         for k in TARGET_NAMES]])

    traj = forward_batch(flat_params, x_in, x_init, n_steps, dt)

    result = {}
    for i, k in enumerate(TARGET_NAMES):
        lo, hi = TARGETS[k]["lo"], TARGETS[k]["hi"]
        vals = [float(denormalize(traj[0, i, s], lo, hi)) for s in range(n_steps + 1)]
        result[k] = vals
    result["hours"] = [s * dt for s in range(n_steps + 1)]
    return result


def prune_weights(flat_params, threshold=0.12):
    """Zero out weak connections. Returns pruned params + connectivity info."""
    W, eq, spring, thresh, steep, mag = unpack_params(flat_params)
    mask = np.abs(W) >= threshold
    W_pruned = W * mask

    connections = []
    for si in range(N_SOURCE):
        for ti in range(N_TARGET):
            if mask[si, ti]:
                src = ALL_NAMES[si]
                tgt = TARGET_NAMES[ti]
                w = float(W[si, ti])
                connections.append({
                    "from": src, "to": tgt,
                    "weight": round(w, 4),
                    "type": "fill" if w > 0 else "leak",
                    "abs_weight": round(abs(w), 4),
                })

    connections.sort(key=lambda c: c["abs_weight"], reverse=True)
    pruned_params = pack_params(W_pruned, eq, spring, thresh, steep, mag)
    return pruned_params, connections


def analyze_tipping_points(flat_params, mag_threshold=0.03):
    """
    Analyze discovered tipping points.
    Returns list of active tipping points with real-unit thresholds.
    """
    W, eq, spring, thresh, steep, mag = unpack_params(flat_params)

    tips = []
    for ti in range(N_TARGET):
        tname = TARGET_NAMES[ti]
        tdef = TARGETS[tname]

        for k in range(K_TIPS):
            m = float(mag[ti, k])
            if abs(m) < mag_threshold:
                continue  # pruned — too weak to matter

            th = float(thresh[ti, k])
            st = float(steep[ti, k])

            # Convert threshold to real units
            th_real = denormalize(th, tdef["lo"], tdef["hi"])

            # Determine behavior
            if st >= 0:
                direction = "above"  # activates when state > threshold
            else:
                direction = "below"  # activates when state < threshold

            if m > 0:
                effect = "accelerate_up"
            else:
                effect = "accelerate_down"

            # Clinical interpretation
            interp = _interpret_tip(tname, direction, effect, th_real, abs(m), abs(st))

            tips.append({
                "reservoir": tname,
                "label": tdef["label"],
                "channel": k,
                "threshold_norm": round(th, 3),
                "threshold_real": round(float(th_real), 2),
                "unit": tdef.get("unit", ""),
                "steepness": round(st, 2),
                "magnitude": round(m, 4),
                "direction": direction,
                "effect": effect,
                "sharpness": "sharp" if abs(st) > 12 else "gradual" if abs(st) > 5 else "soft",
                "strength": "strong" if abs(m) > 0.15 else "moderate" if abs(m) > 0.07 else "weak",
                "interpretation": interp,
                "color": tdef.get("color", "#888"),
            })

    # Sort by magnitude (strongest first)
    tips.sort(key=lambda t: abs(t["magnitude"]), reverse=True)
    return tips


def _interpret_tip(name, direction, effect, threshold, mag, steep):
    """Generate human-readable interpretation of a tipping point."""
    label = TARGETS[name]["label"]
    unit = TARGETS[name].get("unit", "")

    sharp = "sharply" if steep > 12 else "gradually" if steep > 5 else "gently"

    if direction == "below" and effect == "accelerate_down":
        return f"{label} below {threshold:.1f} {unit}: {sharp} destabilizes — collapse/failure dynamics"
    elif direction == "above" and effect == "accelerate_up":
        return f"{label} above {threshold:.1f} {unit}: {sharp} self-reinforcing — cascade/spiral dynamics"
    elif direction == "below" and effect == "accelerate_up":
        return f"{label} below {threshold:.1f} {unit}: {sharp} compensatory rebound — protective response"
    elif direction == "above" and effect == "accelerate_down":
        return f"{label} above {threshold:.1f} {unit}: {sharp} negative feedback — ceiling/suppression"
    return f"{label} tipping at {threshold:.1f} {unit}"


def analyze_hidden(flat_params, threshold=0.12):
    """Analyze hidden reservoir connection patterns."""
    W, eq, spring, thresh_t, steep_t, mag_t = unpack_params(flat_params)
    hidden_analysis = []

    for ti, tname in enumerate(TARGET_NAMES):
        if TARGETS[tname].get("group") != "hidden":
            continue

        incoming = []
        for si in range(N_SOURCE):
            w = W[si, ti]
            if abs(w) >= threshold:
                incoming.append({"source": ALL_NAMES[si], "weight": round(float(w), 3)})
        incoming.sort(key=lambda x: abs(x["weight"]), reverse=True)

        hidden_source_idx = N_INPUT + ti
        outgoing = []
        for oti in range(N_TARGET):
            w = W[hidden_source_idx, oti]
            if abs(w) >= threshold and TARGET_NAMES[oti] != tname:
                outgoing.append({"target": TARGET_NAMES[oti], "weight": round(float(w), 3)})
        outgoing.sort(key=lambda x: abs(x["weight"]), reverse=True)

        in_keys = [x["source"] for x in incoming[:3]]
        out_keys = [x["target"] for x in outgoing[:3]]
        suggested = _suggest_name(in_keys, out_keys, incoming, outgoing)

        hidden_analysis.append({
            "reservoir": tname,
            "label": TARGETS[tname]["label"],
            "suggested_name": suggested,
            "eq": round(float(denormalize(eq[ti], 0, 1)), 3),
            "spring_rate": round(float(spring[ti]), 4),
            "top_incoming": incoming[:5],
            "top_outgoing": outgoing[:5],
        })

    return hidden_analysis


def _suggest_name(in_keys, out_keys, incoming, outgoing):
    """Heuristic name suggestion for hidden reservoirs."""
    all_connected = set(in_keys + out_keys)
    if {"procal", "wbc"} & set(in_keys) and {"map", "lactate"} & set(out_keys):
        return "Inflammatory Burden"
    if "vaso" in in_keys and {"map", "hr"} & set(out_keys):
        return "Vasomotor Tone"
    if {"creatinine", "lactate"} & set(out_keys) and any(w["weight"] > 0 for w in outgoing[:3]):
        return "Organ Stress"
    if "fluid_vol" in in_keys and "map" in out_keys:
        return "Volume Status"
    if {"hr", "map"} & set(in_keys) and {"lactate"} & set(out_keys):
        return "Perfusion State"
    if len(incoming) > 3 and len(outgoing) > 3:
        return "Integration Hub"
    if len(incoming) > len(outgoing):
        return "Convergence Node"
    return "Latent Factor"


def to_gdmt_json(flat_params, connections, hidden_analysis, tipping_analysis):
    """Export as GDMT-compatible JSON with tipping points."""
    W, eq, spring, thresh, steep, mag = unpack_params(flat_params)

    reservoirs = {}
    for k, d in INPUTS.items():
        reservoirs[k] = {
            "label": d["label"], "unit": "", "lo": d["lo"], "hi": d["hi"],
            "init": 0, "color": d["color"], "group": "input",
            "eq": 0, "spring_rate": 0, "noise": 0,
        }
    for i, (k, d) in enumerate(TARGETS.items()):
        name = d["label"]
        for ha in hidden_analysis:
            if ha["reservoir"] == k:
                name = ha["suggested_name"]
        reservoirs[k] = {
            "label": name, "unit": d.get("unit", ""), "lo": d["lo"], "hi": d["hi"],
            "init": round(float(denormalize(eq[i], d["lo"], d["hi"])), 2),
            "color": d["color"], "group": d.get("group", "phys"),
            "eq": round(float(denormalize(eq[i], d["lo"], d["hi"])), 2),
            "spring_rate": round(float(spring[i]), 4),
            "noise": 0,
        }

    conduits = []
    for c in connections:
        conduits.append({
            "from": c["from"], "to": c["to"],
            "type": c["type"], "weight": c["abs_weight"],
            "mechanism": f"Learned: {c['from']}→{c['to']} ({c['type']}, w={c['weight']})",
        })

    # Convert tipping points to GDMT format
    tipping_points = []
    for tip in tipping_analysis:
        tipping_points.append({
            "id": f"tip_{tip['reservoir']}_{tip['channel']}",
            "label": f"{tip['label']} @ {tip['threshold_real']}{tip['unit']}",
            "reservoir": tip["reservoir"],
            "threshold": tip["threshold_real"],
            "direction": tip["direction"],
            "effect": tip["effect"],
            "magnitude": tip["magnitude"],
            "steepness": tip["steepness"],
            "interpretation": tip["interpretation"],
        })

    return {
        "meta": {
            "name": "Shock Reservoir Model v3 (Tipping Points)",
            "version": "3.0",
            "description": "Spring-to-equilibrium with learned multi-channel tipping points.",
        },
        "reservoirs": reservoirs,
        "conduits": conduits,
        "tipping_points": tipping_points,
    }
