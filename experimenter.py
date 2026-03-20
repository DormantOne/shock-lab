"""
Experimenter — LLM Oracle That Tests Interventions on Fitted Model
===================================================================
1. User asks clinical question
2. Mock LLM parses patient profile + generates intervention scenarios
3. Each scenario runs through fitted reservoir model
4. Trajectories compared, scored, ranked
5. Best approach recommended with reasoning
"""

from reservoir_core import (
    INPUTS, TARGETS, INPUT_NAMES, TARGET_NAMES, forward_single, OBSERVABLE_TS,
)

# --- Scoring ---

def score_trajectory(traj, initial):
    """
    Score a predicted trajectory. Higher = better outcome.
    Considers MAP recovery, lactate clearance, HR normalization, renal preservation.
    """
    hours = traj["hours"]
    n = len(hours)
    # Get values at key timepoints
    map_final = traj["map"][-1]
    map_6h = traj["map"][min(6, n - 1)]
    hr_final = traj["hr"][-1]
    lac_init = initial.get("lactate", traj["lactate"][0])
    lac_final = traj["lactate"][-1]
    cr_init = initial.get("creatinine", traj["creatinine"][0])
    cr_final = traj["creatinine"][-1]

    score = 0

    # MAP recovery (above 65 is critical threshold)
    if map_final >= 65:
        score += 25 + min(15, (map_final - 65) * 0.5)
    else:
        score += max(0, (map_final - 40) * 1.0)

    # Early MAP response (at 6h)
    if map_6h >= 60:
        score += 10

    # Lactate clearance
    if lac_init > 0:
        clearance = (lac_init - lac_final) / lac_init
        score += clearance * 25  # up to 25 points for full clearance

    # HR normalization (ideal 60-100)
    hr_dev = abs(hr_final - 80)
    score += max(0, 15 - hr_dev * 0.3)

    # Renal preservation (creatinine shouldn't spike)
    cr_rise = max(0, cr_final - cr_init)
    score -= cr_rise * 10

    return round(score, 1)


# --- Mock LLM Scenario Generation ---

SCENARIO_TEMPLATES = {
    "septic": [
        {
            "name": "Aggressive Norepinephrine",
            "description": "High-dose norepinephrine with standard 30ml/kg fluid resuscitation per Surviving Sepsis guidelines",
            "inputs": {"vaso": 1, "fluid_vol": 30, "mech": 0},
            "color": "#ff6644",
        },
        {
            "name": "Norepinephrine + Vasopressin",
            "description": "Moderate norepi with vasopressin as second-line, conservative fluids to avoid overload",
            "inputs": {"vaso": 2, "fluid_vol": 25, "mech": 0},
            "color": "#a78bfa",
        },
        {
            "name": "Fluids First, Pressors Later",
            "description": "Aggressive fluid resuscitation (40ml/kg) before vasopressors — volume-first strategy",
            "inputs": {"vaso": 0, "fluid_vol": 40, "mech": 0},
            "color": "#60a5fa",
        },
        {
            "name": "Early Norepinephrine + Moderate Fluid",
            "description": "Early vasopressor initiation with moderate fluids — trending evidence approach",
            "inputs": {"vaso": 1, "fluid_vol": 20, "mech": 0},
            "color": "#4ade80",
        },
    ],
    "cardiogenic": [
        {
            "name": "Dobutamine + Conservative Fluids",
            "description": "Inotrope-first with minimal fluids to avoid worsening congestion",
            "inputs": {"vaso": 4, "fluid_vol": 10, "mech": 0},
            "color": "#f472b6",
        },
        {
            "name": "Norepinephrine + IABP",
            "description": "Vasopressor support with mechanical unloading via balloon pump",
            "inputs": {"vaso": 1, "fluid_vol": 15, "mech": 1},
            "color": "#ff6644",
        },
        {
            "name": "Impella Support",
            "description": "Aggressive mechanical support with percutaneous ventricular assist",
            "inputs": {"vaso": 1, "fluid_vol": 10, "mech": 2},
            "color": "#818cf8",
        },
        {
            "name": "Dopamine + Moderate Fluids",
            "description": "Traditional dopamine approach with moderate volume support",
            "inputs": {"vaso": 3, "fluid_vol": 20, "mech": 0},
            "color": "#fbbf24",
        },
    ],
    "hypovolemic": [
        {
            "name": "Massive Fluid Resuscitation",
            "description": "Aggressive volume replacement (45ml/kg) — address the root cause",
            "inputs": {"vaso": 0, "fluid_vol": 45, "mech": 0},
            "color": "#60a5fa",
        },
        {
            "name": "Balanced Fluids + Low-Dose Pressor",
            "description": "Moderate fluids with norepinephrine bridge to maintain perfusion during resuscitation",
            "inputs": {"vaso": 1, "fluid_vol": 35, "mech": 0},
            "color": "#4ade80",
        },
        {
            "name": "Conservative Volume + Vasopressin",
            "description": "More conservative fluid approach with vasopressin to maintain tone",
            "inputs": {"vaso": 2, "fluid_vol": 25, "mech": 0},
            "color": "#a78bfa",
        },
        {
            "name": "Phenylephrine Bolus + Fluids",
            "description": "Pure alpha agonist for immediate pressure support while resuscitating",
            "inputs": {"vaso": 5, "fluid_vol": 35, "mech": 0},
            "color": "#fb923c",
        },
    ],
    "general": [
        {
            "name": "Norepinephrine Standard",
            "description": "First-line vasopressor with guideline-directed fluid resuscitation",
            "inputs": {"vaso": 1, "fluid_vol": 30, "mech": 0},
            "color": "#ff6644",
        },
        {
            "name": "Conservative Approach",
            "description": "Minimal interventions — moderate fluids only, no vasopressors initially",
            "inputs": {"vaso": 0, "fluid_vol": 25, "mech": 0},
            "color": "#94a3b8",
        },
        {
            "name": "Aggressive Multi-Modal",
            "description": "Vasopressor + aggressive fluids + mechanical support",
            "inputs": {"vaso": 1, "fluid_vol": 35, "mech": 1},
            "color": "#818cf8",
        },
        {
            "name": "Vasopressin-Led",
            "description": "Vasopressin as primary with moderate resuscitation",
            "inputs": {"vaso": 2, "fluid_vol": 28, "mech": 0},
            "color": "#a78bfa",
        },
    ],
}


def _extract_number(text, start, max_chars=8):
    """Extract a number from text starting at position. Handles trailing periods/commas."""
    num = ""
    has_dot = False
    for ch in text[start:start + max_chars]:
        if ch.isdigit():
            num += ch
        elif ch == "." and not has_dot:
            # Only count as decimal if next char is a digit
            next_pos = start + len(num) + 1
            if next_pos < len(text) and text[next_pos].isdigit():
                num += ch
                has_dot = True
            else:
                break  # trailing period — end of sentence
        elif num:
            break
    if num:
        try:
            return float(num)
        except ValueError:
            return None
    return None


def _parse_patient_profile(question):
    """Extract patient profile from question text. Simple keyword matching."""
    q = question.lower()
    profile = {}

    search_terms = {
        "map": ["map ", "map=", "map:"],
        "hr": ["hr ", "hr=", "hr:", "heart rate "],
        "lactate": ["lactate ", "lactate=", "lactate:", "lac "],
        "creatinine": ["creatinine ", "cr ", "cr=", "creat "],
        "age": ["age ", "age=", "age:"],
    }

    for key, markers in search_terms.items():
        for marker in markers:
            idx = q.find(marker)
            if idx >= 0:
                val = _extract_number(q, idx + len(marker))
                if val is not None:
                    profile[key] = val
                    break

    # Defaults for missing values (typical shock patient)
    profile.setdefault("map", 55)
    profile.setdefault("hr", 110)
    profile.setdefault("lactate", 4.5)
    profile.setdefault("creatinine", 1.3)
    profile.setdefault("wbc", 15)
    profile.setdefault("procal", 5.0)
    profile.setdefault("age", 65)

    return profile


def _detect_shock_type(question):
    """Detect shock type from question text."""
    q = question.lower()
    if "septic" in q or "sepsis" in q or "infection" in q:
        return "septic"
    if "cardiogenic" in q or "cardiac" in q or "mi " in q or "heart failure" in q:
        return "cardiogenic"
    if "hypovolemic" in q or "hemorrhag" in q or "bleed" in q or "trauma" in q:
        return "hypovolemic"
    return "general"


def run_experiment_with_profile(question, profile, shock_type, fitted_params):
    """
    Run experiment with a pre-confirmed profile (from LLM parse step).
    Profile comes with values already validated by the user.
    """
    templates = SCENARIO_TEMPLATES.get(shock_type, SCENARIO_TEMPLATES.get("general", SCENARIO_TEMPLATES["unknown"] if "unknown" in SCENARIO_TEMPLATES else list(SCENARIO_TEMPLATES.values())[0]))
    if shock_type not in SCENARIO_TEMPLATES:
        templates = SCENARIO_TEMPLATES.get("general", list(SCENARIO_TEMPLATES.values())[0])

    # Build initial conditions for target reservoirs
    init_vals = {
        "map": float(profile.get("map", 55)),
        "hr": float(profile.get("hr", 110)),
        "lactate": float(profile.get("lactate", 4.5)),
        "creatinine": float(profile.get("creatinine", 1.3)),
        "wbc": float(profile.get("wbc", 15)),
        "procal": float(profile.get("procal", 5)),
        "h0": 0.5,
        "h1": 0.5,
        "h2": 0.5,
    }

    age_norm = (float(profile.get("age", 65)) - 18) / 81.0

    scenarios = []
    for tmpl in templates:
        input_vals = {
            "vaso": tmpl["inputs"]["vaso"],
            "fluid_vol": tmpl["inputs"]["fluid_vol"],
            "mech": tmpl["inputs"]["mech"],
            "age_n": age_norm,
        }

        traj = forward_single(fitted_params, input_vals, init_vals)
        sc = score_trajectory(traj, profile)

        key_preds = {
            "map_6h": round(traj["map"][6], 1) if len(traj["map"]) > 6 else traj["map"][-1],
            "map_24h": round(traj["map"][-1], 1),
            "hr_24h": round(traj["hr"][-1], 1),
            "lactate_24h": round(traj["lactate"][-1], 2),
            "creatinine_24h": round(traj["creatinine"][-1], 2),
        }

        scenarios.append({
            "name": tmpl["name"],
            "description": tmpl["description"],
            "color": tmpl["color"],
            "inputs": tmpl["inputs"],
            "trajectory": {
                "hours": [round(h, 1) for h in traj["hours"]],
                "map": [round(v, 1) for v in traj["map"]],
                "hr": [round(v, 1) for v in traj["hr"]],
                "lactate": [round(v, 2) for v in traj["lactate"]],
                "creatinine": [round(v, 2) for v in traj["creatinine"]],
            },
            "key_predictions": key_preds,
            "score": sc,
        })

    scenarios.sort(key=lambda s: s["score"], reverse=True)
    for i, s in enumerate(scenarios):
        s["rank"] = i + 1
        s["is_best"] = i == 0

    best = scenarios[0]
    worst = scenarios[-1]
    rec = _generate_recommendation(best, worst, profile, shock_type)

    return {
        "question": question,
        "shock_type": shock_type,
        "patient_profile": profile,
        "scenarios": scenarios,
        "recommendation": rec,
    }


def run_experiment(question, fitted_params):
    """
    Main entry point for the Oracle.
    1. Parse question → patient profile
    2. Detect shock type → select scenario templates
    3. Run each scenario through fitted model
    4. Score, rank, recommend
    """
    profile = _parse_patient_profile(question)
    shock_type = _detect_shock_type(question)
    templates = SCENARIO_TEMPLATES.get(shock_type, SCENARIO_TEMPLATES["general"])

    # Build initial conditions for target reservoirs
    init_vals = {
        "map": profile["map"],
        "hr": profile["hr"],
        "lactate": profile["lactate"],
        "creatinine": profile["creatinine"],
        "wbc": profile.get("wbc", 15),
        "procal": profile.get("procal", 5),
        "h0": 0.5,
        "h1": 0.5,
        "h2": 0.5,
    }

    age_norm = (profile.get("age", 65) - 18) / 81.0

    scenarios = []
    for tmpl in templates:
        # Build input values
        input_vals = {
            "vaso": tmpl["inputs"]["vaso"],
            "fluid_vol": tmpl["inputs"]["fluid_vol"],
            "mech": tmpl["inputs"]["mech"],
            "age_n": age_norm,
        }

        # Run forward simulation
        traj = forward_single(fitted_params, input_vals, init_vals)

        # Score
        sc = score_trajectory(traj, profile)

        # Extract key predicted values
        key_preds = {
            "map_6h": round(traj["map"][6], 1) if len(traj["map"]) > 6 else traj["map"][-1],
            "map_24h": round(traj["map"][-1], 1),
            "hr_24h": round(traj["hr"][-1], 1),
            "lactate_24h": round(traj["lactate"][-1], 2),
            "creatinine_24h": round(traj["creatinine"][-1], 2),
        }

        scenarios.append({
            "name": tmpl["name"],
            "description": tmpl["description"],
            "color": tmpl["color"],
            "inputs": tmpl["inputs"],
            "trajectory": {
                "hours": [round(h, 1) for h in traj["hours"]],
                "map": [round(v, 1) for v in traj["map"]],
                "hr": [round(v, 1) for v in traj["hr"]],
                "lactate": [round(v, 2) for v in traj["lactate"]],
                "creatinine": [round(v, 2) for v in traj["creatinine"]],
            },
            "key_predictions": key_preds,
            "score": sc,
        })

    # Rank by score
    scenarios.sort(key=lambda s: s["score"], reverse=True)
    for i, s in enumerate(scenarios):
        s["rank"] = i + 1
        s["is_best"] = i == 0

    # Generate recommendation
    best = scenarios[0]
    worst = scenarios[-1]
    rec = _generate_recommendation(best, worst, profile, shock_type)

    return {
        "question": question,
        "shock_type": shock_type,
        "patient_profile": profile,
        "scenarios": scenarios,
        "recommendation": rec,
    }


def _generate_recommendation(best, worst, profile, shock_type):
    """Generate clinical reasoning for the recommendation."""
    parts = []
    parts.append(f"For this {shock_type} shock presentation (MAP {profile['map']}, "
                  f"lactate {profile['lactate']}, HR {profile['hr']}):")
    parts.append("")
    parts.append(f"**{best['name']}** achieves the best predicted outcome (score {best['score']}).")

    kp = best["key_predictions"]
    if kp["map_24h"] >= 65:
        parts.append(f"MAP recovers to {kp['map_24h']} mmHg by 24h — above critical threshold.")
    else:
        parts.append(f"MAP reaches {kp['map_24h']} mmHg by 24h — still below target, consider escalation.")

    lac_clear = profile["lactate"] - kp["lactate_24h"]
    if lac_clear > 0:
        parts.append(f"Lactate clears by {round(lac_clear, 1)} mmol/L ({round(lac_clear/profile['lactate']*100)}% clearance).")
    else:
        parts.append(f"Lactate did not clear — suggests inadequate perfusion restoration.")

    cr_rise = kp["creatinine_24h"] - profile["creatinine"]
    if cr_rise > 0.5:
        parts.append(f"Caution: predicted creatinine rise of {round(cr_rise, 2)} mg/dL — monitor renal function.")
    elif cr_rise < 0:
        parts.append(f"Renal function preserved (creatinine stable/improving).")

    parts.append("")
    parts.append(f"Avoid: **{worst['name']}** (score {worst['score']}) — predicted poorest trajectory "
                  f"with MAP {worst['key_predictions']['map_24h']} and lactate {worst['key_predictions']['lactate_24h']} at 24h.")

    return "\n".join(parts)
