"""
SHOCK LAB v2 — Reservoir Net Arena
====================================
Flask app orchestrating: data fitting → network visualization → LLM oracle experiments
"""

from flask import Flask, render_template, request, jsonify
import threading
import os
import json

from reservoir_core import (
    INPUTS, TARGETS, INPUT_NAMES, TARGET_NAMES, ALL_NAMES,
    N_INPUT, N_TARGET, N_SOURCE, N_PARAMS,
)
from fit_pipeline import run_fit, fit_state
from experimenter import run_experiment_with_profile
from llm_client import parse_clinical_text

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "shock_cases.csv")

# Global: stores fitted params after fitting
fitted_params_global = None
fit_thread = None


@app.route("/")
def index():
    return render_template(
        "index.html",
        inputs=INPUTS,
        targets=TARGETS,
        input_names=INPUT_NAMES,
        target_names=TARGET_NAMES,
        n_params=N_PARAMS,
        n_source=N_SOURCE,
        n_target=N_TARGET,
    )


@app.route("/api/fit", methods=["POST"])
def api_fit():
    """Start model fitting in background thread."""
    global fit_thread, fitted_params_global

    if fit_state.get("status") == "running":
        return jsonify({"error": "Fitting already in progress"}), 400

    def _do_fit():
        global fitted_params_global
        try:
            result = run_fit(DATA_PATH, method="L-BFGS-B", max_iter=250)
            # Extract fitted params from result
            if "_fitted_params" in result:
                import numpy as np
                fitted_params_global = np.array(result["_fitted_params"])
        except Exception as e:
            fit_state["status"] = "error"
            fit_state["result"] = {"error": str(e)}

    fit_thread = threading.Thread(target=_do_fit, daemon=True)
    fit_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/fit/status")
def api_fit_status():
    """Poll fitting progress."""
    return jsonify({
        "status": fit_state.get("status", "idle"),
        "progress": fit_state.get("progress", 0),
        "current_loss": fit_state.get("current_loss"),
        "best_loss": fit_state.get("best_loss"),
        "elapsed": round(fit_state.get("elapsed", 0), 1),
        "n_evals": fit_state.get("n_evals", 0),
        "phase": fit_state.get("phase", ""),
        "loss_history": fit_state.get("loss_history", [])[-50:],
    })


@app.route("/api/fit/result")
def api_fit_result():
    """Get full fitting results."""
    if fit_state.get("status") != "done":
        return jsonify({"error": "Fitting not complete", "status": fit_state.get("status")}), 400
    return jsonify(fit_state.get("result", {}))


@app.route("/api/parse", methods=["POST"])
def api_parse():
    """Step 1: Parse clinical text into structured profile via LLM."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = parse_clinical_text(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiment", methods=["POST"])
def api_experiment():
    """Step 2: Run experiment with confirmed profile."""
    global fitted_params_global

    if fitted_params_global is None:
        return jsonify({"error": "Model not fitted yet. Run fitting first."}), 400

    data = request.get_json()
    profile = data.get("profile", {})
    shock_type = data.get("shock_type", "unknown")
    question = data.get("question", "")

    if not profile:
        return jsonify({"error": "No profile provided"}), 400

    try:
        result = run_experiment_with_profile(question, profile, shock_type, fitted_params_global)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/json")
def api_model_json():
    """Get fitted model as GDMT-compatible JSON."""
    if fit_state.get("status") != "done":
        return jsonify({"error": "Model not fitted"}), 400
    result = fit_state.get("result", {})
    return jsonify(result.get("gdmt_json", {}))


EXAMPLE_QUESTIONS = [
    "I have a septic shock patient with MAP 50, HR 130, lactate 7, creatinine 1.5. What's the best approach?",
    "Cardiogenic shock post-MI, MAP 55, HR 105, lactate 4, troponin elevated. Best intervention?",
    "Hypovolemic shock from GI bleed, MAP 48, HR 140, lactate 8. How should I resuscitate?",
    "Elderly patient age 78, MAP 52, HR 95, lactate 5.5, creatinine 2.1. Best vasopressor?",
    "Septic shock patient not responding to fluids, MAP still 52 after 30ml/kg. Next step?",
]


@app.route("/api/examples")
def api_examples():
    return jsonify(EXAMPLE_QUESTIONS)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SHOCK LAB v2 — Reservoir Net Arena")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000, threaded=True)
