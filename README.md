# Shock Lab

**Experimental data-driven clinical decision support for shock management.**

A nonlinear reservoir computing model that learns hemodynamic dynamics from patient trajectories, discovers hidden physiological states and critical tipping points, and lets clinicians experiment with intervention strategies on a fitted digital twin.

> **Status:** Prototype on synthetic data. Not validated for clinical use.

---

## What It Does

```
  Clinical Question (plain language)
        │
        ▼
  ┌─────────────┐
  │  Local LLM  │  Ollama (gpt-oss:20b)
  │  Parser     │  Extracts structured patient profile
  │             │  Color-codes: cyan=extracted, amber=inferred
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────┐
  │        FITTED RESERVOIR MODEL           │
  │                                         │
  │  Spring-to-equilibrium dynamics         │
  │  + learned multi-channel tipping points │
  │  + 3 hidden latent reservoirs           │
  │                                         │
  │  216 parameters. Every one nameable.    │
  └──────┬──────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │   Oracle    │  Generates 4 intervention strategies
  │             │  Simulates each 24h through fitted model
  │             │  Ranks by composite outcome score
  └─────────────┘
```

**Key features:**
- Model learns connection weights, equilibria, spring rates, AND nonlinear tipping thresholds from patient data
- Hidden reservoirs discover latent physiological states (auto-named by connection patterns)
- Every parameter is interpretable — exportable as human-readable JSON
- LLM runs locally via Ollama — no patient data leaves the building
- Clinician can inspect, edit, and override any parameter

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (optional, for LLM-powered natural language parsing)

### Install

```bash
git clone https://github.com/DormantOne/shock-lab.git
cd shock-lab
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Usage

1. **Click "Fit Model"** — optimizes 216 parameters against 1000 synthetic patient trajectories (~60-90 seconds)
2. **Explore the learned network** — see which connections survived, what hidden reservoirs discovered, what tipping points emerged
3. **Ask the Oracle** — type a clinical scenario in plain language (e.g., "septic shock, MAP 50, lactate 7, HR 130"), review the parsed profile, run the experiment
4. **Compare strategies** — 4 intervention scenarios with predicted 24h trajectories, ranked by composite score

### Optional: Connect Ollama for natural language parsing

```bash
# Install Ollama (https://ollama.ai)
ollama pull gpt-oss:20b
ollama serve
```

The app auto-detects Ollama. Without it, keyword-based parsing still works as fallback.

---

## The Model

### Architecture: Spring-to-Equilibrium + Tipping Points

Each clinical variable is a reservoir with two forces:

**Linear spring:** pulls the state toward a shifted equilibrium
```
state += (effective_eq - state) × spring_rate × dt
effective_eq = base_eq + Σ(source_value × connection_weight)
```

**Nonlinear tipping forces:** learned sigmoids that fire at critical thresholds
```
tip_force = Σ_k  magnitude_k × σ(steepness_k × (state - threshold_k))
```

Each reservoir has 3 tipping channels. Each channel learns:
- **Threshold** — where the edge is (e.g., MAP at 65 mmHg)
- **Steepness** — how sharp (gradual decline vs cliff-edge collapse)
- **Magnitude** — how strong and which direction

Four tipping behaviors emerge:
| Direction | Effect | Clinical Example |
|-----------|--------|-----------------|
| Below threshold | Collapse (accelerate down) | Autoregulatory failure |
| Above threshold | Cascade (accelerate up) | Lactate spiral |
| Below threshold | Rebound (compensate up) | Protective reflex |
| Above threshold | Ceiling (suppress down) | Negative feedback |

### Parameters (216 total)

| Component | Count | What it controls |
|-----------|-------|-----------------|
| Connection weights | 117 | How strongly each reservoir influences each other |
| Equilibria | 9 | Where each reservoir rests |
| Spring rates | 9 | How fast each reservoir chases its equilibrium |
| Tipping thresholds | 27 | Where each nonlinear edge sits (9 reservoirs × 3 channels) |
| Tipping steepness | 27 | How sharp each transition is |
| Tipping magnitude | 27 | How strong each tipping force is |

### Reservoirs

**Inputs (held constant during simulation):**
- Vasopressor type/dose
- Fluid volume (mL/kg)
- Mechanical support (none/IABP/Impella/ECMO)
- Age

**Observable targets (evolve over 24h):**
- MAP, HR, Lactate, Creatinine, WBC, Procalcitonin

**Hidden (discovered from data):**
- 3 unnamed reservoirs that develop identities after fitting

### Fitting Process

1. Load patient data (1000 trajectories with vitals at 0h, 6h, 12h, 24h)
2. Phase 1: fast optimization on 200-patient subsample
3. Phase 2: refinement on full dataset
4. Prune connections below threshold (weak links → zero)
5. Prune tipping points with negligible magnitude
6. Analyze hidden reservoir connection patterns → auto-name

Loss function: MSE on MAP/HR/lactate at 6h/12h/24h + L2 regularization on weights + L1 on tipping magnitudes (drives unused tipping points to zero).

---

## Slideshow / Presentation

A self-contained HTML slideshow is included in `demo 3/`:

```bash
open "demo 3/slideshow.html"
```

Arrow keys to navigate, `m` to mute narration, speaker button to toggle.

### Optional: Premium voice narration (macOS)

1. System Settings → Accessibility → Spoken Content → System Voice → Manage Voices
2. Download "Zoe (Premium)" or another neural voice
3. Generate audio files:

```bash
cd "demo 3"
bash generate_audio.sh
```

Then re-open `slideshow.html` — it auto-detects the `.m4a` files.

---

## File Structure

```
shock-lab/
├── app.py                  ← Flask server + routes
├── reservoir_core.py       ← Core model: spring dynamics + tipping points
├── fit_pipeline.py         ← Data loading, loss function, scipy optimizer
├── experimenter.py         ← Scenario generation, scoring, trajectory comparison
├── llm_client.py           ← Ollama integration + keyword fallback
├── shock_cases.csv         ← 1000 synthetic shock patient trajectories
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Dashboard UI
└── demo 3/
    ├── slideshow.html      ← Narrated presentation (13 slides)
    ├── generate_audio.sh   ← macOS script for premium voice narration
    ├── image1.png          ← Screenshot: fitting + network
    ├── image2.png          ← Screenshot: hidden reservoirs + LLM parsing
    └── image3.png          ← Screenshot: oracle experiment results
```

---

## Important Caveats

**This is a prototype on synthetic data.** The synthetic data generator encodes known relationships (e.g., norepinephrine improves MAP in septic shock). The model is reverse-engineering planted signal, not discovering novel biology. Real clinical data will be noisier, messier, and harder.

**Correlation, not causation.** Sicker patients receive more aggressive treatment. Without causal inference methods, the model may learn that aggressive treatment *causes* bad outcomes (confounding by indication).

**Tipping points are weak in synthetic data.** The planted dynamics are mostly linear, so discovered tipping points are "soft." With real patient data containing autoregulatory failure and lactate spirals, we expect sharper, clinically meaningful thresholds.

**Not validated for clinical use.** No prospective testing. No comparison to clinician decision-making. No regulatory review. This is a research prototype.

---

## What Would Make This Real

1. **Real data** — de-identified shock patient trajectories from an EHR system
2. **Causal inference** — methods to handle confounding by indication
3. **Stress testing** — adversarial scenarios, cross-validation, holdout testing
4. **Clinical review** — physicians inspect every connection and tipping point
5. **Team** — critical care physician, biostatistician, data engineer, regulatory guidance

---

## License

Experimental / research use only. Not licensed for clinical decision-making.
