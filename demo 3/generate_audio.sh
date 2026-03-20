#!/bin/bash
# ============================================================
# Generate narration audio for Shock Lab slideshow
# ============================================================
# Run this on your Mac ONCE after downloading a premium voice:
#   System Settings → Accessibility → Spoken Content → 
#   System Voice → Manage Voices → download "Zoe (Premium)" or "Evan (Premium)"
#
# Then run:  bash generate_audio.sh
# It creates slide_00.m4a through slide_11.m4a
# ============================================================

# Pick your voice (uncomment one):
VOICE="Zoe (Premium)"
# VOICE="Evan (Premium)"
# VOICE="Samantha (Enhanced)"
# VOICE="Tom (Enhanced)"

RATE=185  # words per minute (default ~175, slightly faster feels natural)

echo "Using voice: $VOICE at rate $RATE"
echo "Generating narration for 12 slides..."

say -v "$VOICE" -r $RATE -o slide_00.m4a -- "Welcome to Shock Lab — an experimental data-driven clinical decision support prototype. It uses reservoir computing to learn from patient trajectories, discovers hidden physiological states, and lets clinicians test intervention strategies on a digital twin. This was prepared for Doctor Gavin Hickey and Nurse Nicole Kunz." &
wait

say -v "$VOICE" -r $RATE -o slide_01.m4a -- "Here's the core problem. Shock management is high-stakes and time-critical. Black-box models can predict outcomes but can't explain why. Published guidelines are population averages that don't adapt. What we want is something in between — a model that learns from our own data, shows its reasoning, and lets you experiment before committing." &
wait

say -v "$VOICE" -r $RATE -o slide_02.m4a -- "The architecture uses spring-to-equilibrium physics. Every clinical variable has a resting equilibrium. Connections between variables shift those equilibria. The math is one line: state springs toward its effective equilibrium at a certain rate. Positive connections push up — fill. Negative connections push down — leak. Unlike a neural net, every parameter has a physiological name and meaning. A clinician can read it, challenge it, and edit it." &
wait

say -v "$VOICE" -r $RATE -o slide_03.m4a -- "Alongside the observable variables, the model has three unnamed hidden reservoirs. Before fitting, they're blank slates. After the optimizer runs, they develop distinct connection patterns and get auto-named — Convergence Node, Organ Stress, Latent Factor. These aren't categories — they're continuous. A patient can have high values across multiple hidden reservoirs simultaneously. With real data, they might discover inflammatory burden or compensatory reserve." &
wait

say -v "$VOICE" -r $RATE -o slide_04.m4a -- "Here's the fitting dashboard. The optimizer ran against a thousand patient trajectories and achieved ninety-one percent loss reduction. MAP predictions land within five millimeters of mercury. Lactate within one millimol per liter. Twenty-seven of a hundred seventeen connections survived pruning. Important caveat — this is synthetic data with planted signal. Real data will be harder. But the machinery works." &
wait

say -v "$VOICE" -r $RATE -o slide_05.m4a -- "This is the learned network. Input reservoirs at the top — vasopressor, fluids, mechanical support, age. Physiology in the middle. Labs below. Hidden reservoirs at the bottom with their discovered identities. The connection table shows the strongest relationships. The LLM parsing interface correctly extracted patient values from natural language and flagged what it inferred in amber." &
wait

say -v "$VOICE" -r $RATE -o slide_06.m4a -- "The fitting process: load patients, set their initial state, simulate forward twenty-four hours, compare to what actually happened, compute the loss, adjust weights. A hundred thirty-five parameters total. L-2 regularization drives useless connections to zero. Then we prune. The key question — why not just use a neural net? Because neural net weights are meaningless numbers. Here, every weight connects two named clinical variables. That interpretability is the entire point." &
wait

say -v "$VOICE" -r $RATE -o slide_07.m4a -- "The Oracle is the clinical interface. Type a patient description in plain language. The local LLM parses it — cyan means extracted from your text, amber means the system made it up. Every value is editable. When you run the experiment, four intervention strategies simulate through the fitted model for twenty-four hours and get ranked. The LLM runs locally through Ollama — no patient data leaves the building." &
wait

say -v "$VOICE" -r $RATE -o slide_08.m4a -- "The composite score has four components. MAP recovery is worth up to forty points. Lactate clearance is twenty-five. Heart rate normalization adds fifteen. Renal preservation acts as a penalty — creatinine rises reduce the score. All weights are configurable based on what your team prioritizes." &
wait

say -v "$VOICE" -r $RATE -o slide_09.m4a -- "Let me be honest about what's real and what isn't. The fitting works. Hidden reservoirs discover structure. The model is transparent and editable. The LLM integration works locally. But — this is synthetic data. The connections are correlation, not causation. Real data arrives irregularly. There's no prospective validation. The question is whether this architecture is worth testing against real patient data." &
wait

say -v "$VOICE" -r $RATE -o slide_10.m4a -- "Proposed next steps. First — data access. We need de-identified shock trajectories from the EHR. Second — fit against real data and see what emerges. Third — clinical review. Physicians and nurses inspect the network, name hidden reservoirs, add calibration tests. Fourth — assemble the team. Critical care, biostatistics, data engineering, and eventually regulatory guidance." &
wait

say -v "$VOICE" -r $RATE -o slide_11.m4a -- "To close — the model is interpretable. The parameters are learned. The clinician is in control. This is a toy built on synthetic data, but the architecture is real. The question is whether it survives contact with the beautiful complexity of actual patient outcomes. Let's find out together." &
wait

echo ""
echo "Done! Generated files:"
ls -lh slide_*.m4a
echo ""
echo "Place these files next to slideshow.html and open in browser."
