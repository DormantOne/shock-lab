"""
LLM Client — Ollama Integration for Clinical Text → Structured Profile
========================================================================
Sends natural language to Ollama, gets back structured patient data.
Tracks which values were extracted vs inferred (made up).
Falls back to keyword parsing if Ollama is unavailable.
"""

import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"

SYSTEM_PROMPT = """You are a clinical data extractor. Given a free-text description of a shock patient, extract a structured JSON profile.

Return ONLY a JSON object with these exact fields:
{
  "shock_type": "septic" or "cardiogenic" or "hypovolemic" or "distributive" or "unknown",
  "map": number (mean arterial pressure in mmHg),
  "hr": number (heart rate in bpm),
  "lactate": number (mmol/L),
  "creatinine": number (mg/dL),
  "wbc": number (K/uL),
  "procal": number (procalcitonin ng/mL),
  "age": number (years),
  "extracted": ["list of field names you found in the text"],
  "inferred": ["list of field names you made up because they weren't mentioned"],
  "reasoning": "brief explanation of your extraction and inferences"
}

RULES:
- If a value is explicitly stated in the text, extract it and put the field name in "extracted"
- If a value is NOT mentioned, infer a clinically reasonable value for the shock type and put the field name in "inferred"
- For inferred values, use typical values for the described clinical scenario
- shock_type goes in "extracted" if the type is stated or clearly implied, otherwise "inferred"
- Be generous in interpretation: "pressure is low" → low MAP, "tachycardic" → high HR, "kidney function worsening" → elevated creatinine
- Return ONLY the JSON object. No markdown, no backticks, no explanation outside the JSON."""

# Reasonable defaults if LLM is unavailable
DEFAULTS = {
    "map": 55, "hr": 110, "lactate": 4.5, "creatinine": 1.3,
    "wbc": 15, "procal": 5.0, "age": 65,
}

SHOCK_DEFAULTS = {
    "septic": {"map": 52, "hr": 118, "lactate": 5.5, "creatinine": 1.4, "wbc": 22, "procal": 12, "age": 64},
    "cardiogenic": {"map": 58, "hr": 105, "lactate": 3.8, "creatinine": 1.6, "wbc": 11, "procal": 1.5, "age": 71},
    "hypovolemic": {"map": 48, "hr": 135, "lactate": 6.0, "creatinine": 1.1, "wbc": 13, "procal": 2.0, "age": 55},
    "unknown": {"map": 55, "hr": 110, "lactate": 4.5, "creatinine": 1.3, "wbc": 15, "procal": 5.0, "age": 65},
}


def call_ollama(question):
    """
    Call Ollama to parse clinical text.
    Returns (profile_dict, used_llm:bool)
    """
    prompt = SYSTEM_PROMPT + "\n\nPatient description:\n" + question + "\n\nJSON:"

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 600},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "")

            # Clean up: strip markdown fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()

            # Parse JSON
            parsed = json.loads(clean)

            # Validate and fill missing fields
            profile = {}
            extracted = parsed.get("extracted", [])
            inferred = parsed.get("inferred", [])
            reasoning = parsed.get("reasoning", "")
            shock_type = parsed.get("shock_type", "unknown")

            for field in ["map", "hr", "lactate", "creatinine", "wbc", "procal", "age"]:
                if field in parsed and parsed[field] is not None:
                    try:
                        profile[field] = float(parsed[field])
                    except (ValueError, TypeError):
                        profile[field] = DEFAULTS.get(field, 0)
                        if field not in inferred:
                            inferred.append(field)
                else:
                    profile[field] = DEFAULTS.get(field, 0)
                    if field not in inferred:
                        inferred.append(field)

            return {
                "profile": profile,
                "shock_type": shock_type,
                "extracted": extracted,
                "inferred": inferred,
                "reasoning": reasoning,
                "used_llm": True,
                "model": OLLAMA_MODEL,
                "raw_response": raw[:500],
            }, True

    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
        return None, False
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # LLM returned something but it wasn't valid JSON
        return {
            "error": f"LLM returned invalid JSON: {str(e)}",
            "raw_response": raw[:500] if 'raw' in dir() else "",
        }, False


def fallback_parse(question):
    """Keyword-based fallback when Ollama is unavailable."""
    q = question.lower()

    # Detect shock type
    if "septic" in q or "sepsis" in q or "infection" in q:
        shock_type = "septic"
    elif "cardiogenic" in q or "cardiac" in q or "mi " in q or "heart failure" in q:
        shock_type = "cardiogenic"
    elif "hypovolemic" in q or "hemorrhag" in q or "bleed" in q or "trauma" in q:
        shock_type = "hypovolemic"
    else:
        shock_type = "unknown"

    # Extract numbers
    extracted = []
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
                    extracted.append(key)
                    break

    if shock_type != "unknown":
        extracted.append("shock_type")

    # Fill defaults for missing values
    defaults = SHOCK_DEFAULTS.get(shock_type, DEFAULTS)
    inferred = []
    for field in ["map", "hr", "lactate", "creatinine", "wbc", "procal", "age"]:
        if field not in profile:
            profile[field] = defaults[field]
            inferred.append(field)

    if "shock_type" not in extracted:
        inferred.append("shock_type")

    return {
        "profile": profile,
        "shock_type": shock_type,
        "extracted": extracted,
        "inferred": inferred,
        "reasoning": "Parsed via keyword matching (Ollama unavailable). Inferred values are typical for " + shock_type + " shock.",
        "used_llm": False,
        "model": "keyword_fallback",
    }


def _extract_number(text, start, max_chars=8):
    """Extract a number from text starting at position."""
    num = ""
    has_dot = False
    for ch in text[start:start + max_chars]:
        if ch.isdigit():
            num += ch
        elif ch == "." and not has_dot:
            next_pos = start + len(num) + 1
            if next_pos < len(text) and text[next_pos].isdigit():
                num += ch
                has_dot = True
            else:
                break
        elif num:
            break
    if num:
        try:
            return float(num)
        except ValueError:
            return None
    return None


def parse_clinical_text(question):
    """
    Main entry point. Try Ollama first, fall back to keywords.
    Returns structured profile with extracted/inferred tracking.
    """
    result, used_llm = call_ollama(question)

    if used_llm and result and "profile" in result:
        return result

    # Fallback
    return fallback_parse(question)
