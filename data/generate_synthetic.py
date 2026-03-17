#!/usr/bin/env python3
"""
Generate synthetic reasoning data via Anthropic API (section 5.2 of spec).

Distribution:
  35K examples in Spanish (70%)
  15K examples in English  (30%)

Categories per language:
  1. Propositional logic
  2. Everyday math with explicit steps
  3. Causal reasoning (cause -> effect)
  4. Justified structural analogies

Validation (es_valido) — three-layer filter:
  1. Structural: >=3 steps, each >40 chars
  2. Semantic:   explicit subject + causal/logical connector per step
  3. Math:       "verificacion" field must pass sympy evaluation

Usage:
  python data/generate_synthetic.py --sample          # 500 for review
  python data/generate_synthetic.py --review          # print 5 random
  python data/generate_synthetic.py --full            # remaining 49,500
  python data/generate_synthetic.py --all             # all 50,000 at once
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import re

import anthropic
from sympy import Eq, Rational, nsimplify, sympify

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PHASE2 = SCRIPT_DIR / "raw" / "phase2"
SAMPLE_PATH = RAW_PHASE2 / "synthetic_sample.jsonl"
FULL_PATH = RAW_PHASE2 / "synthetic.jsonl"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = {
    "es": (
        "Eres un generador de datos de entrenamiento para un modelo de lenguaje pequeño. "
        "Genera ejemplos de razonamiento explícito en español. "
        "Cada ejemplo debe mostrar el proceso de pensamiento paso a paso, no solo la respuesta. "
        "Responde SOLO con JSON válido, sin explicaciones adicionales ni bloques de código."
    ),
    "en": (
        "You are a training-data generator for a small language model. "
        "Generate examples of explicit reasoning in English. "
        "Each example must show the step-by-step thought process, not just the answer. "
        "Respond ONLY with valid JSON, no extra explanations or code blocks."
    ),
}

# Math templates request a "verificacion" field with a Python/sympy expression
_VERIFICACION_NOTE_ES = (
    ' Además, incluye un campo "verificacion" con una expresión Python evaluable '
    'que verifique la respuesta final. Ejemplo: "verificacion": "3 * 25 == 75"'
)
_VERIFICACION_NOTE_EN = (
    ' Also include a "verificacion" field with an evaluable Python expression '
    'that verifies the final answer. Example: "verificacion": "3 * 25 == 75"'
)

TEMPLATES = {
    "es": {
        "logica": (
            "Genera un problema ORIGINAL de lógica proposicional con solución paso a paso en español. "
            "El problema debe involucrar al menos 3 premisas y una conclusión. "
            "Varía la dificultad y el tema (no repitas patrones). "
            'Formato JSON: {"pregunta": "...", "razonamiento": ["Paso 1: ...", "Paso 2: ...", '
            '"Paso 3: ..."], "conclusion": "..."}'
        ),
        "matematicas": (
            "Genera un problema matemático cotidiano ORIGINAL (compras, distancias, tiempos, "
            "porcentajes, proporciones) con razonamiento explícito paso a paso en español. "
            "Usa números concretos. NO uses ecuaciones abstractas — todo en lenguaje natural "
            "con operaciones claras."
            + _VERIFICACION_NOTE_ES +
            ' Formato JSON: {"pregunta": "...", "razonamiento": ["Paso 1: ...", "Paso 2: ...", '
            '"Paso 3: ..."], "conclusion": "...", "verificacion": "..."}'
        ),
        "causal": (
            "Genera un ejemplo ORIGINAL de razonamiento causa-efecto con cadena de pasos "
            "explícita en español. Debe incluir una situación inicial, al menos 3 pasos "
            "causales encadenados, y un resultado final. "
            "Temas variados: ciencia, economía, ecología, salud, sociedad. "
            'Formato JSON: {"pregunta": "...", "razonamiento": ["Paso 1: ...", "Paso 2: ...", '
            '"Paso 3: ..."], "conclusion": "..."}'
        ),
        "analogias": (
            "Genera un problema ORIGINAL de analogía estructural con justificación de cada "
            "paso en español. Presenta dos dominios y explica paso a paso por qué la "
            "estructura es análoga. "
            'Formato JSON: {"pregunta": "...", "razonamiento": ["Paso 1: ...", "Paso 2: ...", '
            '"Paso 3: ..."], "conclusion": "..."}'
        ),
    },
    "en": {
        "logic": (
            "Generate an ORIGINAL propositional logic problem with a step-by-step solution "
            "in English. The problem must involve at least 3 premises and a conclusion. "
            "Vary difficulty and topic (don't repeat patterns). "
            'JSON format: {"pregunta": "...", "razonamiento": ["Step 1: ...", "Step 2: ...", '
            '"Step 3: ..."], "conclusion": "..."}'
        ),
        "math": (
            "Generate an ORIGINAL everyday math problem (shopping, distances, times, "
            "percentages, proportions) with explicit step-by-step reasoning in English. "
            "Use concrete numbers. NO abstract equations — everything in natural language "
            "with clear operations."
            + _VERIFICACION_NOTE_EN +
            ' JSON format: {"pregunta": "...", "razonamiento": ["Step 1: ...", "Step 2: ...", '
            '"Step 3: ..."], "conclusion": "...", "verificacion": "..."}'
        ),
        "causal": (
            "Generate an ORIGINAL cause-and-effect reasoning example with an explicit chain "
            "of steps in English. Include an initial situation, at least 3 chained causal "
            "steps, and a final outcome. "
            "Varied topics: science, economics, ecology, health, society. "
            'JSON format: {"pregunta": "...", "razonamiento": ["Step 1: ...", "Step 2: ...", '
            '"Step 3: ..."], "conclusion": "..."}'
        ),
        "analogies": (
            "Generate an ORIGINAL structural analogy problem with justification of each step "
            "in English. Present two domains and explain step by step why the structure is "
            "analogous. "
            'JSON format: {"pregunta": "...", "razonamiento": ["Step 1: ...", "Step 2: ...", '
            '"Step 3: ..."], "conclusion": "..."}'
        ),
    },
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Subjects: nouns, pronouns, determiners that signal an explicit subject.
# We check that at least one appears BEFORE the first verb-like word.
# This catches subjectless fragments like "Sube y se enfría."
_SUBJECT_MARKERS_ES = (
    # determiners
    "el ", "la ", "los ", "las ", "un ", "una ", "unos ", "unas ",
    "este ", "esta ", "estos ", "estas ", "ese ", "esa ",
    # pronouns
    "esto ", "eso ", "ello ",
    # common reasoning subjects
    "cada ", "todo ", "toda ", "todos ", "todas ",
    "dicho ", "dicha ", "dichos ", "dichas ",
    "el resultado ", "el total ", "el valor ", "el precio ",
    "la suma ", "la diferencia ", "la cantidad ",
    "ambos ", "ambas ", "ninguno ", "ninguna ",
    # 1st-person plural (implicit subject "nosotros" in Spanish)
    "sabemos ", "observamos ", "notamos ", "necesitamos ",
    "calculamos ", "multiplicamos ", "sumamos ", "restamos ",
    "dividimos ", "obtenemos ", "aplicamos ", "usamos ",
    # "sin" + noun is a valid adverbial subject phrase
    "sin ",
    # demonstrative reference to prior step
    "esta ", "estas ",
)

_SUBJECT_MARKERS_EN = (
    "the ", "a ", "an ", "this ", "that ", "these ", "those ",
    "each ", "every ", "all ", "both ", "neither ", "either ",
    "it ", "he ", "she ", "they ", "we ", "i ",
    "the result ", "the total ", "the value ", "the price ",
    "the sum ", "the difference ", "the amount ",
    # Preposition-led phrases common in reasoning ("For N units, ...")
    "for ", "in ", "on ", "at ", "with ",
)

# In English, bare nouns and gerunds are valid subjects (e.g. "Voting
# functions like...", "Data packets are...").  We accept any step body
# whose first word is 3+ characters — this catches real subjects while
# still rejecting empty/garbage steps (which fail other checks anyway).
_EN_MIN_FIRST_WORD_LEN = 3

# Structure check: does the step connect to the problem, a prior step,
# or a logical/causal rule?  Three categories:
#
# 1. Causal/logical keywords — "porque", "therefore", "dado que", etc.
# 2. References — "el enunciado", "la premisa", "the problem", etc.
# 3. Computation verbs — "multiplicamos", "calculate", etc.
# 4. Symbols — ×, =
#
# All matched with \b to avoid substring false positives.
_CONNECTOR_PATTERNS = re.compile(
    r"\b(?:"
    # --- Spanish ---
    # Causal / logical
    r"por lo tanto|porque|dado que|ya que|debido a|puesto que"
    r"|entonces|por eso|por esto|por ende|por consiguiente"
    r"|en consecuencia|como resultado|como consecuencia"
    r"|esto causa|esto provoca|esto implica|esto significa|esto genera"
    r"|es decir|o sea|así que|de modo que|de manera que"
    r"|lo que significa|lo que implica|lo cual"
    # Reference to source
    r"|según|conforme a|de acuerdo con"
    r"|sabemos que|observamos que|notamos que|nos dice que"
    r"|establece que|indica que|afirma que|señala que|implica que"
    r"|el enunciado|el problema|la premisa|la condición|el dato"
    # Computation
    r"|multiplicamos|sumamos|restamos|dividimos|calculamos|obtenemos"
    r"|aplicando|usando|sustituyendo|reemplazando|comparando"
    # --- English ---
    # Causal / logical
    r"|therefore|because|since|given that|due to|owing to"
    r"|consequently|as a result|as a consequence|thus|hence"
    r"|this causes|this means|this implies|this leads|this gives"
    r"|which means|which implies|which gives|which leads"
    r"|that is|in other words|it follows"
    # Reference to source
    r"|we know|we observe|we note|we need|we see|we have"
    r"|tells us|states that|establishes|indicates|according to"
    r"|the problem|the premise|the statement|the condition|the data"
    # Computation
    r"|multiply|subtract|divide|calculate|compute|obtain"
    r"|applying|substituting|replacing|combining|comparing"
    r")\b"
    # Symbols (no word boundary needed)
    r"|[×=]",
    re.IGNORECASE,
)


def _paso_tiene_sujeto(paso: str) -> bool:
    """Check that the step has an explicit subject near the start.

    A step is considered to have a subject if ANY of these hold:
    1. Starts with a subject marker (determiner/pronoun/1st-person verb)
    2. Starts with a connector phrase ("Como resultado", "Consequently")
       — these reference the prior step and act as the logical subject
    3. Has a subject marker right after a leading comma-delimited phrase
       ("Aplicando X, los perros...")

    Rejects subjectless fragments like "Asciende a la atmósfera" where
    none of the above patterns match.
    """
    lower = paso.lower()
    for prefix in ("paso ", "step "):
        if lower.startswith(prefix):
            colon = lower.find(":")
            if colon != -1:
                lower = lower[colon + 1:].lstrip()
            break

    all_markers = _SUBJECT_MARKERS_ES + _SUBJECT_MARKERS_EN

    # 1. Check start of sentence for subject marker
    for marker in all_markers:
        if lower.startswith(marker):
            return True

    # 2. Check if step starts with a connector phrase — these implicitly
    #    reference the prior step as subject ("Como resultado, ...")
    if _CONNECTOR_PATTERNS.match(lower):
        return True

    # 3. Check after first comma (introductory phrase)
    comma = lower.find(",")
    if comma != -1:
        after_comma = lower[comma + 1:].lstrip()
        for marker in all_markers:
            if after_comma.startswith(marker):
                return True

    # 4. English: bare nouns/gerunds are valid subjects ("Voting functions
    #    like...", "Data packets are...", "Natural selection works...").
    #    Only applies to steps with "Step N:" prefix.
    if paso.lower().startswith("step "):
        first_word = lower.split()[0] if lower else ""
        if len(first_word) >= _EN_MIN_FIRST_WORD_LEN:
            return True

    return False


def _paso_tiene_conector(paso: str) -> bool:
    """Check that the step has causal/logical structure."""
    return bool(_CONNECTOR_PATTERNS.search(paso))


def es_valido(ejemplo: dict) -> tuple[bool, str]:
    """Validate a generated example.

    Returns (True, "") if valid, (False, reason) if not.

    Three-layer filter:
      1. Structural: >=3 steps, each >40 chars        -> "few_steps" / "short_step"
      2. Semantic:   explicit subject + causal/logical  -> "no_subject" / "no_structure"
         connector in every step
      3. Math:       "verificacion" passes sympy        -> "math_error"
    """
    for key in ("pregunta", "razonamiento", "conclusion"):
        if key not in ejemplo:
            return False, "few_steps"

    pasos = ejemplo["razonamiento"]
    if not isinstance(pasos, list) or len(pasos) < 3:
        return False, "few_steps"

    for paso in pasos:
        if not isinstance(paso, str) or len(paso) <= 40:
            return False, "short_step"

    for paso in pasos:
        if not _paso_tiene_sujeto(paso):
            return False, "no_subject"

    for paso in pasos:
        if not _paso_tiene_conector(paso):
            return False, "no_structure"

    if "verificacion" in ejemplo:
        try:
            expr = ejemplo["verificacion"]
            if "==" in expr:
                lhs, rhs = expr.split("==", 1)
                left = nsimplify(sympify(lhs.strip()), rational=True)
                right = nsimplify(sympify(rhs.strip()), rational=True)
                if not Eq(left, right):
                    return False, "math_error"
            else:
                if not sympify(expr):
                    return False, "math_error"
        except Exception:
            return False, "math_error"

    return True, ""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_example(example: dict, lang: str) -> str:
    """Convert validated example to the unified instruction format."""
    if lang == "es":
        sistema = "Eres un asistente que razona paso a paso antes de responder."
        conclusion_label = "Conclusión"
    else:
        sistema = "You are an assistant that reasons step by step before answering."
        conclusion_label = "Conclusion"

    steps_text = "\n".join(example["razonamiento"])
    respuesta = f"{steps_text}\n{conclusion_label}: {example['conclusion']}"

    return (
        f"<|system|>\n{sistema}\n"
        f"<|user|>\n{example['pregunta']}\n"
        f"<|assistant|>\n{respuesta}\n"
        f"<|end|>"
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from Claude's response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_batch(
    client: anthropic.Anthropic,
    lang: str,
    category: str,
    count: int,
    seed_offset: int = 0,
) -> list[dict]:
    """Generate a batch of examples for one language/category.

    One API call per example for maximum diversity.
    Retries up to 2 times on failure.
    """
    template = TEMPLATES[lang][category]
    system = SYSTEM_PROMPT[lang]
    results = []

    for i in range(count):
        seed = seed_offset + i
        variation = f" (variation #{seed}, be creative and do not repeat previous examples)"
        prompt = template + variation

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                example = _parse_json_response(text)

                if example is None:
                    continue

                is_valid, reason = es_valido(example)
                if not is_valid:
                    continue

                formatted = format_example(example, lang)
                record = {
                    "text": formatted,
                    "lang": lang,
                    "category": category,
                }
                if "verificacion" in example:
                    record["verificacion"] = example["verificacion"]

                results.append(record)
                break

            except anthropic.RateLimitError:
                time.sleep(5)
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] failed after 3 attempts: {e}", file=sys.stderr)
                time.sleep(1)

    return results


def generate_examples(
    client: anthropic.Anthropic,
    n_es: int,
    n_en: int,
    output_path: Path,
    seed_offset: int = 0,
    batch_label: str = "",
) -> int:
    """Generate examples across all categories, balanced per language."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    es_categories = list(TEMPLATES["es"].keys())
    en_categories = list(TEMPLATES["en"].keys())

    per_cat_es = n_es // len(es_categories)
    per_cat_en = n_en // len(en_categories)

    total_written = 0

    with open(output_path, "a") as f:
        for cat in es_categories:
            label = f"{batch_label}[es/{cat}]" if batch_label else f"[es/{cat}]"
            print(f"  {label} generating {per_cat_es} examples ...")
            results = generate_batch(client, "es", cat, per_cat_es, seed_offset)
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_written += len(results)
            print(f"  {label} {len(results)}/{per_cat_es} valid")

        for cat in en_categories:
            label = f"{batch_label}[en/{cat}]" if batch_label else f"[en/{cat}]"
            print(f"  {label} generating {per_cat_en} examples ...")
            results = generate_batch(client, "en", cat, per_cat_en, seed_offset)
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_written += len(results)
            print(f"  {label} {len(results)}/{per_cat_en} valid")

    return total_written


# ---------------------------------------------------------------------------
# Sample review
# ---------------------------------------------------------------------------


def print_sample_review(path: Path, n: int = 5) -> None:
    """Print n random examples from a JSONL file for manual review."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    selected = random.sample(examples, min(n, len(examples)))

    for i, ex in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}  [{ex['lang']}/{ex['category']}]")
        print(f"{'='*60}")
        print(ex["text"])
        if "verificacion" in ex:
            print(f"\n  verificacion: {ex['verificacion']}")
    print(f"\n{'='*60}")
    print(f"Total examples in file: {len(examples)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic reasoning data via Anthropic API.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample", action="store_true",
        help="Generate 500-example sample (250 ES + 250 EN) for review.",
    )
    group.add_argument(
        "--full", action="store_true",
        help="Generate remaining 49,500 examples (after sample review).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Generate all 50,000 examples at once (no review step).",
    )
    group.add_argument(
        "--review", action="store_true",
        help="Print 5 random examples from the sample file for review.",
    )
    args = parser.parse_args()

    if args.review:
        if not SAMPLE_PATH.exists():
            print(f"Sample file not found: {SAMPLE_PATH}", file=sys.stderr)
            print("Run with --sample first.", file=sys.stderr)
            sys.exit(1)
        print_sample_review(SAMPLE_PATH)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set.\n"
            "Export it:  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.sample:
        print(f"Generating 500-example sample -> {SAMPLE_PATH}")
        if SAMPLE_PATH.exists():
            SAMPLE_PATH.unlink()
        total = generate_examples(client, n_es=250, n_en=250, output_path=SAMPLE_PATH)
        print(f"\nSample done: {total} valid examples written to {SAMPLE_PATH}")
        print("\nReview with:  python data/generate_synthetic.py --review")
        print("Then run:     python data/generate_synthetic.py --full")

    elif args.full:
        if not SAMPLE_PATH.exists():
            print("WARNING: no sample file found — generating full set from scratch.",
                  file=sys.stderr)

        print(f"Generating 49,500 remaining examples -> {FULL_PATH}")
        if FULL_PATH.exists():
            FULL_PATH.unlink()

        # Copy sample to full if it exists
        if SAMPLE_PATH.exists():
            import shutil
            shutil.copy2(SAMPLE_PATH, FULL_PATH)
            with open(SAMPLE_PATH) as f:
                existing = sum(1 for _ in f)
            print(f"Copied {existing} sample examples to {FULL_PATH}")
        else:
            existing = 0

        remaining_es = 35_000 - int(existing * 0.5)  # sample is 50/50
        remaining_en = 15_000 - int(existing * 0.5)
        total_remaining = remaining_es + remaining_en

        BATCH_SIZE = 1000
        batch_es = int(BATCH_SIZE * 0.7)
        batch_en = BATCH_SIZE - batch_es

        generated = 0
        batch_num = 0
        while generated < total_remaining:
            batch_num += 1
            this_es = min(batch_es, remaining_es - int(generated * 0.7))
            this_en = min(batch_en, remaining_en - int(generated * 0.3))
            if this_es <= 0 and this_en <= 0:
                break
            this_es = max(0, this_es)
            this_en = max(0, this_en)

            print(f"\n--- Batch {batch_num} ({generated:,}/{total_remaining:,}) ---")
            n = generate_examples(
                client, n_es=this_es, n_en=this_en,
                output_path=FULL_PATH,
                seed_offset=existing + generated,
                batch_label=f"B{batch_num} ",
            )
            generated += n
            print(f"  Batch {batch_num} done: +{n} (total so far: {existing + generated:,})")

        with open(FULL_PATH) as f:
            final_count = sum(1 for _ in f)
        print(f"\nFull generation done: {final_count:,} total examples in {FULL_PATH}")

    elif args.all:
        print(f"Generating all 50,000 examples -> {FULL_PATH}")
        if FULL_PATH.exists():
            FULL_PATH.unlink()

        BATCH_SIZE = 1000
        batch_es = int(BATCH_SIZE * 0.7)
        batch_en = BATCH_SIZE - batch_es
        total_target = 50_000

        generated = 0
        batch_num = 0
        while generated < total_target:
            batch_num += 1
            this_es = min(batch_es, int(total_target * 0.7) - int(generated * 0.7))
            this_en = min(batch_en, int(total_target * 0.3) - int(generated * 0.3))
            if this_es <= 0 and this_en <= 0:
                break
            this_es = max(0, this_es)
            this_en = max(0, this_en)

            print(f"\n--- Batch {batch_num} ({generated:,}/{total_target:,}) ---")
            n = generate_examples(
                client, n_es=this_es, n_en=this_en,
                output_path=FULL_PATH,
                seed_offset=generated,
                batch_label=f"B{batch_num} ",
            )
            generated += n
            print(f"  Batch {batch_num} done: +{n} (total: {generated:,})")

        with open(FULL_PATH) as f:
            final_count = sum(1 for _ in f)
        print(f"\nAll done: {final_count:,} total examples in {FULL_PATH}")


if __name__ == "__main__":
    main()
