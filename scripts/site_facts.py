"""
Collect every number the playable page quotes into one measured file.

The explainer on the table page makes specific claims — how many physics tests
there are, how closely the browser port tracks the reference, what the learned
surrogate costs and buys. Numbers typed into HTML are true on the day they are
typed. These are read from the artefacts the test suite, the parity harness and
the training run leave behind, so the page is either current or visibly stale.

    python scripts/site_facts.py [--check]

``--check`` fails instead of writing, for continuous integration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "facts.json"
PAGE = ROOT / "web" / "index.html"
TIMES = "\N{MULTIPLICATION SIGN}"

# Wall-clock measurements are a property of the machine that took them, so
# ``--check`` reports drift in these but does not fail on it. Everything else
# is a deterministic function of the code and has to match.
VOLATILE = frozenset(
    {
        "parity-speedup",
        "parity-python-seconds",
        "parity-browser-seconds",
        "sim-full-rack",
        "surrogate-latency",
        "closed-form-latency",
    }
)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def count_physics_tests() -> int | None:
    """Ask pytest how many validation tests there are, rather than guessing."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_validation.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.splitlines()):
        if "test" in line and "collected" in line:
            head = line.split()[0]
            if head.isdigit():
                return int(head)
    return None


def mm(value: float) -> str:
    return f"{value:.0f} mm"


def build() -> dict[str, str]:
    facts: dict[str, str] = {}

    tests = count_physics_tests()
    if tests is not None:
        facts["physics-tests"] = str(tests)

    parity = _load(ROOT / "web" / "data" / "parity.json")
    if parity:
        worst = float(parity["worst_mm"])
        # Below a micrometre the reader wants the order of magnitude, not the
        # digits; above it, a plain number reads better than scientific form.
        mantissa, exponent = f"{worst:.1e}".split("e")
        facts["parity-worst"] = (
            f"{mantissa} {TIMES} 10<sup>{int(exponent)}</sup> mm"
            if worst < 1e-3
            else f"{worst:.4f} mm"
        )
        facts["parity-cases"] = str(parity["cases"])
        facts["parity-speedup"] = f"{float(parity['speedup']):.0f}{TIMES}"
        facts["parity-table-seconds"] = f"{float(parity['table_seconds']):.0f} s"
        facts["parity-python-seconds"] = f"{float(parity['python_seconds']):.0f} s"
        facts["parity-browser-seconds"] = f"{float(parity['browser_seconds']):.2f} s"

    metrics = _load(ROOT / "models" / "metrics.json")
    if metrics:
        models = metrics["models"]
        facts["surrogate-error"] = mm(models["cuenet"]["euclidean_mm"])
        facts["analytic-error"] = mm(models["analytic"]["euclidean_mm"])
        facts["gbm-error"] = mm(models["gbm"]["euclidean_mm"])
        facts["samples"] = f"{int(metrics['n_samples']):,}"
        contact = {row["object_ball_contact"]: row for row in metrics["by_object_ball_contact"]}
        if "yes" in contact:
            facts["contact-cuenet"] = mm(contact["yes"]["cuenet_obj_mm"])
            facts["contact-gbm"] = mm(contact["yes"]["gbm_obj_mm"])

    latency = _load(ROOT / "models" / "latency.json")
    if latency:
        full = float(latency["simulator_full_rack"]["mean_ms"])
        facts["sim-full-rack"] = f"{full / 1000:.1f} s" if full >= 1000 else f"{full:.0f} ms"
        facts["surrogate-latency"] = f"{float(latency['surrogate_onnx']['mean_ms']):.2f} ms"
        facts["closed-form-latency"] = f"{float(latency['closed_form']['mean_ms']):.2f} ms"

    return facts


def rewrite_page(html: str, facts: dict[str, str]) -> str:
    """
    Put the measured values into the page's own markup.

    The loader replaces them at runtime too, but only once the fetch resolves,
    and never at all over ``file://``. Writing them in as well means the text
    is never briefly wrong and never wrong at rest. Assumes a ``data-fact``
    element contains no further element of its own tag, which is true here.
    """

    def substitute(match: re.Match[str]) -> str:
        tag, attributes, key = match.group(1), match.group(2), match.group(3)
        if key not in facts:
            return match.group(0)
        return f"<{tag}{attributes}>{facts[key]}</{tag}>"

    pattern = re.compile(
        r"<(\w+)([^>]*\bdata-fact=\"([\w-]+)\"[^>]*)>.*?</\1>",
        re.DOTALL,
    )
    return pattern.sub(substitute, html)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)

    facts = build()
    if not facts:
        raise SystemExit("no measurements found; run `make all` and `make parity` first")

    payload = f"{json.dumps(facts, indent=2)}\n"
    html = PAGE.read_text()
    # Held to the reproducible figures only: timings differ from machine to
    # machine, so failing on those would fail for a reason nobody can fix.
    checkable = rewrite_page(html, {k: v for k, v in facts.items() if k not in VOLATILE})

    if args.check:
        rewritten = checkable
        committed = _load(OUT) or {}
        stale = [
            f"  {key}: committed {committed.get(key, '(missing)')!r}, measured {value!r}"
            for key, value in facts.items()
            if key not in VOLATILE and committed.get(key) != value
        ]
        if stale:
            print("web/data/facts.json is out of date; run `python scripts/site_facts.py`")
            print("\n".join(stale))
            raise SystemExit(1)
        if rewritten != html:
            print("web/index.html quotes numbers the measurements no longer support;")
            print("run `python scripts/site_facts.py`")
            raise SystemExit(1)
        drifted = sum(committed.get(k) != v for k, v in facts.items() if k in VOLATILE)
        print(
            f"web/data/facts.json is current ({len(facts) - len(VOLATILE)} fixed measurements"
            + (f", {drifted} timing figure(s) differ on this machine)" if drifted else ")")
        )
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload)
    rewritten = rewrite_page(html, facts)
    if rewritten != html:
        PAGE.write_text(rewritten)
        print(f"rewrote the figures quoted in {PAGE.relative_to(ROOT)}")
    print(f"wrote {len(facts)} measurements to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
