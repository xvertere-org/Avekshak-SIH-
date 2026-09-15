#!/usr/bin/env python3
"""
Phase 12: Automated Anti-Fabrication & Claims Integrity Linter.

Audits repository documentation, code docstrings, UI labels, and evidence files
for prohibited overstatements, uncalibrated probability claims, unsupported airworthiness/certification
assertions, and stale terminology.

Two-pass architecture:
1. Lexical Pattern Matcher: flags high-risk phrases across scanned files.
2. Contextual Allowlist Filter: evaluates surrounding clause to permit legitimate negative disclaimers,
   engineering explanations, and properly qualified limitations.

Acceptance Criterion: ZERO UNRESOLVED BLOCKER and ZERO UNRESOLVED HIGH findings.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class FindingSeverity(str, Enum):
    BLOCKER = "BLOCKER"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    ACCEPTABLE_DISCLAIMER = "ACCEPTABLE_DISCLAIMER"


@dataclass
class ClaimFinding:
    file_path: str
    line_number: int
    matched_text: str
    surrounding_context: str
    pattern_name: str
    severity: FindingSeverity
    justification: str = ""


# High-Risk Search Patterns (Regex, Case-Insensitive)
HIGH_RISK_PATTERNS = [
    # 1. Certification & Airworthiness
    (
        "CERTIFICATION_AIRWORTHINESS",
        re.compile(r"\b(certified\s+for\s+flight|flight[\s-]certified|airworthy|airworthiness\s+directive|do-178c\s+compliant)\b", re.IGNORECASE),
        FindingSeverity.BLOCKER,
    ),
    # 2. Unsupported Probabilities
    (
        "FAILURE_PROBABILITY",
        re.compile(r"\b(failure\s+probability|probability\s+of\s+failure|survival\s+probability|calibrated\s+probability)\b", re.IGNORECASE),
        FindingSeverity.BLOCKER,
    ),
    # 3. Hardware MTBF / MTTF Claims
    (
        "MTBF_MTTF",
        re.compile(r"\b(mtbf|mttf)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 4. Hard Real-Time / Embedded Claims
    (
        "HARD_REAL_TIME",
        re.compile(r"\b(hard\s+real[\s-]time|guaranteed\s+real[\s-]time|avionics\s+rtos)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 5. Production / Flight Readiness
    (
        "FLIGHT_READINESS",
        re.compile(r"\b(flight[\s-]ready|production[\s-]ready)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 6. Real Flight Validation
    (
        "REAL_FLIGHT_VALIDATION",
        re.compile(r"\b(validated\s+on\s+real\s+flight|flight[\s-]tested\s+on\s+uav|real\s+fleet\s+validation)\b", re.IGNORECASE),
        FindingSeverity.BLOCKER,
    ),
    # 7. Unqualified Causal Inference
    (
        "CAUSAL_INFERENCE",
        re.compile(r"\b(causal\s+degradation\s+tracking|causal\s+inference\s+engine)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 8. Unqualified RUL / TBO
    (
        "UNQUALIFIED_RUL",
        re.compile(r"\b(engine\s+remaining\s+life|tbo\s+prediction|actual\s+hours\s+remaining)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 9. Stale Engine Identity
    (
        "STALE_ENGINE_IDENTITY",
        re.compile(r"\b(ROT912|Rotax\s+912)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
    # 10. Autonomous AI Decisions
    (
        "AUTONOMOUS_AI_DECISION",
        re.compile(r"\b(autonomous\s+ai\s+controller|deep\s+learning\s+twin\s+runtime)\b", re.IGNORECASE),
        FindingSeverity.HIGH,
    ),
]

# Negative qualifiers that make a high-risk phrase acceptable as a disclaimer
ALLOWLIST_NEGATION_PATTERNS = [
    re.compile(r"\b(not|never|neither|no|non|disclaim|without|absence\s+of|is\s+not|are\s+not|strictly\s+not|does\s+not)\b", re.IGNORECASE),
    re.compile(r"\b(out\s+of\s+scope|cannot\s+be|prohibited|not\s+certified|not\s+claimed|not\s+flight[\s-]tested)\b", re.IGNORECASE),
    re.compile(r"\b(rejected|omitted|abandoned|disproved|unsupported)\b", re.IGNORECASE),
    re.compile(r"\b(academic|prototype|research|heuristic|disclaimer|limitation|warning|caution|note)\b", re.IGNORECASE),
    re.compile(r"\b(soft\s+real[\s-]time|host[\s-]side|measured\s+host|synthetic|simulated|replayed|model[\s-]defined)\b", re.IGNORECASE),
    re.compile(r"\b(cross[\s-]check|reference\s+standard|historical|naturally\s+aspirated|reference_anchor|reference_source)\b", re.IGNORECASE),
]

# Directories and file patterns to ignore
IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".gemini",
    "venv",
    ".venv",
    "node_modules",
    "scratch",
    "data",
}

SCANNED_EXTENSIONS = {".py", ".md", ".json"}


def is_context_allowlisted(context: str, matched_text: str, pattern_name: str) -> Tuple[bool, str]:
    """
    Check if the surrounding context constitutes a valid negative disclaimer,
    engineering boundary, or properly qualified limitation.
    """
    context_lower = context.lower()

    # Rule 1: Explicit negative disclaimers
    if any(neg.search(context) for neg in ALLOWLIST_NEGATION_PATTERNS):
        return True, "Context contains explicit negative qualification or engineering disclaimer."

    # Rule 2: Rotax 912 comparative / historical reference anchor
    if pattern_name == "STALE_ENGINE_IDENTITY":
        if any(w in context_lower for w in ["naturally aspirated", "comparison", "predecessor", "anchor", "reference", "baseline", "uls"]):
            return True, "Context is an intentional comparative or historical reference anchor to Rotax 912."

    # Rule 3: Parameter cross-check against EASA TCDS
    if "tcds" in context_lower and "cross-check" in context_lower:
        return True, "Context correctly identifies EASA TCDS as reference specification."

    return False, ""


def scan_file(file_path: Path, repo_root: Path) -> List[ClaimFinding]:
    """Scan a single file for claim violations."""
    findings: List[ClaimFinding] = []
    rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

    # Skip files that are this script itself, or the test file auditing this script
    if rel_path in {"scripts/audit_claims.py", "tests/test_phase12_claims_audit.py"}:
        return []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return findings

    for line_idx, line in enumerate(lines, start=1):
        clean_line = line.strip()
        if not clean_line:
            continue

        for pattern_name, regex, default_severity in HIGH_RISK_PATTERNS:
            match = regex.search(clean_line)
            if match:
                matched_text = match.group(0)
                # Expand context (previous line + current line + next line)
                start_c = max(0, line_idx - 2)
                end_c = min(len(lines), line_idx + 1)
                context = " ".join(l.strip() for l in lines[start_c:end_c])

                allowlisted, reason = is_context_allowlisted(context, matched_text, pattern_name)
                severity = FindingSeverity.ACCEPTABLE_DISCLAIMER if allowlisted else default_severity

                findings.append(
                    ClaimFinding(
                        file_path=rel_path,
                        line_number=line_idx,
                        matched_text=matched_text,
                        surrounding_context=clean_line,
                        pattern_name=pattern_name,
                        severity=severity,
                        justification=reason,
                    )
                )

    return findings


def run_audit(repo_root: Path) -> Dict[str, Any]:
    """Execute full claims audit across the repository."""
    all_findings: List[ClaimFinding] = []

    for root, dirs, files in os.walk(repo_root):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        for fname in files:
            p = Path(root) / fname
            if p.suffix in SCANNED_EXTENSIONS:
                all_findings.extend(scan_file(p, repo_root))

    unresolved_blockers = [f for f in all_findings if f.severity == FindingSeverity.BLOCKER]
    unresolved_highs = [f for f in all_findings if f.severity == FindingSeverity.HIGH]
    unresolved_mediums = [f for f in all_findings if f.severity == FindingSeverity.MEDIUM]
    unresolved_lows = [f for f in all_findings if f.severity == FindingSeverity.LOW]
    acceptable_disclaimers = [f for f in all_findings if f.severity == FindingSeverity.ACCEPTABLE_DISCLAIMER]

    status = "PASS" if len(unresolved_blockers) == 0 and len(unresolved_highs) == 0 else "FAIL"

    return {
        "status": status,
        "total_findings": len(all_findings),
        "unresolved_blocker_count": len(unresolved_blockers),
        "unresolved_high_count": len(unresolved_highs),
        "unresolved_medium_count": len(unresolved_mediums),
        "unresolved_low_count": len(unresolved_lows),
        "acceptable_disclaimer_count": len(acceptable_disclaimers),
        "unresolved_blockers": [f.__dict__ for f in unresolved_blockers],
        "unresolved_highs": [f.__dict__ for f in unresolved_highs],
        "unresolved_mediums": [f.__dict__ for f in unresolved_mediums],
        "acceptable_disclaimers_sample": [f.__dict__ for f in acceptable_disclaimers[:10]],
    }


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Phase 12 Anti-Fabrication & Claims Integrity Linter")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results = run_audit(repo_root)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 70)
        print("PHASE 12: CLAIMS INTEGRITY & ANTI-FABRICATION AUDIT REPORT")
        print("=" * 70)
        print(f"Audit Status:               {results['status']}")
        print(f"Total Matches Scanned:      {results['total_findings']}")
        print(f"Acceptable Disclaimers:     {results['acceptable_disclaimer_count']}")
        print(f"Unresolved BLOCKER Claims:  {results['unresolved_blocker_count']}")
        print(f"Unresolved HIGH Claims:     {results['unresolved_high_count']}")
        print(f"Unresolved MEDIUM Claims:   {results['unresolved_medium_count']}")
        print("=" * 70)

        if results["unresolved_blockers"]:
            print("\n[!] UNRESOLVED BLOCKER CLAIMS:")
            for f in results["unresolved_blockers"]:
                print(f"  - {f['file_path']}:{f['line_number']} [{f['pattern_name']}] '{f['matched_text']}'")
                print(f"    Line: {f['surrounding_context']}")

        if results["unresolved_highs"]:
            print("\n[!] UNRESOLVED HIGH CLAIMS:")
            for f in results["unresolved_highs"]:
                print(f"  - {f['file_path']}:{f['line_number']} [{f['pattern_name']}] '{f['matched_text']}'")
                print(f"    Line: {f['surrounding_context']}")

        if results["status"] == "PASS":
            print("\n[OK] CLAIMS INTEGRITY AUDIT PASSED: ZERO UNRESOLVED BLOCKER / HIGH CLAIMS.")
        else:
            print("\n[FAIL] AUDIT FAILED: UNRESOLVED CLAIMS DETECTED. CORRECTIONS REQUIRED.")

    sys.exit(0 if results["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
