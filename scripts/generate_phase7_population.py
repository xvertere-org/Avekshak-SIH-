"""
Generate Phase 7 Physics-Constrained Synthetic Engine Population.

Generates a reproducible population ensemble, performs physical constraint validation,
partitions engines into train/val/test splits, and outputs a compact summary fixture.

Usage:
    python scripts/generate_phase7_population.py [--count 100] [--seed 42] [--output evidence/phase7_population_sample.json]
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.population import (
    PopulationConfig,
    PopulationGenerator,
    validate_population,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 7 synthetic engine population.")
    parser.add_argument("--count", type=int, default=100, help="Number of engine profiles to generate (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42).")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("evidence", "phase7_population_sample.json"),
        help="Path to output summary JSON.",
    )
    args = parser.parse_args()

    print(f"=== Generating Phase 7 Engine Population (count={args.count}, seed={args.seed}) ===")
    config = PopulationConfig(
        population_id="ROTAX_914_PHASE7",
        num_engines=args.count,
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    gen = PopulationGenerator(config)
    profiles = gen.generate_population()

    print(f"Generated {len(profiles)} engine profiles.")

    # Validation
    report = validate_population(profiles)
    print(f"Validation Result: {'ALL VALID' if report.is_fully_valid else 'FAILURES DETECTED'}")
    print(f"Valid Profiles: {report.valid_profiles} / {report.total_profiles}")

    if not report.is_fully_valid:
        print(f"Errors encountered: {len(report.errors)}")
        for e in report.errors[:10]:
            print(f"  - {e}")
        sys.exit(1)

    # Count splits
    splits = {"train": 0, "val": 0, "test": 0}
    for p in profiles:
        splits[p.split] = splits.get(p.split, 0) + 1
    print(f"Split breakdown: Train={splits['train']}, Val={splits['val']}, Test={splits['test']}")

    # Create compact sample fixture (first 3 engines + statistical summary)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sample_data: Dict[str, Any] = {
        "population_id": config.population_id,
        "total_engines": len(profiles),
        "seed": config.seed,
        "splits": splits,
        "validation": {
            "is_fully_valid": report.is_fully_valid,
            "valid_count": report.valid_profiles,
            "invalid_count": report.invalid_profiles,
        },
        "parameter_statistics": report.parameter_statistics,
        "clipping_counts": report.clipping_counts,
        "sample_profiles": [p.to_dict() for p in profiles[:3]],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2)

    print(f"Successfully wrote population summary and sample fixture to: {args.output}")


if __name__ == "__main__":
    main()
