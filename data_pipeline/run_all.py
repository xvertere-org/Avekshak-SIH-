"""
Master preprocessing runner — executes all dataset adapters and generates reports.

Usage: python -m data_pipeline.run_all
"""

import json
import os
import sys
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_all():
    """Run all dataset preprocessing adapters and collect results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "preprocessing_version": "0.1.0",
        "datasets": {},
    }

    # 1. C-MAPSS
    print("=" * 60)
    print("Processing C-MAPSS...")
    try:
        from data_pipeline.cmapss.adapter import process_and_save as cmapss_process
        results["datasets"]["cmapss"] = cmapss_process()
        print(f"  [OK] C-MAPSS: {results['datasets']['cmapss']['total_train_rows']} train rows, "
              f"{results['datasets']['cmapss']['total_test_rows']} test rows")
    except Exception as e:
        results["datasets"]["cmapss"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] C-MAPSS FAILED: {e}")

    # 2. CWRU
    print("=" * 60)
    print("Processing CWRU...")
    try:
        from data_pipeline.cwru.adapter import process_and_save as cwru_process
        results["datasets"]["cwru"] = cwru_process()
        print(f"  [OK] CWRU: {results['datasets']['cwru']['total_rows']} rows, "
              f"{results['datasets']['cwru']['files_processed']} files")
    except Exception as e:
        results["datasets"]["cwru"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] CWRU FAILED: {e}")

    # 3. NUST
    print("=" * 60)
    print("Processing NUST...")
    try:
        from data_pipeline.nust.adapter import process_and_save as nust_process
        results["datasets"]["nust"] = nust_process()
        print(f"  [OK] NUST: {results['datasets']['nust']['total_rows']} rows, "
              f"{results['datasets']['nust']['files_processed']} files")
    except Exception as e:
        results["datasets"]["nust"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] NUST FAILED: {e}")

    # 4. FEMTO (large — may take time)
    print("=" * 60)
    print("Processing FEMTO (this may take several minutes)...")
    try:
        from data_pipeline.femto.adapter import process_and_save as femto_process
        results["datasets"]["femto"] = femto_process()
        print(f"  [OK] FEMTO: {results['datasets']['femto']['total_train_rows']} train rows, "
              f"{results['datasets']['femto']['total_test_rows']} test rows")
    except Exception as e:
        results["datasets"]["femto"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] FEMTO FAILED: {e}")

    # 5. NASA Battery
    print("=" * 60)
    print("Processing NASA Battery...")
    try:
        from data_pipeline.nasa_battery.adapter import process_and_save as battery_process
        results["datasets"]["nasa_battery"] = battery_process()
        print(f"  [OK] NASA Battery: {results['datasets']['nasa_battery'].get('total_cycles_extracted', 0)} cycles, "
              f"{results['datasets']['nasa_battery'].get('batteries_processed', 0)} batteries")
    except Exception as e:
        results["datasets"]["nasa_battery"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] NASA Battery FAILED: {e}")

    # 6. Paderborn
    print("=" * 60)
    print("Processing Paderborn...")
    try:
        from data_pipeline.paderborn.adapter import process_and_save as paderborn_process
        results["datasets"]["paderborn"] = paderborn_process()
        print(f"  [OK] Paderborn: {results['datasets']['paderborn'].get('total_windows_extracted', 0)} windows, "
              f"{results['datasets']['paderborn'].get('mat_files_processed', 0)} files")
    except Exception as e:
        results["datasets"]["paderborn"] = {"status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  [FAIL] Paderborn FAILED: {e}")

    # 7. BASIC — document as unavailable
    results["datasets"]["basic"] = {
        "status": "UNAVAILABLE",
        "usage": "none",
        "note": "BASIC dataset not downloaded. No data fabricated.",
    }
    print("=" * 60)
    print("BASIC: UNAVAILABLE (documented, no data fabricated)")

    # Save results
    os.makedirs("reports", exist_ok=True)
    with open("reports/preprocessing_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n" + "=" * 60)
    print(f"Results saved to reports/preprocessing_results.json")

    return results


if __name__ == "__main__":
    run_all()
