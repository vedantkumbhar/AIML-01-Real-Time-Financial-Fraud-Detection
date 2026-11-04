
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "candidate_model": PROJECT_ROOT / "models/candidate_fraud_model.json",
    "production_model": PROJECT_ROOT / "models/realtime_fraud_model.json",
    "candidate_features": PROJECT_ROOT / "models/candidate_feature_columns.pkl",
    "production_features": PROJECT_ROOT / "models/realtime_feature_columns.pkl",
    "candidate_metrics": PROJECT_ROOT / "reports/candidate_model_metrics.json",
    "production_metrics": PROJECT_ROOT / "reports/realtime_model_metrics.json",
    "retraining_request": PROJECT_ROOT / "reports/evidently/retraining_request.json",
}

BACKUP_DIR = PROJECT_ROOT / "models/backups"
PROMOTION_LOG = PROJECT_ROOT / "reports/model_promotion_log.json"


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def validate_candidate():
    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    production = load_json(FILES["production_metrics"])
    candidate = load_json(FILES["candidate_metrics"])

    production_features = load_pickle(FILES["production_features"])
    candidate_features = load_pickle(FILES["candidate_features"])

    if production_features != candidate_features:
        raise RuntimeError("Feature compatibility check failed.")

    for metric in ["pr_auc", "precision", "recall", "f1_score"]:
        if float(candidate[metric]) < float(production[metric]):
            raise RuntimeError(f"Candidate failed validation: {metric}")

    request = load_json(FILES["retraining_request"])

    if request.get("status") != "RETRAINING_REQUESTED":
        raise RuntimeError("No pending retraining request found.")

    print("All promotion checks passed.")


def promote_candidate():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)

    production_files = [
        FILES["production_model"],
        FILES["production_features"],
        FILES["production_metrics"],
    ]

    # Back up all existing production files first.
    for path in production_files:
        shutil.copy2(path, backup_path / path.name)

    print("Production backup created:", backup_path)

    try:
        # Promote candidate files.
        shutil.copy2(
            FILES["candidate_model"],
            FILES["production_model"]
        )

        shutil.copy2(
            FILES["candidate_features"],
            FILES["production_features"]
        )

        shutil.copy2(
            FILES["candidate_metrics"],
            FILES["production_metrics"]
        )

        # Update retraining request status.
        request = load_json(FILES["retraining_request"])
        request["status"] = "PROMOTED"
        request["promoted_at"] = timestamp

        with open(FILES["retraining_request"], "w") as file:
            json.dump(request, file, indent=4)

        promotion_record = {
            "status": "PROMOTED",
            "timestamp": timestamp,
            "backup_directory": str(backup_path),
            "model": "Candidate XGBoost",
            "metrics": load_json(FILES["candidate_metrics"]),
        }

        with open(PROMOTION_LOG, "w") as file:
            json.dump(promotion_record, file, indent=4)

        print("\nCANDIDATE MODEL PROMOTED SUCCESSFULLY")
        print("Production model updated.")
        print("Previous model backup:", backup_path)
        print("Promotion log:", PROMOTION_LOG)

    except Exception:
        # Restore production files if promotion fails.
        for path in production_files:
            backup_file = backup_path / path.name
            if backup_file.exists():
                shutil.copy2(backup_file, path)

        raise


def main():
    print("Starting controlled model promotion...\n")

    validate_candidate()
    promote_candidate()


if __name__ == "__main__":
    main()
