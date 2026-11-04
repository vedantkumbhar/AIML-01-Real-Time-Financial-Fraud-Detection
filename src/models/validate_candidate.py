import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_METRICS = PROJECT_ROOT / "reports/realtime_model_metrics.json"
CANDIDATE_METRICS = PROJECT_ROOT / "reports/candidate_model_metrics.json"

PRODUCTION_FEATURES = PROJECT_ROOT / "models/realtime_feature_columns.pkl"
CANDIDATE_FEATURES = PROJECT_ROOT / "models/candidate_feature_columns.pkl"


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def main():
    print("Validating candidate model...\n")

    production = load_json(PRODUCTION_METRICS)
    candidate = load_json(CANDIDATE_METRICS)

    production_features = load_pickle(PRODUCTION_FEATURES)
    candidate_features = load_pickle(CANDIDATE_FEATURES)

    # Check feature compatibility
    features_match = production_features == candidate_features

    print("Feature compatibility:", "PASS" if features_match else "FAIL")

    # Compare model performance
    metrics = ["pr_auc", "precision", "recall", "f1_score"]

    performance_pass = True

    print("\nPerformance comparison:")

    for metric in metrics:
        old = float(production[metric])
        new = float(candidate[metric])

        passed = new >= old

        if not passed:
            performance_pass = False

        print(
            f"{metric}: Production={old:.6f} | "
            f"Candidate={new:.6f} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\n" + "=" * 45)

    if features_match and performance_pass:
        print("VALIDATION PASSED")
        print("Candidate is eligible for promotion.")
    else:
        print("VALIDATION FAILED")
        print("Candidate must NOT be promoted.")

    print("\nProduction model remains unchanged.")


if __name__ == "__main__":
    main()