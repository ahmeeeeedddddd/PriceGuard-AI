"""
classifier.py — Step 3: SVM Pricing-State Classifier
=====================================================
Trains an SVM with RBF kernel via GridSearchCV on labeled_products.csv,
persists the model and scaler to models/, and exposes a real-time
classify_product() inference function.

Public API
----------
train_classifier()                         → (svm, scaler, report_str)
classify_product(product_dict)             → (predicted_label, confidence_score)
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

warnings.simplefilter("ignore", FutureWarning)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
LABELED_CSV  = os.path.join(BASE_DIR, "data", "labeled_products.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURES     = ["price", "rating", "review_count", "discount_percentage", "stock_status"]
TARGET       = "cluster_label"
LABEL_NAMES  = {0: "budget", 1: "mid-range", 2: "premium"}
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────

def train_classifier():
    """
    Load labeled_products.csv, train an SVM with GridSearchCV, print
    evaluation metrics, and save svm_model.pkl + scaler.pkl.

    Returns
    -------
    svm    : trained SVC with probability=True
    scaler : fitted StandardScaler
    report : str — sklearn classification report
    """
    if not os.path.exists(LABELED_CSV):
        raise FileNotFoundError(
            f"Expected {LABELED_CSV} — run clustering first."
        )

    df = pd.read_csv(LABELED_CSV)
    df = df.dropna(subset=FEATURES + [TARGET])
    logger.info("Training SVM on %d samples …", len(df))

    X = df[FEATURES].values
    y = df[TARGET].astype(int).values

    # ── Train / test split (stratified 80/20) ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── GridSearchCV ─────────────────────────────────────────────────────────
    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
    }
    svm_base = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
    grid     = GridSearchCV(
        svm_base, param_grid, cv=5, n_jobs=-1, verbose=1, scoring="f1_weighted"
    )
    grid.fit(X_train, y_train)

    svm = grid.best_estimator_
    logger.info("Best SVM params: %s", grid.best_params_)
    print(f"[classifier] Best params: {grid.best_params_}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred  = svm.predict(X_test)
    report  = classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()))
    cm      = confusion_matrix(y_test, y_pred)

    print("\n── Classification Report ──")
    print(report)
    print("── Confusion Matrix ──")
    print(cm)
    logger.info("SVM classification report:\n%s", report)

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(svm,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved to %s | Scaler saved to %s.", MODEL_PATH, SCALER_PATH)
    print(f"[classifier] ✓  svm_model.pkl and scaler.pkl saved.")

    return svm, scaler, report


def _load_models():
    """
    Lazy-load the SVM and scaler from disk.
    Raises RuntimeError if models not found.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "Model files not found. Run pipeline.py once to train the models first."
        )
    svm    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return svm, scaler


def classify_product(product_dict: dict) -> tuple[str, float]:
    """
    Classify a single product dictionary into a pricing state.

    Parameters
    ----------
    product_dict : dict
        Must contain keys: price, rating, review_count,
                           discount_percentage, stock_status.

    Returns
    -------
    predicted_label : str   — "budget" | "mid-range" | "premium"
    confidence_score : float — max class probability from SVM
    """
    svm, scaler = _load_models()

    x = np.array([[
        float(product_dict.get("price",               0)),
        float(product_dict.get("rating",              3)),
        float(product_dict.get("review_count",        0)),
        float(product_dict.get("discount_percentage", 0)),
        float(product_dict.get("stock_status",        1)),
    ]])

    x_scaled = scaler.transform(x)
    pred_int = int(svm.predict(x_scaled)[0])
    proba    = svm.predict_proba(x_scaled)[0]
    confidence = float(proba.max())

    predicted_label = LABEL_NAMES.get(pred_int, "unknown")
    logger.info(
        "classify_product: price=%.2f → %s (conf=%.3f)",
        x[0, 0], predicted_label, confidence,
    )
    return predicted_label, confidence


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_classifier()
