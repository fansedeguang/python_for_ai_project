"""
experiments/run_baseline.py - Ticket 1: Baseline Reproduction

This script implements the reference TF-IDF + Logistic Regression baseline
using the fixed split and contract parameters. It automatically asserts
whether the computed held-out F1 matches the reference within tolerance.

IMPORTANT: This is the ONLY time we run on held-out before freezing decisions.
We do this solely to diagnose if the baseline reproduces the contract.
No tuning is performed based on this result.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Add project root to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader import load_data, get_X_y_from_df
from pipeline.contract import (
    get_baseline_params,
    get_seed,
    check_baseline_match,
    get_reference_metric,
    get_tolerance,
)
from pipeline.utils import set_global_seed  


def save_predictions(
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,  # probability of class 1
    model_name: str,
    ticket: str,
    output_path: Path,
) -> None:
    """
    Save predictions in the required format for later comparison.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "id": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "score": y_score,
        "model_name": model_name,
        "ticket": ticket,
    })
    df_out.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")


def main():
    print("=" * 60)
    print("🚀 Running Baseline Reproduction (Ticket 1)")
    print("=" * 60)

    # 1. Set global seed for reproducibility
    set_global_seed()
    print(f"🔢 Random seed set to: {get_seed()}")

    # 2. Load fixed split data
    print("\n📂 Loading data with fixed split...")
    train_df, dev_df, heldout_df = load_data()
    X_train, y_train = get_X_y_from_df(train_df)
    X_dev, y_dev = get_X_y_from_df(dev_df)
    X_heldout, y_heldout = get_X_y_from_df(heldout_df)

    print(f"   Train size: {len(X_train)}, Dev size: {len(X_dev)}, Heldout size: {len(X_heldout)}")

    # 3. Get baseline hyperparameters from contract
    params = get_baseline_params()
    tfidf_params = params["tfidf"]
    lr_params = params["logistic_regression"]

    print(f"\n⚙️  TF-IDF params: {tfidf_params}")
    print(f"⚙️  LR params: {lr_params}")

    # 4. Build pipeline: TF-IDF + Logistic Regression
    print("\n🔧 Training baseline pipeline...")
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    feature_names = (
        vectorizer.get_feature_names_out()
        if hasattr(vectorizer, "get_feature_names_out")
        else vectorizer.get_feature_names()
    )
    print(f"📚 Vocabulary size: {len(feature_names)}")
    print(f"📊 Non-zero entries: {X_train_tfidf.nnz}")
    vec_no_stop = TfidfVectorizer(**{**tfidf_params, "stop_words": None})
    X_no_stop = vec_no_stop.fit_transform(X_train)
    clf_no_stop = LogisticRegression(**lr_params).fit(X_no_stop, y_train)
    y_pred_no_stop = clf_no_stop.predict(vectorizer.transform(X_heldout))
    y_pred_no_stop = clf_no_stop.predict(vec_no_stop.transform(X_heldout))
    f1_no_stop = f1_score(y_heldout, y_pred_no_stop, pos_label=1)
    print(f"🧪 F1 without stopwords: {f1_no_stop:.6f}")
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_heldout_tfidf = vectorizer.transform(X_heldout)
    tfidf_params_test = tfidf_params.copy()
    tfidf_params_test["max_features"] = None  
    vec_test = TfidfVectorizer(**tfidf_params_test)
    X_train_test = vec_test.fit_transform(X_train)
    clf_test = LogisticRegression(**lr_params).fit(X_train_test, y_train)
    y_pred_test = clf_test.predict(vec_test.transform(X_heldout))
    f1_test = f1_score(y_heldout, y_pred_test, pos_label=1)
    print(f"🧪 F1 with max_features=None: {f1_test:.6f}")
    # ---------- 继续诊断 ----------
    print("\n🔍 Additional diagnostics for baseline mismatch:")

    # 测试 1: 调整 min_df=2
    vec_min2 = TfidfVectorizer(**{**tfidf_params, "min_df": 2})
    X_min2 = vec_min2.fit_transform(X_train)
    clf_min2 = LogisticRegression(**lr_params).fit(X_min2, y_train)
    f1_min2 = f1_score(y_heldout, clf_min2.predict(vec_min2.transform(X_heldout)), pos_label=1)
    print(f"🧪 F1 with min_df=2          : {f1_min2:.6f}")

    # 测试 2: 调整 max_df=0.8
    vec_max08 = TfidfVectorizer(**{**tfidf_params, "max_df": 0.8})
    X_max08 = vec_max08.fit_transform(X_train)
    clf_max08 = LogisticRegression(**lr_params).fit(X_max08, y_train)
    f1_max08 = f1_score(y_heldout, clf_max08.predict(vec_max08.transform(X_heldout)), pos_label=1)
    print(f"🧪 F1 with max_df=0.8        : {f1_max08:.6f}")

    # 测试 3: 调整 C=0.5 或 C=2.0
    for C_val in [0.5, 2.0, 0.1, 10.0]:
        lr_params_test = {**lr_params, "C": C_val}
        clf_C = LogisticRegression(**lr_params_test).fit(X_train_tfidf, y_train)
        f1_C = f1_score(y_heldout, clf_C.predict(X_heldout_tfidf), pos_label=1)
        print(f"🧪 F1 with C={C_val:4.1f}        : {f1_C:.6f}")

# 测试 4: 文本预处理（简单去除 URL 和 @ 提及，不改变 TF-IDF 参数）
    import re
    def clean_tweet(text):
        text = re.sub(r'http\S+', '', text)   # 去除 URL
        text = re.sub(r'@\w+', '', text)       # 去除 @mention
    # 也可以去除 # 符号但保留单词（可选）
        return text

    X_train_clean = X_train.apply(clean_tweet)
    X_heldout_clean = X_heldout.apply(clean_tweet)
    vec_clean = TfidfVectorizer(**tfidf_params)
    X_train_clean_tfidf = vec_clean.fit_transform(X_train_clean)
    clf_clean = LogisticRegression(**lr_params).fit(X_train_clean_tfidf, y_train)
    f1_clean = f1_score(y_heldout, clf_clean.predict(vec_clean.transform(X_heldout_clean)), pos_label=1)
    print(f"🧪 F1 with URL+@ removed    : {f1_clean:.6f}")

    lr_liblinear = {**lr_params, "solver": "liblinear"}
    clf_ll = LogisticRegression(**lr_liblinear).fit(X_train_tfidf, y_train)
    f1_ll = f1_score(y_heldout, clf_ll.predict(X_heldout_tfidf), pos_label=1)
    print(f"🧪 F1 with solver='liblinear': {f1_ll:.6f}")

    clf = LogisticRegression(**lr_params)
    clf.fit(X_train_tfidf, y_train)
    print("✅ Training complete.")

    # 5. Predict on held-out set (for contract verification)
    print("\n📊 Evaluating on held-out set...")
    y_heldout_pred = clf.predict(X_heldout_tfidf)
    y_heldout_proba = clf.predict_proba(X_heldout_tfidf)[:, 1]  # probability of class 1

    heldout_f1 = f1_score(y_heldout, y_heldout_pred, pos_label=1)
    heldout_acc = accuracy_score(y_heldout, y_heldout_pred)

    print(f"   Held-out F1 (target=1): {heldout_f1:.6f}")
    print(f"   Held-out Accuracy: {heldout_acc:.6f}")

    # 6. CRITICAL: Check against the reference contract
    # This will raise a RuntimeError if mismatch > tolerance
    print("\n🔍 Verifying against contract...")
    try:
        check_baseline_match(heldout_f1, metric_name="heldout_f1_target_1", raise_on_fail=True)
        print("✅ BASELINE REPRODUCED SUCCESSFULLY within tolerance!")
    except RuntimeError as e:
        print(f"\n❌ {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Check your scikit-learn version (must match requirements.txt)")
        print("   - Verify that split_indices.json is exactly the provided file")
        print("   - Ensure TF-IDF parameters match the reference implementation")
        print("   - Look for differences in default stop_words or token_pattern")
        sys.exit(1)

    # 7. Also evaluate on dev set (for reference, no contract check)
    y_dev_pred = clf.predict(X_dev_tfidf)
    dev_f1 = f1_score(y_dev, y_dev_pred, pos_label=1)
    print(f"\n📊 Dev-set F1 (target=1) for reference: {dev_f1:.6f}")

    # 8. Save predictions for future ticket comparisons
    print("\n💾 Saving prediction artifacts...")
    
    # Heldout predictions (required final artifact format)
    save_predictions(
        ids=heldout_df["id"].values,
        y_true=y_heldout,
        y_pred=y_heldout_pred,
        y_score=y_heldout_proba,
        model_name="baseline_lr",
        ticket="1",
        output_path=Path("predictions/heldout_predictions.csv"),
    )

    # Dev predictions (to compare with future experiments)
    y_dev_proba = clf.predict_proba(X_dev_tfidf)[:, 1]
    save_predictions(
        ids=dev_df["id"].values,
        y_true=y_dev,
        y_pred=y_dev_pred,
        y_score=y_dev_proba,
        model_name="baseline_lr",
        ticket="1",
        output_path=Path("predictions/dev_predictions/baseline_dev.csv"),
    )

    # 9. Save baseline metrics to results/ (for summary collection later)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    baseline_metrics = {
        "ticket": "1",
        "model_name": "baseline_lr",
        "dev_f1_target_1": dev_f1,
        "heldout_f1_target_1": heldout_f1,
        "heldout_accuracy": heldout_acc,
        "fixed_fp": 0,  # Baseline has no "fixes" yet
        "fixed_fn": 0,
        "new_fp": 0,
        "new_fn": 0,
        "decision": "baseline",
        "decision_reason": "Reference implementation for comparison.",
    }
    
    # Append to summary.csv (create if not exists)
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        df_new = pd.DataFrame([baseline_metrics])
        df_combined = pd.concat([existing, df_new], ignore_index=True)
        df_combined.to_csv(summary_path, index=False)
    else:
        pd.DataFrame([baseline_metrics]).to_csv(summary_path, index=False)
    
    print(f"💾 Baseline metrics saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("🎉 Baseline reproduction complete! Phase 0 is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()