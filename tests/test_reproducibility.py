import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.contract import get_baseline_params, get_reference_metric, get_tolerance
from pipeline.loader import load_data, get_X_y_from_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def test_contract_baseline_params_reproduce_reference_f1():
    train_df, _, heldout_df = load_data()
    X_train, y_train = get_X_y_from_df(train_df)
    X_heldout, y_heldout = get_X_y_from_df(heldout_df)

    params = get_baseline_params()
    tfidf_params = params["tfidf"]
    lr_params = params["logistic_regression"]

    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(**lr_params).fit(X_train_tfidf, y_train)
    y_pred = clf.predict(vectorizer.transform(X_heldout))
    heldout_f1 = f1_score(y_heldout, y_pred, pos_label=1)

    reference = get_reference_metric()
    tolerance = get_tolerance()

    assert heldout_f1 >= reference - tolerance
    assert heldout_f1 <= reference + tolerance
