"""Trained situation classifier: does a supervised model route situations better than rules or nearest neighbour?

    python3 train_router.py            # train, cross-validate, test on the blind sets, save models/situation_clf.joblib
    python3 train_router.py --show     # also print every test miss

Task: message -> one of the 27 playbook ids, or "none" (civil matter, definition question, nothing fits).
Features: bge-small-en-v1.5 sentence embeddings (384 numbers, L2-normalised), the same model the app already loads.
Model: multinomial logistic regression (class-balanced), compared with 1-nearest-neighbour on the same features.

Train: the 115 example situations written into data/playbooks.json + eval_situations.jsonl and eval_situations2.jsonl
(the development sets). Test: eval_situations3/4/5 (65 situations), never used for training or for choosing C.
Note: sets 3-5 were later used to fix the trigger rules, so the rules' scores on them are optimistic; their first-run
(blind) rules scores were 11/20, 8/20 and 14/25. The classifier never saw them.
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from llm_compare import wilson
from src.playbooks import load_playbooks, match

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT = os.path.join(ROOT, "models", "situation_clf.npz")
DEV = ["data/eval_situations.jsonl", "data/eval_situations2.jsonl"]
TEST = ["data/eval_situations3.jsonl", "data/eval_situations4.jsonl", "data/eval_situations5.jsonl"]


def read(path):
    out = []
    for line in open(os.path.join(ROOT, path), encoding="utf-8"):
        if line.strip():
            q = json.loads(line)
            exp = q["expect"] if isinstance(q["expect"], list) else [q["expect"]]
            out.append((q["query"], exp))
    return out


def encoder():
    from sentence_transformers import SentenceTransformer
    from src.retrieval import DEFAULT_DENSE_MODEL
    m = SentenceTransformer(DEFAULT_DENSE_MODEL)
    return lambda texts: m.encode(list(texts), normalize_embeddings=True, batch_size=64)


def ok(pred, exp):
    return pred in exp


class Knn:
    """1-nearest neighbour by cosine similarity (what the app does today, without its threshold)."""
    def fit(self, X, y):
        self.X, self.y = X, np.array(y)
        return self

    def ranked(self, X):
        sims = X @ self.X.T
        out = []
        for row in sims:
            best = {}
            for s, lab in zip(row, self.y):
                best[lab] = max(best.get(lab, -1), s)
            out.append(sorted(best, key=lambda k: -best[k]))
        return out


def lr_ranked(clf, X):
    P = clf.predict_proba(X)
    return [list(clf.classes_[np.argsort(-p)]) for p in P], P.max(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    pbs = load_playbooks()
    ex = [(e, pb.id) for pb in pbs for e in pb.examples]
    dev = [(q, exp[0]) for path in DEV for q, exp in read(path)]          # dev labels: first acceptable id
    test = [(q, exp, os.path.basename(p)) for p in TEST for q, exp in read(p)]
    enc = encoder()

    X_ex = enc([t for t, _ in ex]); y_ex = [l for _, l in ex]
    X_dev = enc([t for t, _ in dev]); y_dev = [l for _, l in dev]
    X_tr = np.vstack([X_ex, X_dev]); y_tr = y_ex + y_dev
    X_te = enc([q for q, _, _ in test])
    print(f"train: {len(ex)} playbook examples + {len(dev)} dev situations = {len(y_tr)} texts, "
          f"{len(set(y_tr))} classes (incl. 'none': {y_tr.count('none')} texts) | test: {len(test)} blind situations\n")

    # choose C by 5-fold cross-validation on the training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_arr = np.array(y_tr)
    counts = {c: y_tr.count(c) for c in set(y_tr)}
    keep = np.array([counts[c] >= 5 for c in y_tr])          # stratified CV needs >= 5 per class
    best_c, best_acc = None, -1
    print("5-fold cross-validation on training data (logistic regression):")
    for C in [0.5, 1, 2, 5, 10, 20, 50]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        pred = cross_val_predict(clf, X_tr[keep], y_arr[keep], cv=cv)
        acc = (pred == y_arr[keep]).mean()
        print(f"  C={C:<4} accuracy {acc:.1%}  macro-F1 {f1_score(y_arr[keep], pred, average='macro'):.3f}")
        if acc > best_acc:
            best_c, best_acc = C, acc
    print(f"  -> C={best_c}\n")

    systems = {}
    lr_ex = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(X_ex, y_ex)
    lr_all = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(X_tr, y_tr)
    knn_ex, knn_all = Knn().fit(X_ex, y_ex), Knn().fit(X_tr, y_tr)
    systems["1-NN, examples only (current suggestions)"] = knn_ex.ranked(X_te)
    systems["1-NN, examples + dev"] = knn_all.ranked(X_te)
    systems["Logistic regression, examples only"] = lr_ranked(lr_ex, X_te)[0]
    ranked_lr, conf = lr_ranked(lr_all, X_te)
    systems["Logistic regression, examples + dev"] = ranked_lr

    # rules first (as in the app), classifier only when no rule fires
    rule_top = []
    for q, _, _ in test:
        m = match(pbs, q)
        rule_top.append(m[0].id if m else None)
    systems["Rules alone (seen*)"] = [[r or "none"] for r in rule_top]
    systems["Rules, then logistic regression (seen*)"] = [[r] if r else lr for r, lr in zip(rule_top, ranked_lr)]

    sets = sorted({s for _, _, s in test})
    print("| System | " + " | ".join(sets) + " | All 65: top-1 (95% CI) | top-3 | macro-F1 |")
    print("|---|" + "---|" * (len(sets) + 3))
    for name, ranked in systems.items():
        cells = []
        for s in sets:
            idx = [i for i, t in enumerate(test) if t[2] == s]
            k = sum(ok(ranked[i][0], test[i][1]) for i in idx)
            cells.append(f"{k}/{len(idx)}")
        k1 = sum(ok(r[0], t[1]) for r, t in zip(ranked, test))
        k3 = sum(any(ok(x, t[1]) for x in r[:3]) for r, t in zip(ranked, test))
        lo, hi = wilson(k1, len(test))
        gold = [t[1][0] if r[0] not in t[1] else r[0] for r, t in zip(ranked, test)]
        f1 = f1_score(gold, [r[0] for r in ranked], average="macro")
        top3 = f"{k3/len(test):.0%}" if len(ranked[0]) > 1 or "Rules alone" not in name else "–"
        print(f"| {name} | " + " | ".join(cells) + f" | {k1}/{len(test)} {k1/len(test):.0%} ({lo:.0%}–{hi:.0%}) | {top3} | {f1:.3f} |")
        if a.show and name == "Logistic regression, examples + dev":
            for r, t in zip(ranked, test):
                if not ok(r[0], t[1]):
                    print(f"    MISS expect={t[1]} got={r[0]} | {t[0]}")
    print("\n* rules were fixed after seeing sets 3-5; first-run rules scores were 11/20, 8/20, 14/25 (33/65 = 51%).")

    # confidence: is the classifier's probability useful for 'answer vs ask'?
    right = np.array([ok(r[0], t[1]) for r, t in zip(ranked_lr, test)])
    print("\nLogistic-regression confidence on the test set (answer only when probability >= t):")
    for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
        sel = conf >= t
        if sel.sum():
            print(f"  t={t:.1f}: answers {sel.mean():.0%} of messages, accuracy when answering {right[sel].mean():.0%}")

    # the app's threshold is chosen on cross-validated probabilities of the TRAINING data, not on the test set:
    # the lowest t at which cross-validated accuracy, when answering, reaches 95% (a wrong plan is worse than a question)
    P = cross_val_predict(LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced"),
                          X_tr[keep], y_arr[keep], cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y_arr[keep])))
    cv_pred, cv_conf = classes[P.argmax(1)], P.max(1)
    cv_right = cv_pred == y_arr[keep]
    threshold = 0.6
    print("\nCross-validated confidence on the training data (used to choose the app's threshold):")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        sel = cv_conf >= t
        acc = cv_right[sel].mean() if sel.sum() else 0
        print(f"  t={t:.1f}: answers {sel.mean():.0%}, accuracy when answering {acc:.0%}")
        if acc >= 0.95 and t < threshold:
            threshold = t
    sel = conf >= threshold
    print(f"  -> threshold {threshold:.1f}; on the blind test set it answers {sel.mean():.0%} "
          f"with accuracy {right[sel].mean():.0%}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    # plain weights (no pickle), so any scikit-learn version, or none, can load them: p = softmax(x W^T + b)
    np.savez(MODEL_OUT, coef=lr_all.coef_, intercept=lr_all.intercept_, classes=lr_all.classes_.astype(str),
             threshold=threshold, C=best_c, n_train=len(y_tr))
    Xc = X_te[:5]
    z = Xc @ lr_all.coef_.T + lr_all.intercept_
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    assert np.allclose(p, lr_all.predict_proba(Xc), atol=1e-6), "saved weights do not reproduce predict_proba"
    print(f"\nsaved {os.path.relpath(MODEL_OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
