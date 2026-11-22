"""Compare two scaled models using training-only selection and a separate holdout."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def estimator(name):
    classifier = SVC(C=3) if name == "SVC" else KNeighborsClassifier(n_neighbors=3)
    return make_pipeline(StandardScaler(), classifier)


def evaluation_groups(frame):
    """Link source groups connected by identical resized images before deduplication."""
    sources = frame.get("source_id", frame.image_hash).fillna(frame.image_hash)
    parents = {}
    seen = {}

    def root(source):
        while source in parents:
            source = parents[source]
        return source

    for image_hash, source in zip(frame.image_hash, sources):
        first = seen.setdefault(image_hash, source)
        first_root, source_root = root(first), root(source)
        if first_root != source_root:
            parents[source_root] = first_root
    return sources.map(root)


def conflicting_labels(frame):
    return frame.groupby("image_hash").grade.transform("nunique").gt(1)


def train(frame, columns, seed):
    if "grade" not in frame or not frame.grade.isin(["A", "B", "C"]).all():
        raise ValueError("Every training row needs an independent grade A, B or C.")
    if conflicting_labels(frame).any():
        raise ValueError("Identical images have conflicting labels.")
    group_ids = evaluation_groups(frame)
    data = frame.drop_duplicates("image_hash")
    groups = group_ids.loc[data.index].reset_index(drop=True)
    data = data.reset_index(drop=True)
    X, y = data[columns], data.grade
    counts = pd.DataFrame({"grade": y, "group": groups}).drop_duplicates().grade.value_counts()
    if len(counts) < 2 or counts.min() < 6:
        raise ValueError("Supply at least six independent sources per observed grade, and two or more grades.")
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for a, b in outer.split(X, y, groups):
        if set(y[a]) == set(y[b]) == set(y):
            break
    else:
        raise ValueError("No split covers all grades. Add independent labeled sources.")
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X.iloc[a], y[a], groups[a]))
    if any(set(y.iloc[a].iloc[i]) != set(y) or set(y.iloc[a].iloc[j]) != set(y) for i,j in folds):
        raise ValueError("Training folds lack a grade. Add sources or adjust the split seed.")
    labels = sorted(set(y))
    report = {"unique_rows": len(data),
              "train_rows":len(a), "holdout_rows":len(b), "labels":labels, "models":{}}
    prediction_columns = ["item_id", "grade", "name", "category", "original_path"]
    predictions = data.iloc[b][[column for column in prediction_columns if column in data]].copy()
    for name in ["SVC", "KNN"]:
        model = estimator(name)
        score = cross_val_score(model, X.iloc[a], y[a], cv=folds, scoring="f1_macro").mean()
        model.fit(X.iloc[a], y[a])
        pred = model.predict(X.iloc[b])
        predictions[name] = pred
        report["models"][name] = {
            "training_cv_macro_f1":float(score),
            "held_out":classification_report(y[b], pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix":confusion_matrix(y[b], pred, labels=labels).tolist()}
    winner = max(report["models"], key=lambda name: report["models"][name]["training_cv_macro_f1"])
    report["selected"] = winner
    return estimator(winner).fit(X, y), report, predictions
