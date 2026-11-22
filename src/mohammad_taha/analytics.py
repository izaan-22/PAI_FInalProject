"""Create item, category, batch and context reports from inspection results."""
import json
import pandas as pd
from ..abdul_haseeb.utils import write_json


def export(frame, context, folder):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "items.csv", index=False)
    frame.to_json(folder / "items.json", orient="records", indent=2)
    for key in ["batch_code","category"]:
        summary = frame.groupby(key).agg(items=("item_id","size"), mean_defect=("defect_ratio","mean"),
                                        mean_weight_g=("weight_gram","mean"), mean_rule_score=("quality_score","mean"))
        counts = pd.crosstab(frame[key], frame.predicted_grade).reindex(columns=list("ABC"), fill_value=0)
        summary.join(counts).to_csv(folder / f"by_{key}.csv")
    rankings = frame.assign(grade_points=frame.predicted_grade.map({"A":3,"B":2,"C":1})).groupby(
        "batch_code").grade_points.mean().to_dict()
    ranked = sorted(rankings.items(), key=lambda pair: pair[1], reverse=True)
    defective = list(filter(lambda r: r["predicted_grade"] == "C", frame.to_dict("records")))
    pd.DataFrame(defective, columns=frame.columns).to_csv(folder / "grade_c.csv", index=False)
    write_json(folder / "context.json", context)
    item_label = "item" if len(frame) == 1 else "items"
    lines = [f"{len(frame)} {item_label} inspected. {len(defective)} received Grade C.",
             "Batch ranking by mean predicted grade points (A=3, B=2, C=1):",
             *[f"  {key}: {score:.2f}" for key, score in ranked]]
    price = context["prices"]["data"]
    if price and price.get("previous_price"):
        diff = price["price"] - price["previous_price"]
        lines.append(f"{price['market']} {price['commodity']}: price changed by {diff:+.2f} "
                     f"{price['currency']}/{price['unit']} since the previous available quote.")
    (folder / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf8")
    return "\n".join(lines)


def context_history(root):
    """Compare observed conditions over distinct dated runs with the same market settings."""
    records = []
    for file in root.glob("runs/*/context.json"):
        ctx = json.loads(file.read_text())
        weather, price = ctx["weather"]["data"], ctx["prices"]["data"]
        items = pd.read_csv(file.parent / "items.csv")
        for crop, group in items.groupby("name"):
            records.append({"run":file.parent.name, "crop":crop, "defect_ratio":group.defect_ratio.mean(),
                            **ctx.get("location", {}), **weather, **price})
    history = pd.DataFrame(records)
    history.to_csv(root / "context_history.csv", index=False)
    rows = []
    for measure, date in [("temperature_c","observed_at_utc"), ("price","observation_date")]:
        if measure not in history:
            continue
        keys = ["crop","latitude","longitude"] if measure == "temperature_c" else [
            "crop","market","commodity","price_type","currency","unit"]
        for key, group in history.dropna(subset=[measure, date]).groupby(keys):
            paired = group.groupby(date)[[measure,"defect_ratio"]].mean()
            if len(paired) >= 3 and paired.nunique().min() > 1:
                rows.append(dict(zip(keys,key)) | {"measure":measure, "periods":len(paired),
                            "pearson_r":paired.corr().iloc[0,1]})
    write_json(root / "context_correlations.json", rows)
