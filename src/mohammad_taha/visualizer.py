"""Render three labeled figures and a data-dependent interpretation for each."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot(frame, folder, grades="ABC"):
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    captions = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / (name+".png"), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots()
    for c, color in zip("rgb", ["#b73e43","#318058","#4474ae"]):
        height = frame[[f"{c}_bin{i}" for i in range(8)]].mean()
        ax.stairs(height, np.linspace(0,1,9), label=c.upper(), color=color, linewidth=2)
    ax.legend()
    ax.set(title="Foreground color distribution", xlabel="Normalized intensity", ylabel="Mean fraction of crop pixels")
    save(fig, "color_distribution")
    channel = frame[["mean_red","mean_green","mean_blue"]].mean().idxmax()[5:]
    captions.append(f"{channel.title()} has the highest average foreground intensity across items.")
    counts = frame.predicted_grade.value_counts().reindex(list(grades), fill_value=0)
    colors = {"A": "#297957", "B": "#dfaa44", "C": "#ab5252"}
    fig, ax = plt.subplots()
    ax.bar([f"Grade {grade}" for grade in counts.index], counts, color=[colors[grade] for grade in counts.index])
    ax.set(title="Predicted quality grades", ylabel="Number of items")
    save(fig, "grade_frequency")
    captions.append("Predicted counts: " + ", ".join(f"{k} = {v}" for k,v in counts.items()) + ".")
    fig, ax = plt.subplots()
    measured = frame[frame.weight_gram.notna() & ~frame.weight_imputed]
    if len(measured):
        sns.scatterplot(data=measured,x="weight_gram",y="defect_ratio",hue="predicted_grade",ax=ax)
    else:
        ax.text(.5,.5,"No measured weights supplied",ha="center",transform=ax.transAxes)
    ax.set(title="Visible defect estimate by measured weight",xlabel="Weight (g)",ylabel="Estimated fraction of crop surface",ylim=(-.03,1.03))
    save(fig,"severity_vs_weight")
    corr = measured.weight_gram.corr(measured.defect_ratio) if len(measured)>2 and measured.weight_gram.nunique()>1 and measured.defect_ratio.nunique()>1 else np.nan
    captions.append(f"Across {len(measured)} measured items, weight and estimated defect fraction have Pearson r = {corr:.2f}."
                    if np.isfinite(corr) else "The measured weights and defect estimates do not support a correlation estimate.")
    (folder/"interpretation.txt").write_text("\n".join(captions)+"\n",encoding="utf8")
