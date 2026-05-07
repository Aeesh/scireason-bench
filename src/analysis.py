import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCORED_DIR = "results/scored"
ANALYSIS_DIR = "results/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

QUESTION_TYPES = ["factual", "conceptual", "numerical", "synthesis", "calibration"]
TYPE_LABELS = {
    "factual": "Factual Recall",
    "conceptual": "Conceptual Explanation",
    "numerical": "Numerical Reasoning",
    "synthesis": "Cross-Domain Synthesis",
    "calibration": "Calibration & Uncertainty"
}


####### Load all scored results #######
model_data = {}
for filename in os.listdir(SCORED_DIR):
    if not filename.endswith("_scored.json"):
        continue
    model_key = filename.replace("_scored.json", "")
    with open(f"{SCORED_DIR}/{filename}") as f:
        model_data[model_key] = json.load(f)

print(f"Loaded results for: {list(model_data.keys())}")


####### Build summary dataframe #######
rows = []
for model_key, data in model_data.items():
    for item in data["responses"]:
        if item["score"] is None or item["score"] == -1:
            continue
        rows.append({
            "model": data["model_info"]["display_name"],
            "model_key": model_key,
            "params": data["model_info"]["params"],
            "provider": data["model_info"]["provider"],
            "question_id": item["id"],
            "type": item["type"],
            "domain": item["domain"],
            "difficulty": item["difficulty"],
            "score": item["score"],
            "score_pct": item["score"] / 3 * 100,  # normalise to 0-100
            "latency": item.get("latency_seconds", None)
        })

df = pd.DataFrame(rows)
df.to_csv(f"{ANALYSIS_DIR}/all_results.csv", index=False)
print(f"Total scored responses: {len(df)}")

####### Build summary dataframe #######
summary = df.groupby("model")["score_pct"].agg(["mean", "std", "count"]).round(2)
summary.columns = ["Mean Score %", "Std Dev", "Questions Scored"]
print("\nOverall Summary:")
print(summary)
summary.to_csv(f"{ANALYSIS_DIR}/overall_summary.csv")

####### Summary per type #######
type_summary = df.groupby(["model", "type"])["score_pct"].mean().unstack(fill_value=0)
type_summary.to_csv(f"{ANALYSIS_DIR}/type_summary.csv")

####### Summary per difficulty #######
diff_summary = df.groupby(["model", "difficulty"])["score_pct"].mean().unstack(fill_value=0)
diff_summary.to_csv(f"{ANALYSIS_DIR}/difficulty_summary.csv")


####### Plot 1: Overall mean scores #######
fig, ax = plt.subplots(figsize=(9, 5))
models = summary.index.tolist()
scores = summary["Mean Score %"].values
colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"][:len(models)]

bars = ax.bar(models, scores, color=colors, width=0.5, edgecolor="white")
ax.set_ylim(0, 100)
ax.set_ylabel("Mean Score (% of max)", fontsize=12)
ax.set_title("Overall Model Performance — SciReason-Bench", fontsize=14, fontweight="bold")
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{score:.1f}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/overall_scores.png", dpi=150)
print("Saved: overall_scores.png")


####### Plot 2: Per-type heatmap #######
fig, ax = plt.subplots(figsize=(11, 5))
heatmap_data = type_summary.rename(columns=TYPE_LABELS)
sns.heatmap(
    heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
    vmin=0, vmax=100, ax=ax, linewidths=0.5,
    annot_kws={"size": 11}
)
ax.set_title("Score by Question Type (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Model", fontsize=11)
ax.set_xlabel("Question Type", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/type_heatmap.png", dpi=150)
print("Saved: type_heatmap.png")

####### Plot 3: Radar chart — model profiles #######
categories = [TYPE_LABELS[t] for t in QUESTION_TYPES]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

for i, (model_key, model_scores) in enumerate(type_summary.iterrows()):
    values = [model_scores.get(t, 0) for t in QUESTION_TYPES]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=model_key,
            color=colors_radar[i % len(colors_radar)])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i % len(colors_radar)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25", "50", "75", "100"], size=8)
ax.set_title("Model Capability Profiles", size=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/radar_chart.png", dpi=150, bbox_inches="tight")
print("Saved: radar_chart.png")


####### Plot 4: Score by difficulty #######
fig, ax = plt.subplots(figsize=(9, 5))
diff_order = ["easy", "medium", "hard"]
x = np.arange(len(diff_order))
width = 0.2

for i, model_key in enumerate(diff_summary.index):
    scores_by_diff = [diff_summary.loc[model_key].get(d, 0) for d in diff_order]
    ax.bar(x + i * width, scores_by_diff, width, label=model_key,
           color=colors[i % len(colors)])

ax.set_xticks(x + width * (len(diff_summary) - 1) / 2)
ax.set_xticklabels(["Easy", "Medium", "Hard"])
ax.set_ylabel("Mean Score (%)", fontsize=12)
ax.set_title("Performance by Difficulty Level", fontsize=14, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/difficulty_scores.png", dpi=150)
print("Saved: difficulty_scores.png")


####### Plot 5: Latency comparison #######
latency_df = df.dropna(subset=["latency"])
if not latency_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    latency_summary = latency_df.groupby("model")["latency"].mean()
    ax.bar(latency_summary.index, latency_summary.values,
           color=colors[:len(latency_summary)])
    ax.set_ylabel("Mean Latency (seconds)", fontsize=12)
    ax.set_title("Average Response Latency by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_DIR}/latency.png", dpi=150)
    print("Saved: latency.png")

print(f"\nAll analysis saved to {ANALYSIS_DIR}/")

