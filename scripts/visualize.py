import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.autolayout": True})

def load_flat():
    return pd.read_csv("analysis/all_metrics_flat.csv")

def load_summary():
    p = "analysis/summary_by_ds_model.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def bar_bleu(df, summary):
    os.makedirs("visualizations", exist_ok=True)
    order_ds = ["lectures","podcasts","youtube_shorts","ted_talks"]
    order_ds = [d for d in order_ds if d in df["dataset"].unique()]
    order_model = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(10,6))
    width = 0.2
    x = np.arange(len(order_ds))

    for i, m in enumerate(order_model):
        means = []
        lo_err = []
        hi_err = []
        for ds in order_ds:
            g = df[(df["dataset"]==ds) & (df["model"]==m)]
            if summary is not None:
                s = summary[(summary["dataset"]==ds) & (summary["model"]==m) & (summary["metric"]=="BLEU")]
                if len(s):
                    mu = s["mean"].values[0]
                    lo = s["ci95_lo"].values[0]
                    hi = s["ci95_hi"].values[0]
                else:
                    mu = g["BLEU"].mean()
                    lo, hi = mu, mu
            else:
                mu = g["BLEU"].mean()
                lo, hi = mu, mu
            means.append(mu)
            lo_err.append(max(0, mu - lo))
            hi_err.append(max(0, hi - mu))

        xshift = x + (i - (len(order_model)-1)/2)*width
        ax.bar(xshift, means, width, label=m,
               yerr=np.vstack([lo_err, hi_err]),
               capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(order_ds, rotation=0)
    ax.set_ylabel("BLEU")
    ax.set_title("BLEU by Model × Dataset (95% CI)")
    ax.legend()
    out = "visualizations/bleu_by_model_dataset.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

def scatter_wer_bleu(df, asr_model):
    os.makedirs("visualizations", exist_ok=True)
    for ds in df["dataset"].unique():
        sub = df[df["dataset"]==ds].dropna(subset=["WER","BLEU"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7,5))
        for m in sorted(sub["model"].unique()):
            g = sub[sub["model"]==m]
            ax.scatter(g["WER"], g["BLEU"], alpha=0.7, label=m)
        ax.set_xlabel(f"WER ({asr_model})")
        ax.set_ylabel("BLEU")
        ax.set_title(f"WER vs BLEU — {ds}")
        ax.legend()
        out = f"visualizations/scatter_WER_BLEU_{ds}.png"
        plt.savefig(out, dpi=200)
        print(f"Saved {out}")

def correlation_heatmap(df):
    os.makedirs("visualizations", exist_ok=True)
    cols = ["BLEU","chrF","TER"]
    present = [c for c in cols if c in df.columns and df[c].notna().sum()>0]
    if len(present) < 2:
        return
    corr = df[present].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr, interpolation='nearest')
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticklabels(present)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", color="w")
    ax.set_title("Metric Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = "visualizations/metric_correlations.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model", default="whisper")
    args = parser.parse_args()

    df = load_flat()
    summary = load_summary()
    bar_bleu(df, summary)
    scatter_wer_bleu(df, args.asr_model)
    correlation_heatmap(df)

if __name__ == "__main__":
    main()
