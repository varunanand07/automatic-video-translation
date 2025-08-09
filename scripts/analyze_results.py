import os
import json
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
except Exception:
    sm = None
    ols = None
    anova_lm = None


DATASETS = ["ted_talks", "lectures", "podcasts", "youtube_shorts"]
BASE = "data"


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_asr_wer_map(dataset: str) -> Dict[str, Optional[float]]:
    path = os.path.join(BASE, dataset, "asr_evaluation.json")
    if not os.path.exists(path):
        return {}

    data = _read_json(path)
    wer_map: Dict[str, Optional[float]] = {}

    if isinstance(data, list):
        for entry in data:
            vid = entry.get("id")
            if vid is None:
                continue
            wer_field = entry.get("wer")
            whisper_val: Optional[float] = None
            if isinstance(wer_field, dict):
                whisper_val = wer_field.get("whisper")
            elif isinstance(wer_field, (int, float)):
                whisper_val = float(wer_field)
            wer_map[vid] = whisper_val
    else:
        pass

    return wer_map

def load_mt_rows(dataset: str) -> List[dict]:
    mt_path = os.path.join(BASE, dataset, "mt_evaluation.json")
    if not os.path.exists(mt_path):
        return []

    mt = _read_json(mt_path)
    asr_map = load_asr_wer_map(dataset)

    rows: List[dict] = []
    for model, entries in mt.items():
        for e in entries:
            vid = e.get("id")
            rows.append({
                "dataset": dataset,
                "model": model,
                "id": vid,
                "title": e.get("title", ""),
                "BLEU": e.get("bleu_score"),
                "chrF": e.get("chrf_score"),
                "TER":  e.get("ter_score"),
                "WER":  asr_map.get(vid, None)
            })
    return rows

def ci95_mean(values: np.ndarray) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (math.nan, math.nan)
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = 1.96 * (s / math.sqrt(n)) if n > 1 else 0.0
    return (m - half, m + half)

def pearson_r_and_p(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(x) != len(y) or len(x) < 2:
        return None, None
    r = float(np.corrcoef(x, y)[0, 1])
    if scipy_stats is None or len(x) < 3:
        return r, None
    n = len(x)
    denom = max(1e-12, 1.0 - r**2)
    t = r * math.sqrt((n - 2) / denom)
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), df=n - 2))
    return r, float(p)

def main():
    all_rows: List[dict] = []
    for ds in DATASETS:
        all_rows.extend(load_mt_rows(ds))

    df = pd.DataFrame(all_rows)
    df.to_csv("all_metrics_flat.csv", index=False)

    summaries = []
    for (ds, model), grp in df.groupby(["dataset", "model"], dropna=False):
        for metric in ["BLEU", "chrF", "TER"]:
            vals = grp[metric].dropna().values.astype(float)
            n = len(vals)
            if n == 0:
                mean_v = std_v = ci_lo = ci_hi = None
            else:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                ci_lo, ci_hi = ci95_mean(vals)
            summaries.append({
                "dataset": ds,
                "model": model,
                "metric": metric,
                "mean": round(mean_v, 2) if mean_v is not None else None,
                "std":  round(std_v, 2) if n > 1 else None,
                "ci95_lo": round(ci_lo, 2) if n > 1 else None,
                "ci95_hi": round(ci_hi, 2) if n > 1 else None,
                "n": n
            })
    pd.DataFrame(summaries).to_csv("summary_by_ds_model.csv", index=False)

    corr_rows = []
    for (ds, model), grp in df.groupby(["dataset", "model"], dropna=False):
        sub = grp.dropna(subset=["WER", "BLEU"])
        n = len(sub)
        if n >= 2:
            r, p = pearson_r_and_p(sub["WER"].values.astype(float),
                                   sub["BLEU"].values.astype(float))
        else:
            r, p = (None, None)
        corr_rows.append({
            "dataset": ds,
            "model": model,
            "pearson_r_WER_BLEU": None if r is None else round(r, 3),
            "p": None if p is None else float(p),
            "n": n
        })
    pd.DataFrame(corr_rows).to_csv("correlations_WER_BLEU.csv", index=False)

    try:
        if sm is not None and ols is not None and anova_lm is not None:
            df_anova = df.dropna(subset=["BLEU"]).copy()
            df_anova["BLEU"] = df_anova["BLEU"].astype(float)
            model = ols('BLEU ~ C(model) + C(dataset) + C(model):C(dataset)', data=df_anova).fit()
            anova_table = anova_lm(model, typ=2)
            anova_table.to_csv("anova_model_x_dataset_bleu.csv")
        else:
            pd.DataFrame({
                "note": ["statsmodels not installed"]
            }).to_csv("anova_model_x_dataset_bleu.csv", index=False)
    except Exception as e:
        pd.DataFrame({
            "note": [f"ANOVA failed: {e}"]
        }).to_csv("anova_model_x_dataset_bleu.csv", index=False)

    print("All csv files saved to analysis/")


if __name__ == "__main__":
    main()
