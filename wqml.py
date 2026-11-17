"""ML replacement for the interval-valued fuzzy AHP ranking of
Srinivas & Singh (2018), Environ Dev Sustain 20:2373-2397.

Fuzzy model:  expert opinion -> pairwise AHP -> fuzzy weights -> rank
This module:  measurements -> predicted physical impact -> policy rule -> rank

One row = one source x one sampling event x one pollutant (see index.html #data).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Columns the collected CSV must have. Concentrations in mg/L, flow in MLD.
# Nondetect = 0, not measured = blank.
COLUMNS = [
    "source_id", "industrial_area", "industry_type", "sample_date", "pollutant",
    "discharge_flow_mld", "production_rate", "treatment_status",
    "rainfall_mm", "river_flow_m3s",
    "inlet", "outlet", "upstream", "downstream", "near_well", "control_well",
    "detection_limit",
]

# CPCB effluent standards, mg/L (paper Table 2).
# ponytail: pH left out, it is a range (6.5-8) not a ceiling; add a range rule if needed.
# ponytail: duplicates data/paper/standards.csv; read it instead if limits start changing.
LIMITS = {"TDS": 2100, "TSS": 100, "Cu": 3, "Ni": 3, "Pb": 0.1, "Cd": 1, "Zn": 5,
          "Cr": 2, "As": 0.2, "F": 2, "COD": 250, "BOD": 30,
          # General standards printed in the source report (UPPCB 2013, Tables 7.2-7.5)
          "Fe": 3, "Mn": 2, "Cr6": 0.1, "NO3": 10, "PO4": 5}

# PLACEHOLDERS. Value judgments, not learnable from chemistry. Must be set by
# domain/regulatory experts before any ranking is published.
PH_RANGE = (5.5, 9.0)  # general standard, inland surface water (source report Tables 7.x; paper prints 6.5-8)
TOXICITY = {"Cr": 3, "As": 3, "Pb": 3, "Cd": 3, "F": 2, "Ni": 2}  # default 1
POLICY = {"exceedance": 0.4, "river": 0.25, "groundwater": 0.25, "violation": 0.1}

NUMERIC = ["discharge_flow_mld", "production_rate", "rainfall_mm", "river_flow_m3s",
           "inlet", "outlet", "upstream", "control_well", "month"]
CATEGORICAL = ["pollutant", "industry_type", "treatment_status", "season"]

# target -> (task, features to drop because they leak the label)
# downstream / near_well are never features: they are outcomes, not conditions.
TARGETS = {
    "river_change": ("reg", []),
    "gw_change": ("reg", []),
    "outlet_load_kg_day": ("reg", ["outlet"]),
    "violation": ("clf", ["outlet"]),
}

# Everything printed in the paper, transcribed. Table 10 = comparison only, never a label.
PAPER = Path(__file__).parent / "data" / "paper"


def audit(df):
    """Raise on schema errors; return a table of data-quality problems to fix at source."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    conc = ["inlet", "outlet", "upstream", "downstream", "near_well", "control_well"]
    checks = {
        "negative concentration": (df[conc] < 0).any(axis=1),
        "negative flow": df["discharge_flow_mld"] < 0,
        "unknown pollutant": ~df["pollutant"].isin([*LIMITS, "pH"]),
        "duplicate source/event/pollutant": df.duplicated(
            ["source_id", "campaign" if "campaign" in df else "sample_date", "pollutant"], keep=False),
        "outlet > inlet (treatment adds pollutant?)": df["outlet"] > df["inlet"],
        "no discharge flow (load impossible)": df["discharge_flow_mld"].isna(),
        "no upstream (river change impossible)": df["upstream"].isna(),
        "no control well (gw change impossible)": df["control_well"].isna(),
    }
    return pd.DataFrame({"rows": {k: int(v.sum()) for k, v in checks.items()}})


ROLES = ["inlet", "outlet", "upstream", "downstream", "near_well", "control_well"]
SAMPLE_IDS = ["station_id", "campaign", "sample_date", "season", "discharge_flow_mld"]
ALIASES = {"copper": "Cu", "nickel": "Ni", "lead": "Pb", "cadmium": "Cd", "zinc": "Zn", "chromium": "Cr",
           "total chromium": "Cr", "arsenic": "As", "fluoride": "F", "iron": "Fe", "ph": "pH"}


def parse_value(v):
    """Lab-sheet cell -> (value, detection_limit). 'ND'/'BDL' -> 0 (nondetect), '<0.01' -> 0 with DL 0.01,
    '-' / blank -> missing."""
    s = str(v).strip().upper()
    if s in ("ND", "BDL"):
        return 0.0, np.nan
    if s.startswith("<"):
        return 0.0, float(s[1:])
    try:
        return float(s), np.nan
    except ValueError:
        return np.nan, np.nan


def reshape(samples, stations):
    """Lab sheet (one row per physical sample, one column per parameter) + station register
    (source_id, station_id, role) -> model rows (one per source x campaign x pollutant).

    A station may serve several sources/roles (SW-4 is S3's outlet and CETP Unnao's downstream).
    `campaign` pairs samples of one sampling round taken on different days; defaults to sample_date."""
    s = samples.copy()
    for c in SAMPLE_IDS:
        if c not in s:
            s[c] = np.nan
    s["sample_date"] = pd.to_datetime(s["sample_date"])
    s["campaign"] = s["campaign"].fillna(s["sample_date"].astype(str).where(s["sample_date"].notna()))
    if s["campaign"].isna().any():
        raise ValueError("every sample needs a campaign or sample_date")
    reg = stations.dropna(subset=["station_id"])
    if bad := set(reg["role"]) - set(ROLES):
        raise ValueError(f"unknown roles in register: {sorted(bad)}")
    if unknown := set(s["station_id"]) - set(reg["station_id"]):
        raise ValueError(f"stations missing from register: {sorted(unknown)}")

    long = s.melt(id_vars=SAMPLE_IDS, var_name="pollutant", value_name="raw")
    long["pollutant"] = long["pollutant"].map(lambda p: ALIASES.get(p.strip().lower(), p.strip()))
    long[["value", "dl"]] = pd.DataFrame(long["raw"].map(parse_value).tolist(), index=long.index)
    m = long.dropna(subset=["value"]).merge(reg[["station_id", "source_id", "role"]], on="station_id")

    key = ["source_id", "campaign", "pollutant"]
    out = m.pivot_table(index=key, columns="role", values="value", aggfunc="mean")  # ponytail: replicates averaged
    g = m.groupby(key)
    out["detection_limit"] = g["dl"].max()
    out["sample_date"] = g["sample_date"].min()
    out["season"] = g["season"].first()
    if "discharge_flow_mld" in reg:  # flow as a station property (e.g. from a plant register)
        m["discharge_flow_mld"] = m["discharge_flow_mld"].fillna(
            m.set_index(["source_id", "station_id"]).index.map(
                reg.dropna(subset=["discharge_flow_mld"]).set_index(["source_id", "station_id"])["discharge_flow_mld"]).to_series(index=m.index))
    out["discharge_flow_mld"] = m[m["role"] == "outlet"].groupby(key)["discharge_flow_mld"].mean()
    meta = reg.drop_duplicates("source_id")[["source_id", "industrial_area", "industry_type"]]
    out = out.reset_index().merge(meta, on="source_id", how="left")
    return out.reindex(columns=COLUMNS + ["season", "campaign"])


def season(month):
    # Indian seasons; the paper sampled pre- and post-monsoon.
    return pd.cut(month, [0, 2, 5, 9, 11, 12], labels=["winter", "pre_monsoon", "monsoon", "post_monsoon", "winter"],
                  ordered=False).astype(str)


def prepare(df):
    """Clean, add time features and the physical targets."""
    df = df.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df["month"] = df["sample_date"].dt.month
    given = df["season"] if "season" in df else None  # paper rows: season known, date not
    df["season"] = season(df["month"]).where(df["month"].notna(), given)
    conc = ["inlet", "outlet", "upstream", "downstream", "near_well", "control_well"]
    df[conc] = df[conc].mask(df[conc] < 0)  # impossible -> missing
    # Convention: nondetect entered as 0, not-measured left blank. Nondetect -> DL/2 when the detection
    # limit is known, else stays 0 (below any limit; still counts as a pass, not as missing).
    # ponytail: DL/2 substitution; upgrade to censored regression (Tobit) if nondetects are common.
    for c in conc:
        df[c] = df[c].mask((df[c] == 0) & df["detection_limit"].notna(), df["detection_limit"] / 2)
    if "limit" not in df:  # a limit column in the data wins; else CPCB
        df["limit"] = df["pollutant"].map(LIMITS)
    df["outlet_load_kg_day"] = df["outlet"] * df["discharge_flow_mld"]  # mg/L x ML/day = kg/day
    df["river_change"] = df["downstream"] - df["upstream"]
    df["gw_change"] = df["near_well"] - df["control_well"]
    df["exceedance_ratio"] = df["outlet"] / df["limit"]
    df["violation"] = (df["outlet"] > df["limit"]).astype(float).where(df["outlet"].notna() & df["limit"].notna())
    ph = df["pollutant"].eq("pH") & df["outlet"].notna()  # pH has a range, not a ceiling
    df.loc[ph, "violation"] = (~df.loc[ph, "outlet"].between(*PH_RANGE)).astype(float)
    return df


def _pre(drop=()):
    num = [c for c in NUMERIC if c not in drop]
    # arcsinh: log-like for heavy-tailed concentrations/flows, still defined at 0.
    num_pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), FunctionTransformer(np.arcsinh), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, CATEGORICAL)])


def _reg(est):
    # Targets span orders of magnitude and can be negative (river_change) -> arcsinh.
    return TransformedTargetRegressor(est, func=np.arcsinh, inverse_func=np.sinh)


def models(task, drop=()):
    """Candidates from index.html. Elastic Net / logistic is the required baseline.
    ponytail: fixed hyperparameters; add grouped nested CV tuning once real data exists."""
    if task == "reg":
        zoo = {
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5),
            "random_forest": RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=0),
            "xgboost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8),
            "svr": SVR(C=3.0),
        }
        return {k: make_pipeline(_pre(drop), _reg(v)) for k, v in zoo.items()}
    zoo = {
        "logistic": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8),
        "svm": CalibratedClassifierCV(SVC(C=3.0), ensemble=False),
    }
    return {k: make_pipeline(_pre(drop), v) for k, v in zoo.items()}


SCORING = {
    "reg": {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
    "clf": {"recall": "recall", "precision": "precision", "ROC_AUC": "roc_auc", "Brier": "neg_brier_score"},
}


def xy(df, target):
    d = df.dropna(subset=[target])
    return d, d[NUMERIC + CATEGORICAL], d[target], d["source_id"]


def compare(df, target, folds=5):
    """Grouped CV: whole sources held out, so the score answers 'how well for an unseen industry'.
    Never split repeated rows of the same source randomly."""
    task, drop = TARGETS[target]
    _, X, y, g = xy(df, target)
    cv = GroupKFold(min(folds, g.nunique()))
    rows = {}
    for name, m in models(task, drop).items():
        s = cross_validate(m, X, y, groups=g, cv=cv, scoring=SCORING[task])
        rows[name] = {k: abs(s[f"test_{k}"]).mean() if k != "R2" else s[f"test_{k}"].mean() for k in SCORING[task]}
    return pd.DataFrame(rows).T


def future_holdout(df, target, model, test_frac=0.2):
    """Train on the past, test on the latest dates AND unseen sources (the strictest check)."""
    task, drop = TARGETS[target]
    d, X, y, g = xy(df, target)
    cutoff = d["sample_date"].quantile(1 - test_frac)
    test_src = set(pd.Series(g.unique()).sample(frac=test_frac, random_state=0))
    tr = (d["sample_date"] < cutoff) & ~g.isin(test_src)
    te = (d["sample_date"] >= cutoff) & g.isin(test_src)
    m = models(task, drop)[model].fit(X[tr], y[tr])
    if task == "reg":
        err = m.predict(X[te]) - y[te]
        return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()), "n_test": int(te.sum())}
    p = m.predict_proba(X[te])[:, 1]
    return {"Brier": ((p - y[te]) ** 2).mean(), "recall": ((p > .5) & (y[te] == 1)).sum() / max((y[te] == 1).sum(), 1),
            "n_test": int(te.sum())}


def oof_predictions(df, target, model, folds=5, interval=0.8):
    """Out-of-fold predictions (source-grouped) + quantile interval. Replaces the paper's
    fuzzy interval [(p',p); q; (r,r')] with a measured prediction interval."""
    task, drop = TARGETS[target]
    d, X, y, g = xy(df, target)
    cv = GroupKFold(min(folds, g.nunique()))
    out = d[["source_id", "industrial_area", "pollutant", "sample_date"]].copy()
    out["y"] = y
    if task == "clf":
        out["pred"] = cross_val_predict(models(task, drop)[model], X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
        return out
    out["pred"] = cross_val_predict(models(task, drop)[model], X, y, groups=g, cv=cv)
    lo_q, hi_q = (1 - interval) / 2, (1 + interval) / 2
    for name, q in [("lo", lo_q), ("hi", hi_q)]:
        m = make_pipeline(_pre(drop), _reg(HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0)))
        out[name] = cross_val_predict(m, X, y, groups=g, cv=cv)
    return out


def coverage(oof):
    return ((oof["y"] >= oof["lo"]) & (oof["y"] <= oof["hi"])).mean()


def rank_sources(df, preds):
    """Policy layer replacing paper Eq. 10 (weighted sum of criteria).
    preds: {"river": oof, "groundwater": oof, "violation": oof} from oof_predictions.
    Every component -> percentile within pollutant (scale-free), x toxicity, mean per source.
    Ranked within industrial area like the paper. Higher risk = higher priority."""
    key = ["source_id", "industrial_area", "pollutant", "sample_date"]
    parts = df[key + ["exceedance_ratio"]].rename(columns={"exceedance_ratio": "exceedance"})
    for name, oof in preds.items():
        cols = key + ["pred"] + (["hi", "lo"] if "hi" in oof else [])
        o = oof[cols].rename(columns={"pred": name, "hi": f"{name}_hi", "lo": f"{name}_lo"})
        parts = parts.merge(o, on=key, how="left")
    for c in POLICY:
        parts[c] = parts.groupby("pollutant")[c].rank(pct=True) if c in parts else np.nan
    tox = parts["pollutant"].map(TOXICITY).fillna(1)
    parts["risk"] = sum(w * parts[c].fillna(0.5) for c, w in POLICY.items()) * tox
    # Uncertainty: interval width, percentile within pollutant (0 = tightest, 1 = widest).
    widths = [(parts[f"{c}_hi"] - parts[f"{c}_lo"]).groupby(parts["pollutant"]).rank(pct=True)
              for c in preds if f"{c}_hi" in parts]
    parts["uncertainty"] = sum(widths) / len(widths) if widths else np.nan
    out = parts.groupby(["industrial_area", "source_id"]).agg(risk=("risk", "mean"), uncertainty=("uncertainty", "mean"))
    out["rank_in_area"] = out.groupby("industrial_area")["risk"].rank(ascending=False).astype(int)
    return out.sort_values(["industrial_area", "rank_in_area"])


def paper_ranking():
    t = pd.read_csv(PAPER / "table10_weights.csv")
    t["source_id"] = t["industrial_area"] + "-" + t["source"]
    return t.rename(columns={"final": "fuzzy_weight"}).set_index(["industrial_area", "source_id"])[["fuzzy_weight"]]


def synthetic(months=24, seed=0):
    """FAKE data with the real schema, sites and pollutant panel, so the pipeline runs end to end
    before field data arrives. Magnitudes loosely follow paper Table 2. Never report results from it."""
    rng = np.random.default_rng(seed)
    base = {"TDS": 12000, "TSS": 390, "Cu": .1, "Ni": .15, "Pb": .08, "Cd": .01, "Zn": .6, "Cr": 3,
            "As": .09, "F": 1.5, "COD": 3500, "BOD": 700}
    types = ["tannery", "textile", "paper_pulp", "food", "cetp", "drain"]
    srcs = [(sid, a, rng.choice(types), rng.beta(5, 2), rng.lognormal(0, 1.2))
            for a, sid in paper_ranking().index]
    dates = pd.date_range("2024-01-15", periods=months, freq="MS") + pd.Timedelta(days=14)
    rows = []
    for sid, area, typ, eff, flow_scale in srcs:
        for d in dates:
            monsoon = d.month in (6, 7, 8, 9)
            rain = rng.gamma(2, 80 if monsoon else 5)
            river = rng.lognormal(7 if monsoon else 5.5, 0.3)
            flow = flow_scale * rng.lognormal(0, 0.2)
            status = "down" if rng.random() < 0.1 else "operating"
            e = 0.05 if status == "down" else eff
            for p, b in base.items():
                tan = 5 if (typ == "tannery" and p == "Cr") else 1
                inlet = b * tan * rng.lognormal(0, 0.5)
                outlet = inlet * (1 - e) * rng.lognormal(0, 0.2)
                up = b * 0.01 * rng.lognormal(0, 0.3)
                ctrl = b * 0.005 * rng.lognormal(0, 0.3)
                rows.append({
                    "source_id": sid, "industrial_area": area, "industry_type": typ, "sample_date": d,
                    "pollutant": p, "discharge_flow_mld": flow, "production_rate": flow * rng.lognormal(0, .1),
                    "treatment_status": status, "rainfall_mm": rain, "river_flow_m3s": river,
                    "inlet": inlet, "outlet": outlet, "upstream": up,
                    "downstream": up + 50 * outlet * flow / river * rng.lognormal(0, 0.3),
                    "near_well": ctrl + outlet * 0.02 * (0.5 if monsoon else 1) * rng.lognormal(0, 0.4),
                    "control_well": ctrl, "detection_limit": b * 1e-3,
                })
    df = pd.DataFrame(rows)
    for c in ["discharge_flow_mld", "upstream", "control_well", "near_well", "rainfall_mm"]:
        df.loc[rng.random(len(df)) < 0.03, c] = np.nan  # realistic gaps
    return df


# --- Replacing the expert ranking (method_a / method_b notebooks) ------------------------------

CRITERIA = [f"C{i}" for i in range(1, 11)]
WORSE_IF_HIGH = {c: c not in ("C2", "C5") for c in CRITERIA}
# Paper Table 4 C5 scale (9 = less organic ... 1 = hazardous). PLACEHOLDER mapping: confirm with domain experts.
C5_CLASS = {"tannery": 1, "steel": 1, "chemical": 1, "cetp": 3, "drain": 3, "textile": 3,
            "sewage_drain": 5, "food": 5, "meat": 5, "animal_byproduct": 5, "oil": 5, "paper_pulp": 5}


def summarise(df):
    """Prepared rows -> one row per industry, one column per paper criterion C1-C10."""
    d = df.copy()
    # event = one sampling round: date, else season (paper rows are undated), else one 'undated' round
    d["event"] = d["sample_date"].dt.strftime("%Y-%m-%d").fillna(d["season"]).fillna("undated")
    undated = d["season"].isna()  # paper text rows: no season -> count in both seasons
    pre, post = undated | d["season"].eq("pre_monsoon"), undated | d["season"].eq("post_monsoon")
    every = pd.Series(True, index=d.index)
    d["excess"] = (d["exceedance_ratio"] - 1).clip(lower=0) * d["pollutant"].map(TOXICITY).fillna(1)
    for c in ["river_change", "gw_change"]:  # units differ by pollutant -> percentile within pollutant
        d[c] = d.groupby("pollutant")[c].rank(pct=True)
    eff = ((d["inlet"] - d["outlet"]) / d["inlet"]).where(d["inlet"].gt(0) & d["pollutant"].ne("pH"))

    def per_event(mask, col):  # sum over pollutants within one sampling event, then mean over events
        return d[mask].groupby(["source_id", "event"])[col].sum(min_count=1).groupby(level=0).mean()

    s = pd.DataFrame({
        "C1": d.groupby("source_id")["discharge_flow_mld"].mean(),
        "C2": eff.groupby(d["source_id"]).mean(),
        "C3": per_event(pre, "violation"),
        "C4": per_event(post, "violation"),
        "C6": d[pre].groupby("source_id")["river_change"].mean(),
        "C7": d[post].groupby("source_id")["river_change"].mean(),
        "C8": d[pre].groupby("source_id")["gw_change"].mean(),
        "C9": d[post].groupby("source_id")["gw_change"].mean(),
        "C10": per_event(every, "excess"),
    })
    s = d.groupby("source_id")[["industrial_area", "industry_type"]].first().join(s)
    s["C5"] = s["industry_type"].map(C5_CLASS)
    # how many parameters were tested at the outlet: C3/C4/C10 grow with a wider panel (report sensitivity)
    s["n_tested"] = d[d["violation"].notna()].groupby("source_id")["pollutant"].nunique().reindex(s.index).fillna(0).astype(int)
    return s[["industrial_area", "industry_type", "n_tested"] + CRITERIA]


def local_weights(s, min_measured=2):
    """Paper-shaped local weights from measured criteria (method B / C).
    Per criterion and area: rank (worse = lower); measured ranks are stretched onto 1..n so their mean is
    the neutral middle (n+1)/2; missing = neutral; criteria measured for fewer than `min_measured`
    industries in an area are neutral for the whole area. Then scale each area to sum to 1."""
    area = s["industrial_area"]
    n = area.map(area.value_counts())
    out = pd.DataFrame(index=s.index)
    for c in CRITERIA:
        good = -s[c] if WORSE_IF_HIGH[c] else s[c]
        r = good.groupby(area).rank()
        k = good.notna().groupby(area).transform("sum")
        r = (1 + (r - 1) * (n - 1) / (k - 1).clip(lower=1)).where(k >= min_measured)
        r = r.fillna((n + 1) / 2)
        out[c] = r / r.groupby(area).transform("sum")
    return out


def paper_labels():
    """Table 10 scores; target = score / area mean (scores are only relative within an area).
    The target uses every label of an area: fine for display, but a model test must normalise with
    training labels only (method A does this per fold)."""
    p = paper_ranking().reset_index()
    p["target"] = p["fuzzy_weight"] / p.groupby("industrial_area")["fuzzy_weight"].transform("mean")
    return p.set_index("source_id")[["fuzzy_weight", "target"]]


def agreement(t, col, ref="fuzzy_weight"):
    """How closely `col` reproduces the `ref` ranking (low = more polluting). The paper ranks only
    within each industrial area, so agreement is measured within areas: Spearman per area and their
    mean, and the share of areas whose most-polluting industry agrees (ties count as sets: agree if
    the two sets of tied-worst industries overlap)."""
    from scipy.stats import spearmanr
    by_area = t.groupby("industrial_area")
    rho = {a: spearmanr(g[col], g[ref]).statistic for a, g in by_area if len(g) > 2}
    worst = lambda x: set(x.index[x == x.min()])
    # an area whose scores are all equal has no order (rho undefined): left out of the mean
    out = {"spearman_within_area": float(np.nanmean(list(rho.values())))} | {f"spearman_{a}": v for a, v in rho.items()}
    out["same_worst_industry"] = by_area.apply(lambda g: bool(worst(g[col]) & worst(g[ref]))).mean()
    return pd.Series(out).round(3)


def dummy(seed=0):
    """Schema-identical FAKE data. Each industry's pollution is scaled by how bad the paper's
    experts scored it (area mean / fuzzy score), plus noise. Agreement with the paper on it is
    EXPECTED by construction; it proves the code works, not the method."""
    rng = np.random.default_rng(seed)
    raw = synthetic(seed=seed)
    p = paper_ranking().reset_index()
    sev = p.groupby("industrial_area")["fuzzy_weight"].transform("mean") / p["fuzzy_weight"]
    k = raw["source_id"].map(pd.Series(sev.to_numpy() * rng.lognormal(0, 0.3, len(p)), index=p["source_id"]))
    raw["outlet"] *= k
    raw["downstream"] = raw["upstream"] + (raw["downstream"] - raw["upstream"]) * k
    raw["near_well"] = raw["control_well"] + (raw["near_well"] - raw["control_well"]) * k
    raw["discharge_flow_mld"] *= k ** 0.5
    return raw


UPPCB = Path(__file__).parent / "data" / "uppcb"


def load(paths=("data/measurements.csv", "data/samples.csv", "data/stations.csv")):
    """Returns (raw, source). Order: your model rows, your lab sheet + register, the paper's source
    report (UPPCB 2013, data/uppcb/), dummy."""
    meas, samples, stations = map(Path, paths)
    if meas.exists():
        return pd.read_csv(meas), "user"
    if samples.exists():
        reg = pd.read_csv(stations if stations.exists() else UPPCB / "stations.csv")
        return reshape(pd.read_csv(samples), reg), "user"
    if (UPPCB / "samples.csv").exists():
        return reshape(pd.read_csv(UPPCB / "samples.csv"), pd.read_csv(UPPCB / "stations.csv")), "uppcb"
    return dummy(), "dummy"


if __name__ == "__main__":
    # Formula checks against hand-computed values.
    t = prepare(pd.DataFrame([{
        "source_id": "x", "industrial_area": "a", "industry_type": "tannery", "sample_date": "2012-04-01",
        "pollutant": "BOD", "discharge_flow_mld": 2.0, "production_rate": 1, "treatment_status": "operating",
        "rainfall_mm": 0, "river_flow_m3s": 100, "inlet": 680, "outlet": 98, "upstream": 3, "downstream": 12.5,
        "near_well": 5, "control_well": 1, "detection_limit": 1}]))
    r = t.iloc[0]
    assert r.outlet_load_kg_day == 196 and r.river_change == 9.5 and r.gw_change == 4
    assert r.exceedance_ratio == 98 / 30 and r.violation == 1 and r.season == "pre_monsoon"
    assert audit(t).loc["outlet > inlet (treatment adds pollutant?)", "rows"] == 0
    # Reshape: paper Table 2 as a lab sheet must reproduce the hand-transcribed rows.
    reg = pd.read_csv(PAPER / "stations.csv")
    got = reshape(pd.read_csv(PAPER / "table2_samples.csv"), reg).set_index(["season", "pollutant"])
    want = pd.read_csv(PAPER / "measurements.csv").query("source_id == 'site2-S1'").set_index(["season", "pollutant"])
    assert len(got) == len(want) == 23 and got.loc[want.index, ["inlet", "outlet"]].equals(want[["inlet", "outlet"]])
    assert got["discharge_flow_mld"].eq(1.6).all() and got.loc[("post_monsoon", "Cd"), "outlet"] == 0
    # One station, several roles: SW-4 is S3's outlet and CETP Unnao's downstream.
    lab = pd.DataFrame({"station_id": ["SW-2", "SW-3", "SW-4"], "sample_date": ["2012-04-02", "2012-04-02", "2012-04-03"],
                        "campaign": "2012-pre", "BOD": [98, 10, "50"], "Lead": ["<0.01", "BDL", "-"]})
    r = prepare(reshape(lab, reg)).set_index(["source_id", "pollutant"])
    assert r.loc[("site2-S1", "BOD"), "river_change"] == 40 and r.loc[("site2-S3", "BOD"), "outlet"] == 50
    assert r.loc[("site2-S1", "Pb"), "outlet"] == 0.005  # '<0.01' -> DL/2
    assert r.loc[("site2-S1", "BOD"), "sample_date"] == pd.Timestamp("2012-04-02")
    # Per-industry summary on the paper's real rows: CETP Unnao pre-monsoon has 4 over the limit.
    s = summarise(prepare(pd.read_csv(PAPER / "measurements.csv")))
    assert s.loc["site2-S1", "C3"] == 4 and s.loc["site2-S8", "C3"] == 1 and len(s) == 8
    assert paper_labels().groupby(paper_ranking().reset_index()["industrial_area"].to_numpy())["target"].mean().round(6).eq(1).all()
    # Paper's source report (data/uppcb/): all 33 industries, CETP Unnao identical to paper Table 2.
    u = reshape(pd.read_csv(UPPCB / "samples.csv"), pd.read_csv(UPPCB / "stations.csv"))
    assert u["source_id"].nunique() == 33 and audit(u).loc["unknown pollutant", "rows"] == 0
    cetp = u[u["source_id"].eq("site2-S1")].set_index(["campaign", "pollutant"])
    assert cetp.loc[("2012-post", "BOD"), ["inlet", "outlet"]].tolist() == [902, 353]
    assert cetp.loc[("2012-pre", "BOD"), ["upstream", "downstream"]].tolist() == [280, 126]
    # Table 10: every final score = sum(criterion weight x local weight), as printed (to rounding).
    t10 = pd.read_csv(PAPER / "table10_weights.csv")
    w = pd.read_csv(PAPER / "criteria.csv").set_index("criterion")["weight"]
    assert (t10[w.index].mul(w).sum(axis=1) - t10["final"]).abs().max() < 0.001
    # Nondetect without a detection limit = 0 (a pass), not missing; pH uses a range.
    nd = prepare(pd.DataFrame([{**{c: np.nan for c in COLUMNS}, "pollutant": "Cd", "outlet": 0.0, "sample_date": "2012-11-05"},
                               {**{c: np.nan for c in COLUMNS}, "pollutant": "pH", "outlet": 11.2, "sample_date": "2012-11-05"}]))
    assert nd["outlet"].tolist() == [0.0, 11.2] and nd["violation"].tolist() == [0.0, 1.0]
    # Local weights: measured ranks average to neutral, so a criterion measured for few industries
    # cannot push them all below the unmeasured ones; too few measured -> whole area neutral.
    lw = local_weights(pd.DataFrame({"industrial_area": ["a"] * 5, **{c: np.nan for c in CRITERIA}}
                                    ).assign(C6=[3.0, 1.0, 2.0, np.nan, np.nan], C7=[1.0] + [np.nan] * 4))
    assert abs(lw["C6"].iloc[:3].mean() - 0.2) < 1e-9 and lw["C6"].iloc[3] == 0.2 and (lw["C7"] == 0.2).all()
    # Agreement: within areas only; tied worst industries count as a set.
    ag = agreement(pd.DataFrame({"industrial_area": ["a"] * 3, "x": [1, 1, 2], "fuzzy_weight": [2, 1, 3]}, index=list("pqr")), "x")
    assert ag["same_worst_industry"] == 1.0 and "spearman_all" not in ag
    # Pipeline smoke test on synthetic data.
    df = prepare(synthetic(months=6))
    assert audit(df)["rows"].loc["unknown pollutant"] == 0
    oof = oof_predictions(df, "river_change", "elastic_net", folds=3)
    assert oof["pred"].notna().all() and 0 < coverage(oof) <= 1
    ranks = rank_sources(df, {"river": oof})
    assert len(ranks) == 33 and len(paper_ranking()) == 33 and ranks.groupby("industrial_area")["rank_in_area"].min().eq(1).all()
    print("ok")
