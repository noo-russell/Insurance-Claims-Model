"""Frequency-severity models: classical GLM vs gradient boosting.

Reads the modelling views from SQLite, fits
    frequency: Poisson GLM (log link, exposure offset)   vs LightGBM Poisson
    severity:  Gamma GLM (log link, weighted by claims)  vs LightGBM Gamma
and compares them on a held-out test set. Pure premium = frequency x severity.

Outputs go to results/ (metrics.json, tables as CSV, figures as PNG).
"""
from pathlib import Path
import json
import sqlite3

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "insurance.db"
OUT = ROOT / "results"
FIG = OUT / "figures"

# GLM: banded / categorical rating factors, as a pricing actuary would set them up.
GLM_CATEGORICAL = ["driver_age_band", "veh_age_band", "veh_power_band",
                   "veh_brand", "veh_gas", "region"]
GLM_NUMERIC = ["bonus_malus_log", "density_log"]
# Boosting: raw features, the trees find their own splits and interactions.
GBM_NUMERIC = ["veh_power", "veh_age", "driver_age", "bonus_malus", "density"]
GBM_CATEGORICAL = ["veh_brand", "veh_gas", "area", "region"]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    con = sqlite3.connect(DB)
    pol = pd.read_sql("SELECT * FROM v_pure_premium", con)
    sev = pd.read_sql("SELECT * FROM v_severity_model", con)
    con.close()
    for df in (pol, sev):
        df["bonus_malus_log"] = np.log(df["bonus_malus"] / 50.0)
        df["density_log"] = np.log(df["density"])
        for c in GBM_CATEGORICAL:
            df[c] = pd.Categorical(df[c], categories=sorted(pol[c].unique()))
    return pol, sev


def glm_design(df, reference):
    """Dummy-coded design matrix; the reference level of each factor is dropped."""
    cols = [pd.Series(1.0, index=df.index, name="intercept")]
    for c in GLM_CATEGORICAL:
        d = pd.get_dummies(df[c].astype(str), prefix=c, prefix_sep="=", dtype=float)
        for level in reference[c]["levels"]:           # same columns in train and test
            if f"{c}={level}" not in d:
                d[f"{c}={level}"] = 0.0
        d = d[[f"{c}={l}" for l in reference[c]["levels"] if l != reference[c]["base"]]]
        cols.append(d)
    cols.append(df[GLM_NUMERIC].astype(float))
    return pd.concat(cols, axis=1)


def reference_levels(train):
    """Base level of each factor = the level with the most exposure."""
    ref = {}
    for c in GLM_CATEGORICAL:
        expo = train.groupby(train[c].astype(str))["exposure"].sum()
        ref[c] = {"levels": sorted(expo.index), "base": expo.idxmax()}
    return ref


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def gini(actual, predicted, weight):
    """Gini from the Lorenz curve: policies ordered from lowest to highest predicted risk."""
    order = np.argsort(predicted / weight)
    cum_w = np.cumsum(weight[order]) / weight.sum()
    cum_a = np.cumsum(actual[order]) / actual.sum()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # works on old and new numpy
    return 1 - 2 * trapezoid(cum_a, cum_w), cum_w, cum_a


def decile_table(df, pred_col, actual_col, weight_col="exposure"):
    """Sort by predicted rate, cut into 10 equal-exposure groups, compare with actual."""
    d = df[[pred_col, actual_col, weight_col]].copy()
    d["rate"] = d[pred_col] / d[weight_col]
    d = d.sort_values("rate")
    d["decile"] = np.minimum((d[weight_col].cumsum() / d[weight_col].sum() * 10).astype(int), 9) + 1
    g = d.groupby("decile").agg(exposure=(weight_col, "sum"),
                                predicted=(pred_col, "sum"), actual=(actual_col, "sum"))
    g["predicted_rate"] = g["predicted"] / g["exposure"]
    g["actual_rate"] = g["actual"] / g["exposure"]
    return g.reset_index()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def fit_frequency(pol):
    fit_rows = pol[pol.split != "test"]
    train, valid = pol[pol.split == "train"], pol[pol.split == "valid"]
    ref = reference_levels(fit_rows)

    # Poisson GLM with log(exposure) offset, fitted on train + valid.
    glm = sm.GLM(fit_rows["claim_count"], glm_design(fit_rows, ref),
                 family=sm.families.Poisson(),
                 offset=np.log(fit_rows["exposure"])).fit()
    pol["freq_glm"] = glm.predict(glm_design(pol, ref), offset=np.log(pol["exposure"]))

    # LightGBM Poisson. The exposure offset enters as the starting score.
    base = fit_rows["claim_count"].sum() / fit_rows["exposure"].sum()
    feats = GBM_NUMERIC + GBM_CATEGORICAL
    params = dict(objective="poisson", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)

    def dset(d):
        return lgb.Dataset(d[feats], d["claim_count"],
                           init_score=np.log(d["exposure"] * base), free_raw_data=False)

    stopper = lgb.train(params, dset(train), num_boost_round=3000, valid_sets=[dset(valid)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = stopper.best_iteration
    gbm = lgb.train(params, dset(fit_rows), num_boost_round=rounds)   # refit on train + valid
    pol["freq_gbm"] = pol["exposure"] * base * np.exp(gbm.predict(pol[feats], raw_score=True))
    pol["freq_null"] = pol["exposure"] * base
    return glm, gbm, rounds, ref


def fit_severity(pol, sev):
    fit_rows = sev[sev.split != "test"]
    train, valid = sev[sev.split == "train"], sev[sev.split == "valid"]
    ref = reference_levels(fit_rows)

    # Gamma GLM on average cost per claim, weighted by number of claims.
    glm = sm.GLM(fit_rows["avg_severity"], glm_design(fit_rows, ref),
                 family=sm.families.Gamma(link=sm.families.links.Log()),
                 var_weights=fit_rows["n_claims"]).fit()

    feats = GBM_NUMERIC + GBM_CATEGORICAL
    params = dict(objective="gamma", learning_rate=0.02, num_leaves=6,
                  min_data_in_leaf=150, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, cat_smooth=50, verbose=-1, seed=42)

    def dset(d):
        return lgb.Dataset(d[feats], d["avg_severity"], weight=d["n_claims"], free_raw_data=False)

    stopper = lgb.train(params, dset(train), num_boost_round=3000, valid_sets=[dset(valid)],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
    rounds = max(stopper.best_iteration, 1)
    gbm = lgb.train(params, dset(fit_rows), num_boost_round=rounds)

    null = np.average(fit_rows["avg_severity"], weights=fit_rows["n_claims"])
    for d in (pol, sev):                      # severity predicted for every policy
        d["sev_glm"] = glm.predict(glm_design(d, ref))
        d["sev_gbm"] = gbm.predict(d[feats])
        d["sev_null"] = null
    return glm, gbm, rounds


# --------------------------------------------------------------------------
# evaluation and figures
# --------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def evaluate(pol, sev):
    test, sev_test = pol[pol.split == "test"].copy(), sev[sev.split == "test"]
    y, w = test["claim_count"].to_numpy(), test["exposure"].to_numpy()
    m = {"test_policies": int(len(test)), "test_claims": int(y.sum()),
         "frequency": {}, "severity": {}, "pure_premium": {}}

    for name in ("null", "glm", "gbm"):
        mu = test[f"freq_{name}"].to_numpy()
        g, _, _ = gini(y, mu, w)
        m["frequency"][name] = {
            "poisson_deviance": mean_poisson_deviance(y, mu),
            "predicted_over_actual": mu.sum() / y.sum(),
            "gini": g if name != "null" else 0.0}
        m["severity"][name] = {
            "gamma_deviance": mean_gamma_deviance(sev_test["avg_severity"], sev_test[f"sev_{name}"],
                                                  sample_weight=sev_test["n_claims"]),
            "predicted_over_actual": float((sev_test[f"sev_{name}"] * sev_test["n_claims"]).sum()
                                           / sev_test["total_amount_capped"].sum())}
        test[f"pp_{name}"] = test[f"freq_{name}"] * test[f"sev_{name}"]
        g, _, _ = gini(test["loss_capped"].to_numpy(), test[f"pp_{name}"].to_numpy(), w)
        m["pure_premium"][name] = {
            "gini": g if name != "null" else 0.0,
            "predicted_over_actual": test[f"pp_{name}"].sum() / test["loss_capped"].sum()}
    for part, key in (("frequency", "poisson_deviance"), ("severity", "gamma_deviance")):
        for name in ("glm", "gbm"):
            m[part][name]["deviance_explained_pct"] = 100 * (1 - m[part][name][key] / m[part]["null"][key])
    return m, test


def figures(test, glm_freq, gbm_freq, ref):
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Frequency lift chart: predicted vs actual by decile of predicted risk.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name, color, title in ((axes[0], "glm", BLUE, "Poisson GLM"),
                                   (axes[1], "gbm", ORANGE, "Gradient boosting")):
        t = decile_table(test, f"freq_{name}", "claim_count")
        t.to_csv(OUT / f"frequency_deciles_{name}.csv", index=False)
        ax.plot(t.decile, t.actual_rate, color=INK, lw=2, marker="o", ms=5, label="Actual")
        ax.plot(t.decile, t.predicted_rate, color=color, lw=2, marker="o", ms=5, label="Predicted")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Decile of predicted frequency (1 = lowest risk)", fontsize=9, color=MUTED)
        ax.set_xticks(range(1, 11))
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axes[0].set_ylabel("Claims per policy-year", fontsize=9, color=MUTED)
    fig.suptitle("Test set: claim frequency by predicted-risk decile", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "frequency_lift.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)

    # 2. Lorenz curve for pure premium.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    w, loss = test["exposure"].to_numpy(), test["loss_capped"].to_numpy()
    for name, color, label in (("glm", BLUE, "GLM"), ("gbm", ORANGE, "Gradient boosting")):
        g, cw, ca = gini(loss, test[f"pp_{name}"].to_numpy(), w)
        ax.plot(cw, ca, color=color, lw=2, label=f"{label} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="No differentiation")
    ax.set_xlabel("Share of exposure, lowest predicted risk first", fontsize=9, color=MUTED)
    ax.set_ylabel("Share of claim cost", fontsize=9, color=MUTED)
    ax.set_title("Test set: Lorenz curve of pure premium", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "pure_premium_lorenz.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)

    # 3. GLM relativities for driver age (exp of coefficient, base level = 1.00).
    coef = pd.DataFrame({"coef": glm_freq.params, "se": glm_freq.bse, "p_value": glm_freq.pvalues})
    coef["relativity"] = np.exp(coef["coef"])
    coef["rel_low"] = np.exp(coef["coef"] - 1.96 * coef["se"])
    coef["rel_high"] = np.exp(coef["coef"] + 1.96 * coef["se"])
    coef.to_csv(OUT / "frequency_glm_coefficients.csv", index_label="term")

    levels = ref["driver_age_band"]["levels"]
    base = ref["driver_age_band"]["base"]
    rel = [1.0 if l == base else coef.loc[f"driver_age_band={l}", "relativity"] for l in levels]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(levels, rel, color=BLUE, width=0.6)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    for b, r in zip(bars, rel):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("Driver age band", fontsize=9, color=MUTED)
    ax.set_ylabel(f"Frequency relativity (base: {base})", fontsize=9, color=MUTED)
    ax.set_title("Poisson GLM: driver age relativities, other factors held fixed",
                 loc="left", fontsize=12, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "glm_driver_age_relativities.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)

    # 4. Boosting feature importance (share of total gain).
    imp = pd.Series(gbm_freq.feature_importance("gain"), index=gbm_freq.feature_name())
    imp = (100 * imp / imp.sum()).sort_values()
    imp.sort_values(ascending=False).round(2).to_csv(OUT / "frequency_gbm_importance.csv",
                                                     header=["gain_pct"], index_label="feature")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.values, color=ORANGE, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Share of total split gain", fontsize=9, color=MUTED)
    ax.set_title("Gradient boosting: what drives predicted frequency", loc="left",
                 fontsize=12, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "gbm_feature_importance.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    pol, sev = load()
    glm_f, gbm_f, rounds_f, ref = fit_frequency(pol)
    glm_s, gbm_s, rounds_s = fit_severity(pol, sev)
    metrics, test = evaluate(pol, sev)
    metrics["gbm_rounds"] = {"frequency": int(rounds_f), "severity": int(rounds_s)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "frequency_glm_summary.txt").write_text(str(glm_f.summary()))
    (OUT / "severity_glm_summary.txt").write_text(str(glm_s.summary()))
    figures(test, glm_f, gbm_f, ref)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
