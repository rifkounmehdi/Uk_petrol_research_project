#!/usr/bin/env python3
"""Local competition and UK pump prices: reproducible cross-sectional analysis.

Data: fuelpricetoday.co.uk (UK DESNZ Fuel Finder), Open Government Licence v3.0.
Methods: geodesic forecourt proximity, OLS with geographic & brand controls,
postcode-area-clustered standard errors; robustness checks.

Usage:
    pip install -r requirements.txt
    python analysis.py
    python analysis.py --stations stations.csv --history history-2026-11.csv.gz

The program only makes factual empirical claims after it downloads and analyses
actual data. It does not use simulated observations in the deliverable.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree
import statsmodels.formula.api as smf

BASE = "https://fuelpricetoday.co.uk/data"
SNAPSHOT_DATE = "2026-11-16"    # complete common observation day
EARTH_RADIUS_KM = 6371.0088
MAJOR_NON_SUPERMARKET_BRANDS = {"BP", "SHELL", "ESSO", "TEXACO", "JET", "GULF", "TOTAL", "ESSAR"}
RADIUS_LIST = (3, 5, 10)


def fetch_once(url: str, dest: Path, minimum_bytes: int = 100_000) -> None:
    """Download and cache exact raw source bytes for a reproducible local run."""
    if dest.exists():
        print(f"Using cached data: {dest}")
        return
    print(f"Downloading {url}")
    response = requests.get(url, headers={"User-Agent": "IndependentEconomicResearch/1.0"}, timeout=90)
    response.raise_for_status()
    content = response.content
    # A CDN can transparently decompress a .gz response; preserve gzip on disk.
    if dest.suffix == ".gz" and content[:2] != b"\x1f\x8b":
        content = gzip.compress(content)
    if len(content) < minimum_bytes:
        raise RuntimeError(f"File looks incomplete ({len(content):,} bytes): {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_stations(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "node_id", "latitude", "longitude", "postcode_area", "motorway",
        "supermarket", "members_only", "temporarily_closed", "brand", "nation",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"Missing station columns: {sorted(missing)}")

    df = raw.copy()
    df["node_id"] = df["node_id"].astype(str)
    df = df.drop_duplicates(subset=["node_id"])
    for col in ("motorway", "supermarket", "members_only", "temporarily_closed"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["postcode_area"] = df["postcode_area"].fillna("").astype(str).str.upper().str.strip()
    df["nation"] = df["nation"].fillna("").astype(str).str.strip()
    df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    df["brand_upper"] = df["brand"].str.upper()
    df["brand_group"] = df["brand_upper"].apply(
        lambda b: b if b in MAJOR_NON_SUPERMARKET_BRANDS else "OTHER"
    )
    start_n = len(df)
    # Broad UK bounding box; includes Northern Ireland, Scottish islands and Cornwall.
    mask = (
        df["latitude"].between(49.5, 61.5)
        & df["longitude"].between(-9.0, 2.5)
        & df["postcode_area"].ne("")
        & df["nation"].ne("")
        & df["motorway"].eq(0)
        & df["members_only"].eq(0)
        & df["temporarily_closed"].eq(0)
    )
    df = df.loc[mask].reset_index(drop=True)
    info = {"raw_unique_stations": start_n, "eligible_station_universe": len(df)}
    if len(df) < 30:
        raise ValueError("Too few eligible forecourts; check source data and column labels")
    return df, info


def add_local_competition(stations: pd.DataFrame) -> pd.DataFrame:
    """Count other eligible forecourts, regardless of operator, within radius.

    Distance is the shortest path over the earth's surface, not driving distance.
    Same-brand forecourts may be under different ownership; we deliberately label
    the measure as 'nearby forecourts', not independent legal competitors.
    """
    df = stations.copy()
    rad_coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(rad_coords, metric="haversine")
    supermarket = df["supermarket"].to_numpy(dtype=np.int32)

    for radius in RADIUS_LIST:
        neighbours = tree.query_radius(rad_coords, r=radius / EARTH_RADIUS_KM)
        counts = np.empty(len(df), dtype=int)
        supermarkets_nearby = np.empty(len(df), dtype=int)
        for i, ix in enumerate(neighbours):
            ix = ix[ix != i]  # self excluded
            counts[i] = len(ix)
            supermarkets_nearby[i] = int(supermarket[ix].any())
        df[f"nearby_{radius}km"] = counts
        df[f"supermarket_nearby_{radius}km"] = supermarkets_nearby
        df[f"log_nearby_{radius}km"] = np.log1p(counts)
    return df


def prepare_prices(raw: pd.DataFrame, stations: pd.DataFrame,
                   snapshot_date: str, freshness_days: int) -> tuple[pd.DataFrame, dict]:
    cols = {"node_id", "fuel", "date", "price_ppl", "price_last_updated"}
    absent = cols.difference(raw.columns)
    if absent:
        raise KeyError(f"Missing price history columns: {sorted(absent)}")

    x = raw.loc[(raw["fuel"] == "E10") & (raw["date"] == snapshot_date)].copy()
    x["node_id"] = x["node_id"].astype(str)
    x = x.drop_duplicates(subset=["node_id"], keep="last")
    x["price_ppl"] = pd.to_numeric(x["price_ppl"], errors="coerce")
    x["price_last_updated"] = pd.to_datetime(x["price_last_updated"], utc=True, errors="coerce")
    end_of_day = pd.Timestamp(snapshot_date, tz="UTC") + pd.Timedelta(days=1)
    x["age_days"] = (end_of_day - x["price_last_updated"]).dt.total_seconds() / 86400

    # Exclude unreported/very old prices. The last-change timestamp may be old
    # because a genuine pump price has remained unchanged: sensitivity tested.
    x = x.loc[x["price_ppl"].notna() & x["age_days"].between(0, freshness_days)]
    median = x["price_ppl"].median()
    if pd.isna(median):
        raise ValueError("No E10 prices on requested date. Check downloaded history file.")
    # Conservative outlier screen, aligned with the data republisher's convention.
    x = x.loc[x["price_ppl"].between(median - 25.0, median + 25.0)]
    sample = stations.merge(x[["node_id", "price_ppl", "age_days"]], on="node_id", how="inner")
    info = {"priced_E10_stations_before_geography": len(x), "regression_sample": len(sample),
            "E10_median_for_outlier_screen": float(median), "freshness_days": freshness_days}
    if len(sample) < 30:
        raise ValueError("Fewer than 30 usable observations. Check snapshot_date and file.")
    return sample, info


def estimate(formula: str, data: pd.DataFrame, cluster: bool = True):
    estimator = smf.ols(formula, data=data)
    if cluster:
        return estimator.fit(cov_type="cluster", cov_kwds={"groups": data["postcode_area"]})
    return estimator.fit(cov_type="HC3")


def model_suite(df: pd.DataFrame) -> dict:
    base = "price_ppl ~ log_nearby_5km"
    controls = " + supermarket + supermarket_nearby_5km + C(brand_group)"
    main = "price_ppl ~ log_nearby_5km" + controls + " + C(postcode_area)"
    results = {
        "unadjusted": estimate(base, df, cluster=False),
        "national_controls": estimate("price_ppl ~ log_nearby_5km" + controls + " + C(nation)", df, cluster=False),
        "preferred": estimate(main, df, cluster=True),
        "3km": estimate(main.replace("log_nearby_5km", "log_nearby_3km")
                         .replace("supermarket_nearby_5km", "supermarket_nearby_3km"), df, cluster=True),
        "10km": estimate(main.replace("log_nearby_5km", "log_nearby_10km")
                          .replace("supermarket_nearby_5km", "supermarket_nearby_10km"), df, cluster=True),
    }
    nonsuper = df.loc[df["supermarket"].eq(0)].copy()
    if len(nonsuper) > 100 and nonsuper["postcode_area"].nunique() > 10:
        nonsuper_f = "price_ppl ~ log_nearby_5km + supermarket_nearby_5km + C(brand_group) + C(postcode_area)"
        results["non_supermarket_only"] = estimate(nonsuper_f, nonsuper, cluster=True)
    return results


def effect_row(model, var: str) -> dict:
    beta = float(model.params[var])
    low, high = map(float, model.conf_int().loc[var])
    return {"N": int(model.nobs), "beta": beta, "ci_low": low,
            "ci_high": high, "pvalue": float(model.pvalues[var]),
            "r_squared": float(model.rsquared), "groups": None}


def coefficient_table(results: dict) -> pd.DataFrame:
    rows = []
    for label, model in results.items():
        var = ("log_nearby_3km" if label == "3km" else
               "log_nearby_10km" if label == "10km" else "log_nearby_5km")
        rows.append({"model": label, "variable": var, **effect_row(model, var)})
    return pd.DataFrame(rows)


def bin_table(sample: pd.DataFrame) -> pd.DataFrame:
    labels = ["0", "1–2", "3–5", "6–10", "11+"]
    x = sample.copy()
    x["competition_bin"] = pd.cut(x["nearby_5km"],
        bins=[-0.5, 0.5, 2.5, 5.5, 10.5, np.inf], labels=labels)
    return (x.groupby("competition_bin", observed=False)["price_ppl"]
            .agg(forecourts="count", avg_E10_ppl="mean", sd_E10_ppl="std")
            .reset_index())


def make_charts(sample: pd.DataFrame, bins: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    use = bins.loc[bins["forecourts"] > 1].copy().reset_index(drop=True)
    means = use["avg_E10_ppl"].to_numpy(dtype=float)
    interval = 1.96 * use["sd_E10_ppl"].to_numpy(dtype=float) / np.sqrt(use["forecourts"].to_numpy(dtype=float))
    ax.errorbar(np.arange(len(use)), means, yerr=interval, fmt="o", capsize=4)
    ax.set_xticks(np.arange(len(use)), use["competition_bin"].astype(str).tolist())
    ax.set(xlabel="Other non-motorway forecourts within 5 km",
           ylabel="Mean E10 pump price (pence/litre)",
           title="Local forecourt density and E10 price | 16 Nov 2026")
    ax.set_ylim(float((means - interval).min() - 2), float((means + interval).max() + 3))
    for i, row in enumerate(use.itertuples(index=False)):
        ax.text(i, row.avg_E10_ppl + interval[i] + 0.5, f"n={row.forecourts:,}",
                ha="center", va="bottom", fontsize=9)
    ax.text(0, -0.15, "Unadjusted mean prices; error bars are ±1.96 standard errors (not cluster-adjusted)",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure1_prices_by_local_competition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    groups = [sample.loc[sample["supermarket"].eq(v), "price_ppl"] for v in (0, 1)]
    ax.boxplot(groups, tick_labels=["Other forecourts", "Supermarket forecourts"], showfliers=False)
    ax.set(ylabel="E10 pump price (pence/litre)",
           title="Distribution of UK petrol prices by forecourt type | 16 Nov 2026")
    fig.tight_layout()
    fig.savefig(out / "figure2_supermarket_vs_other.png", dpi=180)
    plt.close(fig)


def p_significance(beta: float, low: float, high: float) -> str:
    if low < 0 and high < 0:
        return "negative and statistically distinguishable from zero at the 5% level"
    if low > 0 and high > 0:
        return "positive and statistically distinguishable from zero at the 5% level"
    return "not statistically distinguishable from zero at the 5% level"


def report_md(sample: pd.DataFrame, quality: dict, results: dict, coef: pd.DataFrame,
              bins: pd.DataFrame, snapshot_date: str, inputs: dict) -> str:
    main = results["preferred"]
    eff = effect_row(main, "log_nearby_5km")
    # Regression uses log(1+N); a doubling of 1+N changes price by beta*log(2).
    doubled = eff["beta"] * math.log(2)
    low = eff["ci_low"] * math.log(2)
    high = eff["ci_high"] * math.log(2)
    raw_supermarket_mean = sample.groupby("supermarket")["price_ppl"].mean()
    raw_gap = (raw_supermarket_mean.get(1, float("nan")) -
               raw_supermarket_mean.get(0, float("nan")))
    nearest_super = effect_row(main, "supermarket_nearby_5km")

    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    coef_header = "| Specification | N | β: log(1+nearby stations), p/l | 95% CI | p-value | R² |\n|---|---:|---:|---|---:|---:|"
    coef_lines = [coef_header]
    for r in coef.itertuples(index=False):
        coef_lines.append(f"| {r.model.replace('_', ' ')} | {r.N:,} | {r.beta:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {fmt_p(r.pvalue)} | {r.r_squared:.3f} |")
    bins_lines = ["| Other forecourts within 5 km | N | Average E10 price (p/l) |",
                  "|---|---:|---:|"]
    for row in bins.itertuples(index=False):
        if row.forecourts:
            bins_lines.append(f"| {row.competition_bin} | {row.forecourts:,} | {row.avg_E10_ppl:.2f} |")

    ids = f"{inputs['stations_hash'][:12]} / {inputs['history_hash'][:12]}"
    return f"""# Does local competition affect petrol prices in the UK?

**Independent economic research | 16 November 2026 station-day snapshot | E10 unleaded**

## Executive summary

This study examines the association between local forecourt density and retail petrol prices using a cross-section of **{len(sample):,} UK filling stations**. Prices are drawn from Fuel Finder observations on **{snapshot_date}**, with geographic and retailer details from a linked station directory. Nearby forecourts are measured by **great-circle distance** (3, 5 and 10 km); the main specification controls for brand type, whether the station is a supermarket, presence of a nearby supermarket, and postcode-area fixed effects.

In the preferred specification, the estimated coefficient on log(1 + nearby forecourts within 5 km) is **{eff['beta']:+.3f} pence/litre** (95% CI [{eff['ci_low']:+.3f}, {eff['ci_high']:+.3f}], p={fmt_p(eff['pvalue'])}; N={eff['N']:,}, R²={eff['r_squared']:.3f}). This corresponds to **{doubled:+.2f} pence/litre** (95% CI [{low:+.2f}, {high:+.2f}]) for a doubling of (1 + number of nearby forecourts), holding model controls constant. The estimated association is **{p_significance(eff['beta'], eff['ci_low'], eff['ci_high'])}**.

**Interpretation:** These are cross-sectional associations, **not causal estimates** of the effect of adding competitors or evidence of anticompetitive conduct. They do not measure retailer margins or consumer welfare.

## 1. Economic motivation and hypothesis

In a standard differentiated-products/Hotelling framework, closer substitutes may constrain local retail prices. Retail fuel stations nevertheless differ in delivery costs, local incomes, format, convenience and land costs. Supermarkets may also price aggressively to attract grocery shoppers. The UK's Competition and Markets Authority (CMA) has discussed the local nature of road-fuel competition and the importance of nearby competitors, supermarkets and rurality. The testable hypothesis is that stations with more nearby forecourts tend to charge less per litre, conditional on observable controls.

## 2. Data and sample construction

- **Source:** Daily station-level retail prices, archived by fuelpricetoday.co.uk from the UK Department for Energy Security and Net Zero (DESNZ) Fuel Finder scheme, plus its station directory.
- **Sample date:** {snapshot_date} (one common day, to avoid mixing fuel price levels from different periods).
- **Price:** E10 standard unleaded petrol, pence per litre. The file reports the last price observed for each station on that day, not an average transaction price.
- **Market geometry:** Other eligible non-motorway forecourts within 3, 5 and 10 km (great-circle distance). Motorway services, membership-only forecourts and temporarily closed sites are excluded from the geographic universe; the station itself is excluded from neighbour counts.
- **Freshness:** Reported price change must be within 30 days of the snapshot. A price unchanged for more than 30 days might still be genuine; the filter is a conservative quality screen, not proof of error.
- **Outliers:** Prices more than 25 pence/litre above/below the day's median excluded as a pre-specified screen.
- **Coverage:** {quality['raw_unique_stations']:,} unique stations initially; {quality['eligible_station_universe']:,} eligible stations for local-density computation; {quality['priced_E10_stations_before_geography']:,} valid E10 prices after time/outlier screening; {len(sample):,} matched observations in regressions.
- **Source provenance:** Cached input file SHA-256 prefixes (stations/history): {ids}. The source station directory is updated over time; preserve the cached copies for exact replication.

## 3. Descriptive statistics

Mean E10 price: **{sample['price_ppl'].mean():.2f} pence/litre**; median: **{sample['price_ppl'].median():.2f}**. Mean number of other non-motorway stations within 5 km: **{sample['nearby_5km'].mean():.1f}**. Supermarket share of sampled stations: **{100 * sample['supermarket'].mean():.1f}%**.

The unadjusted supermarket-minus-other-station mean price difference is **{raw_gap:+.2f} pence/litre** (descriptive; not causal).

{chr(10).join(bins_lines)}

![Mean E10 prices by nearby-station count](figure1_prices_by_local_competition.png)

![E10 price distributions by station type](figure2_supermarket_vs_other.png)

## 4. Empirical approach

The preferred specification is:

`E10_price_i = alpha + beta * log(1 + nearby_forecourts_5km_i) + gamma * supermarket_station_i + delta * supermarket_within_5km_i + brand_group_FE + postcode_area_FE + epsilon_i`

Observations are filling stations. OLS standard errors for the preferred specification are **clustered at the postcode-area level** to permit correlated residuals within these geographic groups. Baseline models report HC3 heteroskedasticity-robust standard errors. The coefficient of interest is beta; the logarithm allows diminishing marginal association as station density increases. Because the regressor uses `log(1 + N)`, interpret a doubling of **1+N**, rather than a literal doubling of N.

The location of stations is not random. Neither postcode-area fixed effects nor brand controls identify a causal effect; unobserved costs, local demand and urbanisation could drive both prices and station entry.

## 5. Regression results

{chr(10).join(coef_lines)}

The preferred model is the **postcode-area-fixed-effects 5 km regression**. Estimates at 3 km and 10 km test sensitivity to the approximate geographic market definition; the non-supermarket-only estimate tests dependence on supermarket pricing models.

A nearby supermarket is associated with **{nearest_super['beta']:+.2f} pence/litre** (95% CI [{nearest_super['ci_low']:+.2f}, {nearest_super['ci_high']:+.2f}]; p={fmt_p(nearest_super['pvalue'])}), conditional on controls. This is a correlation, not a causal effect or merger simulation.

## 6. Limitations and next steps

1. **Endogeneity:** Retailers choose sites and prices; high-demand areas may have more stations and different costs. A cross-section cannot cleanly separate competitive pressure from local market fundamentals.
2. **Geographic markets:** 5 km 'as the crow flies' does not account for roads, driving time or travel barriers. CMA market work commonly considers drive-time competition, so a network-based specification would be preferable.
3. **Ownership:** Station-brand labels are not a substitute for ultimate-ownership data. Several nearby stations may be controlled by one owner; the analysis counts forecourts, not independent firms.
4. **Missing controls:** Wholesale supply contracts, fuel volumes, property costs, local wages, local traffic and income may all affect prices.
5. **Data timing:** The source contains reported pump prices, not verified transaction-level sales. Timestamp-age filters can select against retailers with stable genuine prices.
6. **Inference:** Postcode-area clustering partly addresses spatial dependence but not all cross-border spatial correlation. The sample is station-weighted rather than sales-weighted.

**Higher-quality extension:** Link geocoded stations to road-network drive times, ONS small-area socioeconomic indicators and reliable operator ownership. A merger/event-study design or credible exogenous change in competition would be required for stronger causal claims.

## 7. Competition-policy interpretation

This analysis can indicate whether local price patterns are consistent with competitive pressure and where further investigation might be useful. It **cannot** establish collusion, a competition-law infringement, a causal harm estimate, or a remedy effect. A consulting-grade merger assessment would instead assess diversion, local catchment areas, ownership overlaps, relevant cost evidence and a defensible counterfactual.

## 8. Sources and data licence

- UK DESNZ, **Fuel Finder**: https://www.gov.uk/guidance/access-the-latest-fuel-prices-and-forecourt-data-via-api-or-email
- Fuel Price Today (2026), **UK fuel prices by filling station, daily**, https://fuelpricetoday.co.uk/data (accessed on the day of the analysis). The republisher archives the official scheme and publishes data documentation.
- UK Competition and Markets Authority, **Road fuel review**, https://www.gov.uk/government/publications/road-fuel-review/road-fuel-review
- UK Competition and Markets Authority, **Enhanced road fuel monitoring report: June 2026**, https://www.gov.uk/government/publications/enhanced-road-fuel-monitoring-report-june-2026

Contains public sector information licensed under the **Open Government Licence v3.0**. Data via fuelpricetoday.co.uk.

## CV description (use only after running and checking analysis)

**Independent Economic Research | Local Competition and UK Petrol Pricing | Nov 2026**

- Analysed **{len(sample):,} UK petrol forecourts** using geospatial competitor counts and station-level Fuel Finder price data; applied Python and econometric methods to examine local pricing patterns.
- Estimated OLS models with **brand and postcode-area fixed effects**, postcode-area-clustered standard errors and **3/5/10-km market-definition robustness checks**; summarised findings and limitations in a competition-policy research note.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--stations", type=Path, default=None,
                        help="Optional local stations.csv to avoid downloading")
    parser.add_argument("--history", type=Path, default=None,
                        help="Optional local history-2026-11.csv.gz")
    parser.add_argument("--date", type=str, default=SNAPSHOT_DATE,
                        help="Snapshot date in November 2026, YYYY-MM-DD")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cached_source_data"
    cache.mkdir(exist_ok=True)
    stations_path = args.stations or (cache / "stations.csv")
    history_path = args.history or (cache / "history-2026-11.csv.gz")
    if args.stations is None:
        fetch_once(f"{BASE}/stations.csv", stations_path)
    if args.history is None:
        fetch_once(f"{BASE}/history-2026-11.csv.gz", history_path)

    print("Loading station directory and price history...")
    stations_raw = pd.read_csv(stations_path, dtype={"node_id": "str"}, low_memory=False)
    prices_raw = pd.read_csv(history_path, dtype={"node_id": "str"}, low_memory=False)
    stations, quality = clean_stations(stations_raw)
    stations = add_local_competition(stations)
    sample, sample_quality = prepare_prices(prices_raw, stations, args.date, freshness_days=30)
    quality.update(sample_quality)

    models = model_suite(sample)
    # Freshness-screen sensitivity; computed with unchanged geographic universe.
    fresh7, _ = prepare_prices(prices_raw, stations, args.date, freshness_days=7)
    if len(fresh7) > 100 and fresh7["postcode_area"].nunique() > 10:
        fresher = ("price_ppl ~ log_nearby_5km + supermarket + supermarket_nearby_5km"
                   " + C(brand_group) + C(postcode_area)")
        models["prices_updated_7d"] = estimate(fresher, fresh7, cluster=True)

    coef = coefficient_table(models)
    bins = bin_table(sample)
    coef.to_csv(out / "regression_results.csv", index=False)
    bins.to_csv(out / "descriptives_by_competition.csv", index=False)
    sample[["node_id", "brand", "nation", "postcode_area", "latitude", "longitude",
            "price_ppl", "age_days", "supermarket", "nearby_3km", "nearby_5km", "nearby_10km",
            "supermarket_nearby_5km"]].to_csv(out / "analysis_sample.csv", index=False)
    make_charts(sample, bins, out)
    inputs = {"stations_hash": hash_file(stations_path), "history_hash": hash_file(history_path)}
    (out / "research_report.md").write_text(
        report_md(sample, quality, models, coef, bins, args.date, inputs), encoding="utf-8"
    )
    print("\n=== Results (coefficients in pence/litre per log-point of station density) ===")
    print(coef[["model", "N", "beta", "ci_low", "ci_high", "pvalue", "r_squared"]].to_string(index=False))
    print(f"\nCompleted {len(sample):,} observed forecourts on {args.date}.")
    print(f"Files written to {out.resolve()}")
    print("Read research_report.md for interpretation and research limitations.")


if __name__ == "__main__":
    main()
