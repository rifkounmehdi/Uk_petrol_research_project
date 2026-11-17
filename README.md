# Does local competition affect UK petrol prices?

**Independent Economic Research — applied competition economics**  
**Reference date:** 16 November 2026  
**Output:** Text-based research report (Markdown), regression CSV, analysis sample, and plots. **No PDF.**

## Research question

Are E10 petrol stations with more nearby forecourts associated with lower retail prices, after controlling for retailer attributes and geography?

This is a **cross-sectional descriptive / associational econometric exercise**, not a causal identification design or proof of a competition problem. The main result is estimated, not assumed.

## Run locally

1. Install Python 3.10+.
2. Install libraries: `pip install -r requirements.txt`
3. Run: `python analysis.py`
4. Open `results/research_report.md` (fully populated, with real estimated coefficients), `results/regression_results.csv` and generated PNG charts.

The script downloads and caches the source files automatically. You can also pass local copies: `python analysis.py --stations stations.csv --history history-2026-11.csv.gz`.

## Run in Google Colab

Upload `UK_Fuel_Competition_Research.ipynb` to Colab and click **Runtime → Run all**. It installs dependencies, downloads the source, runs the estimates and writes `results/research_report.md`. For exact replication later, preserve the `results/cached_source_data` files because the station directory is updated over time.

## Model

```
E10_price_i = alpha
            + beta log(1 + forecourts_within_5km_i)
            + gamma own_supermarket_i
            + delta any_other_supermarket_within_5km_i
            + brand_group fixed effects
            + postcode_area fixed effects
            + epsilon_i
```

OLS standard errors for the preferred model are clustered by postcode area. Compare unadjusted and more fully controlled estimates, 3-km and 10-km alternatives, non-supermarket stations, and prices updated within 7 days.

**Geometry:** Great-circle distance rather than road-network driving time.  
**Competitor proxy:** Other nearby eligible forecourts, not unique legal owners; location and ownership limitations must be acknowledged.  
**Sample:** E10 prices for 16 November 2026, reports no more than 30 days old, excludes motorway / membership-only / temporarily closed stations, filters ±25 pence around date median to screen anomalous prices.  
**Interpretation:** Coefficient `beta` is pence/litre per unit change in log(1 + count). A doubling of (1 + count) predicts `beta * ln(2)` pence/litre, not a causal effect.

## Data references and attribution

- UK DESNZ Fuel Finder: https://www.gov.uk/guidance/access-the-latest-fuel-prices-and-forecourt-data-via-api-or-email
- fuelpricetoday.co.uk (2026), UK fuel prices by filling station, daily, from Fuel Finder, https://fuelpricetoday.co.uk/data (accessed on running). It includes the 2026-11 price-history CSV and the geocoded station directory.
- UK CMA, Road fuel review: https://www.gov.uk/government/publications/road-fuel-review/road-fuel-review

Contains public sector information licensed under the **Open Government Licence v3.0**. Data via fuelpricetoday.co.uk.

## CV-ready description — use after running and checking results

**Independent Economic Research — UK Petrol Retail Competition (Nov 2026)**
- Analysed station-level UK petrol prices and geocoded forecourt locations using Python to evaluate the relationship between local market density and retail pricing.
- Estimated OLS regressions with brand and postcode-area fixed effects, clustered standard errors and alternative geographic market definitions (3, 5 and 10 km); documented robustness and causal limitations.

## Important honesty note

The provided files were authored and code-tested without access to the external data download in the author's execution environment. **No regression coefficient has been invented or prefilled.** Running the notebook produces the real empirical report from the source CSVs. Inspect the model results and plots before claiming the study is finished on a CV.
