# Research brief — Local competition and UK retail petrol prices

**Title:** Does Local Competition Lower Petrol Prices? Evidence from UK Fuel Finder  
**Discipline:** Competition economics / industrial organisation  
**Date:** 17 November 2026  
**Status:** Research design and code complete. Empirical estimates pending retrieval/execution of the live public CSVs. No figures have been fabricated.

## Executive summary

UK petrol retailing is a natural setting to investigate local geographic competition because consumers can choose between nearby filling stations, and station-level prices and coordinates have recently become available under the UK government's Fuel Finder reporting regime. The CMA has previously noted that retail fuel competition is predominantly local and that distance to competing sites, supermarket presence and local conditions may help explain geographic price differences. This study constructs measures of the number of other forecourts within 3, 5 and 10 km and relates them to station-level E10 pump prices on a common date, conditional on observable retailer attributes and location fixed effects.

**Hypothesis:** All else equal, petrol stations facing a larger number of nearby alternative forecourts tend to set lower prices. **Counter-hypothesis:** Demand and cost differences, urban density, and supply-chain choices may overwhelm this relationship or reverse it. The empirical evidence will determine the direction and precision of the association.

## Data

- DESNZ Fuel Finder retail prices and station attributes, archived by https://fuelpricetoday.co.uk/data.
- `stations.csv`: 8,155 station-directory rows listed on the source as of 17 November 2026; station IDs, lat/lon, postcode areas, brands, supermarket/motorway status.
- `history-2026-11.csv.gz`: one observed price per station, fuel and day; restrict to E10 on 16 November 2026.
- Cleaning: non-motorway, public, open sites with valid coordinates, price ages <=30 days and prices within 25 p/l of the day's median.

## Economic framework

In local Bertrand competition with spatially differentiated products, each site competes more strongly with accessible substitutes when travel costs are low. Counts of other geographically close retailers are an intuitive proxy for substitution opportunities, though not direct estimates of cross-price elasticities. Supermarket operators may use fuel pricing to attract shoppers; geographic market boundaries should be tested rather than presumed.

## Specification

\[
P_i=\alpha+\beta\ln(1+N_{i,5km})+\gamma S_i+\delta SM_{i,5km}+\mu_{brand(i)}+\lambda_{postcodearea(i)}+\varepsilon_i
\]

where `P_i` is E10 pence/litre, `N_i,5km` is the number of other nearby eligible forecourts, `S_i` is the station's supermarket status, and `SM_i,5km` indicates another supermarket station within 5 km. `mu` and `lambda` are categorical fixed effects. Heteroskedasticity-robust standard errors are used in the first two specifications; the preferred specification clusters errors at the postcode-area level.

## What results would mean

A negative `beta` would mean stations with more nearby alternatives have lower *conditional* average prices. A positive or imprecisely estimated `beta` would require discussion of geographic costs, omitted demand variables, supermarket pricing, measurement and model limitations. In either event, **cross-sectional estimates do not establish causation**. A doubling of `(1+N)` corresponds to `beta × ln(2)` pence/litre in predicted prices.

## Analyses delivered by the notebook

1. Full sample descriptive statistics and mean prices across 5 local-competition bins.
2. Unadjusted OLS and models with location/retailer controls.
3. Preferred postcode-area-FE specification with clustered standard errors.
4. 3-km and 10-km alternative market definitions.
5. Non-supermarket-only and 7-day-updated-price robustness checks.
6. Two figures, regression CSV, processed sample and automatically populated Markdown report.

## Limitations

A station's presence is not exogenous; locations may respond to demand and profit expectations. A 5 km straight-line radius is not the same as the CMA's driving-time-based local analysis. Brands do not necessarily represent independent ultimate owners. Prices are reported and unweighted by volumes. Cost variation, wholesale contracts, wages, rents and traffic are not controlled perfectly. Station geocodes, reported timestamps and the updating station directory can introduce measurement error. Consequently, this is an exploratory applied-econometrics project, **not** a causal competition impact assessment.

## References

- [UK DESNZ, Fuel Finder public price-data guidance (2 Feb 2026)](https://www.gov.uk/guidance/access-the-latest-fuel-prices-and-forecourt-data-via-api-or-email)
- [Fuel Price Today, daily station-level price archive and data dictionary](https://fuelpricetoday.co.uk/data)
- [UK CMA, Road Fuel Review (2022)](https://www.gov.uk/government/publications/road-fuel-review/road-fuel-review)
- [UK CMA, Enhanced Road Fuel Monitoring Report (June 2026)](https://www.gov.uk/government/publications/enhanced-road-fuel-monitoring-report-june-2026)

Attribution: Contains public sector information licensed under the Open Government Licence v3.0. Data via fuelpricetoday.co.uk.
