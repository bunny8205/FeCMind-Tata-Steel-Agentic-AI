# Data Sources

## Steel demo data
Synthetic steel maintenance data is generated for four industrial assets:
MTR-204, GBX-17, PMP-09, and HPP-12.

It includes sensor logs, maintenance history, failure reports, delay logs,
incident records, spare inventory, feedback logs, and a digital logbook.

## Public AI4I benchmark
The AI4I 2020 Predictive Maintenance dataset is used only as an external
public benchmark to demonstrate ML validation.

Leakage control:
- `Machine failure` is used only as the supervised target label.
- Failure subtype labels such as TWF, HDF, PWF, OSF, and RNF are not used as model features.
- Sensor proxy fields are engineered only from non-target process variables.
- The steel app decision layer uses a separate steel demo model plus operational rules.
