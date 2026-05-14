# Full Suite Results (2026-05-13)

This folder contains the final requested run set:

- Extra-dense PID map generation
- Rigorous inverse-map evaluation: 500 cases x 3 seeds
- Log files excluded by policy

## Map Run

- Source run id: `pid_map_extra_dense_full_20260513`
- Manifest: `map/run_manifest.json`
- Grid:
  - lead speed: 2.0 to 10.0 m/s, step 0.25 (33 values)
  - initial gap: 8.0 to 32.0 m, step 0.5 (49 values)
  - target gap: 8.0 to 32.0 m, step 0.5
- Trials: 1617
- Forward cells with data: 833 / 1617

## Rigorous Eval Runs

- Seed 123: `eval/seed123/eval_summary.json`
  - n_good: 496/500
  - MAE: 0.4882 m
  - RMSE: 0.8217 m
  - P95 abs error: 1.3862 m
- Seed 456: `eval/seed456/eval_summary.json`
  - n_good: 498/500
  - MAE: 0.4523 m
  - RMSE: 0.7097 m
  - P95 abs error: 1.4333 m
- Seed 789: `eval/seed789/eval_summary.json`
  - n_good: 495/500
  - MAE: 0.4785 m
  - RMSE: 0.8140 m
  - P95 abs error: 1.3793 m

## Aggregate Across 3 Seeds

- Good cases: 1489 / 1500 (99.27%)
- Mean MAE: 0.4730 m
- Mean RMSE: 0.7818 m
- Mean P95 abs error: 1.3996 m

## Notes

- Runtime logs were intentionally excluded.
- The canonical map used for eval is `map/inverse_map.csv`.
