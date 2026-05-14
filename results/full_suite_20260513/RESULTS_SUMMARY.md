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
  - n_good: 495/500
  - Distance MAE: 0.5044 m
  - Distance RMSE: 0.9273 m
  - Distance P95 abs error: 1.3989 m
  - Velocity MAE: 0.1304 m/s
  - Velocity RMSE: 0.1771 m/s
  - Velocity P95 abs error: 0.3727 m/s
- Seed 456: `eval/seed456/eval_summary.json`
  - n_good: 498/500
  - Distance MAE: 0.4669 m
  - Distance RMSE: 0.7291 m
  - Distance P95 abs error: 1.4394 m
  - Velocity MAE: 0.1344 m/s
  - Velocity RMSE: 0.1808 m/s
  - Velocity P95 abs error: 0.3610 m/s
- Seed 789: `eval/seed789/eval_summary.json`
  - n_good: 495/500
  - Distance MAE: 0.4657 m
  - Distance RMSE: 0.8029 m
  - Distance P95 abs error: 1.2588 m
  - Velocity MAE: 0.1306 m/s
  - Velocity RMSE: 0.1761 m/s
  - Velocity P95 abs error: 0.3645 m/s

## Aggregate Across 3 Seeds

- Good cases: 1488 / 1500 (99.20%)
- Mean distance MAE: 0.4790 m
- Mean distance RMSE: 0.8198 m
- Mean distance P95 abs error: 1.3657 m
- Mean velocity MAE: 0.1318 m/s
- Mean velocity RMSE: 0.1780 m/s
- Mean velocity P95 abs error: 0.3661 m/s

## Notes

- Runtime logs were intentionally excluded.
- The canonical map used for eval is `map/inverse_map.csv`.
- Each eval plot now uses two subplots: distance error (top) and velocity error (bottom).
