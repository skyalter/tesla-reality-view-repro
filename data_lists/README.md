# Which Argoverse 2 Sensor logs are used for what

| File | Split | Logs | Use |
|---|---|---|---|
| `av2_train_log_city.tsv` | train | 700 | all training logs with their city |
| `av2_val_log_city.tsv` | val | 150 | all validation logs with their city |
| `av2_test_log_city.tsv` | test | 150 | all test logs with their city (not used; the test split has no 3D boxes) |
| `av2_val_report20.txt` | val | 20 | reporting: final numbers only, never used for tuning |
| `av2_val_dev10.txt` | val | 10 | tuning: loss weights, scales, thresholds |
| `av2_train_order.tsv` | train | 700 | all training logs in a fixed ranking, with the numbers the ranking used |
| `av2_train_stage1.txt` | train | 150 | training: the first 150 logs of the ranking |

The three lists that are used never overlap.

- `av2_val_report20.txt` is the first 20 validation logs in sorted order.
- `av2_val_dev10.txt` is produced by `scripts/select_val_dev.py`: validation logs outside the reporting
  list, one per city in turn, ordered within a city by a fixed hash.
- `av2_train_order.tsv` and `av2_train_stage1.txt` are produced by `scripts/select_train_logs.py` from the
  ego poses of all 700 training logs. Logs that drive less than 10 m are dropped (14 of 700). Places are
  handed out one at a time across cities, and inside a city across slow / medium / fast logs, each weighted
  by its size to the power 0.75. Larger training sets take more logs from the same ranking, so the 150
  logs stay a subset. The first 150 logs are 40 minutes of driving: Miami 48, Pittsburgh 48,
  Washington 22, Detroit 19, Austin 8, Palo Alto 5.

Do not re-rank after training has started: changing the rule or the seed changes which logs are in the set.
