# Which Argoverse 2 Sensor logs are used for what

| File | Split | Logs | Use |
|---|---|---|---|
| `av2_train_log_city.tsv` | train | 700 | all training logs with their city |
| `av2_val_log_city.tsv` | val | 150 | all validation logs with their city |
| `av2_test_log_city.tsv` | test | 150 | all test logs with their city (not used; the test split has no 3D boxes) |
| `av2_val_report20.txt` | val | 20 | reporting: final numbers only, never used for tuning |
| `av2_val_dev10.txt` | val | 10 | tuning: loss weights, scales, thresholds |
| `av2_train_stage1.txt` | train | 150 | training (not generated yet) |

The three lists that are used never overlap.

- `av2_val_report20.txt` is the first 20 validation logs in sorted order.
- `av2_val_dev10.txt` is produced by `scripts/select_val_dev.py`: validation logs outside the reporting
  list, one per city in turn, ordered within a city by a fixed hash.
- `av2_train_stage1.txt` will be the first 150 logs of a fixed ranking of all 700 training logs. The
  ranking needs each log's poses (driven distance and speed), so it is generated after the training
  metadata has been downloaded, once, and then committed here. Larger training sets take more logs from
  the same ranking, so the 150 logs stay a subset.
