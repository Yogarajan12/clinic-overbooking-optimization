# Data

No data is committed to this repository. Everything under `data/` is produced
on the fly and ignored by git.

## Raw

The study uses the public Kaggle *Medical Appointment No Shows* dataset
(`joniarroba/noshowappointments`): 110,527 appointments from Vitoria, Espirito
Santo, Brazil, recorded between April and June 2016. Stage 1 downloads it
through `kagglehub`, which needs Kaggle credentials in `~/.kaggle/kaggle.json`
or in the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

The dataset carries its own licence terms on Kaggle. Review them before
redistributing any derivative of it.

## Processed

Stage 1 writes the cleaned cohort and the chronological splits here:

| File | Rows | Contents |
|---|---|---|
| `appointments_cleaned.csv` | 71,959 | cleaned cohort, 22 columns |
| `X_train.csv` / `y_train.csv` | 43,175 | earliest period |
| `X_val.csv` / `y_val.csv` | 14,392 | middle period |
| `X_test.csv` / `y_test.csv` | 14,392 | latest period, used for simulation |
| `train_full.csv` / `val_full.csv` / `test_full.csv` | — | features plus demographics, needed for the fairness audit |

The 38,568 rows dropped during cleaning are appointments whose scheduled date
falls after the appointment date, which is a data error rather than a
same-day booking.

One property of this split matters for every downstream number: the test
period has a 43% no-show rate against 28.5% across the pooled cohort. The
chronological split is the right choice methodologically, but it means the
simulation runs on an unusually no-show-heavy stretch. See
`docs/limitations.md`.
