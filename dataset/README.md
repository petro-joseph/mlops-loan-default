# German Credit dataset

`german_credit.csv`: 1,000 past loans, 20 applicant attributes and the outcome
(`class`: `good` = repaid, `bad` = defaulted; 700 / 300).

- Source: Statlog (German Credit Data), Prof. Hans Hofmann, Universität Hamburg, 1994.
  UCI Machine Learning Repository, https://doi.org/10.24432/C5NC77, via OpenML dataset 31.
- License: Creative Commons Attribution 4.0 (CC BY 4.0).
- SHA-256: `38b6dbf6fb4b0311a3ffc005730f42623128591fb36473ab3c22d270c0467632`

Kept in the repository so training is reproducible offline and in CI, and so a
change to the data shows up in review like a change to the code. Every MLflow
run records this file's SHA-256 as `data.sha256`.
