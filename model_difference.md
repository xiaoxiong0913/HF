# Model Difference

- The web application uses the final parsimonious CatBoost deployment model selected for clinical translation.
- The deployed CatBoost model consumes 10 harmonized first-day ICU variables. Feature set: compact10_rich.
- GWTG-HF uses 7 fixed variables with a fixed point-based rule originally designed for hospitalized heart failure patients.
- ADHERE uses 3 fixed thresholds and is intended as a simple bedside tree rather than a rich ICU risk engine.
- GWTG-HF and ADHERE are implemented here as database-adapted bedside comparators rather than exact bedside reproductions.
