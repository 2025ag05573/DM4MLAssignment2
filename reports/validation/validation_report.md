# Data Validation Report

- Generated: 2026-11-08T06:56:54+00:00
- Source: `local:ml-1m.zip`
- Quality gate: quarantine rate per table must be <= 1.00%
- Checks run: 41 (40 pass, 1 warn, 0 fail)

## Summary by table

| table | rows_in | rows_valid | rows_quarantined | error_rate | rows_with_warnings | gate |
|---|---|---|---|---|---|---|
| users | 6040 | 6040 | 0 | 0.0000% | 81 | PASS |
| movies | 3883 | 3883 | 0 | 0.0000% | 0 | PASS |
| ratings | 1000209 | 1000209 | 0 | 0.0000% | 0 | PASS |

## Check results

Severity **error** = row quarantined to `data/errors/`; **warning** = row kept, logged in `data/errors/warnings.parquet`, and repaired in the preparation stage. Every quarantined row is also listed (table, source line, ids, failed checks) in `reports/validation/quarantined_records.csv`.

| table | check | column | severity | failed_rows | status | description | examples |
|---|---|---|---|---|---|---|---|
| users | bad_dtype | user_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| users | missing_key | user_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| users | missing_attribute | gender | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | bad_dtype | age | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| users | missing_attribute | age | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | bad_dtype | occupation | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| users | missing_attribute | occupation | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | missing_attribute | zip_code | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | id_positive | user_id | error | 0 | PASS | Identifier must be a positive integer |  |
| users | gender_valid | gender | warning | 0 | PASS | Gender must be M or F (else set to 'unknown') |  |
| users | age_code_valid | age | warning | 0 | PASS | Age must be one of the documented age-bucket codes |  |
| users | occupation_valid | occupation | warning | 0 | PASS | Occupation code must be within 0-20 |  |
| users | zip_format | zip_code | warning | 81 | WARN | ZIP code is not 5 digits (ZIP+4 truncated, others set unknown) | 98107-2117, 37919-4204, 55337-4056 |
| users | duplicate_primary_key | user_id | error | 0 | PASS | Primary key appears more than once (later copies quarantined) |  |
| users | row_volume | * | warning | 0 | PASS | Table row count outside the expected range | 6040 rows (expected 5000-10000) |
| movies | bad_dtype | movie_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| movies | missing_key | movie_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| movies | missing_attribute | title | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| movies | missing_attribute | genres | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| movies | id_positive | movie_id | error | 0 | PASS | Identifier must be a positive integer |  |
| movies | title_has_year | title | warning | 0 | PASS | Title should end with the release year '(YYYY)' |  |
| movies | duplicate_primary_key | movie_id | error | 0 | PASS | Primary key appears more than once (later copies quarantined) |  |
| movies | genre_vocabulary | genres | warning | 0 | PASS | Genre token outside the documented 18-genre vocabulary |  |
| movies | row_volume | * | warning | 0 | PASS | Table row count outside the expected range | 3883 rows (expected 3000-5000) |
| ratings | bad_dtype | user_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | user_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | movie_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | movie_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | rating | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | rating | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | timestamp | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | timestamp | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | id_positive | user_id | error | 0 | PASS | Identifier must be a positive integer |  |
| ratings | id_positive | movie_id | error | 0 | PASS | Identifier must be a positive integer |  |
| ratings | rating_in_scale | rating | error | 0 | PASS | Rating must be one of the allowed values 1-5 |  |
| ratings | timestamp_in_range | timestamp | error | 0 | PASS | Timestamp must be after MovieLens launch and not in the future |  |
| ratings | duplicate_interaction | user_id+movie_id | warning | 0 | PASS | Same (user, movie) rated more than once (latest kept) |  |
| ratings | orphan_user | user_id | error | 0 | PASS | Rating references a user_id that is not in users |  |
| ratings | orphan_movie | movie_id | error | 0 | PASS | Rating references a movie_id that is not in movies |  |
| ratings | rated_before_release | timestamp | warning | 0 | PASS | Rating timestamp precedes the movie's release year |  |
| ratings | row_volume | * | warning | 0 | PASS | Table row count outside the expected range | 1000209 rows (expected 900000-1100000) |
