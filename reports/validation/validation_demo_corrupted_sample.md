# Data Validation Report - corrupted-sample demo

- Generated: 2026-11-08T07:06:44+00:00
- Source: `src/validation/samples.py dirty_sample() - demo only, not pipeline data`
- Quality gate: quarantine rate per table must be <= 1.00%
- Checks run: 41 (22 pass, 10 warn, 9 fail)

## Summary by table

| table | rows_in | rows_valid | rows_quarantined | error_rate | rows_with_warnings | gate |
|---|---|---|---|---|---|---|
| users | 7 | 4 | 3 | 42.8571% | 5 | FAIL |
| movies | 4 | 4 | 0 | 0.0000% | 2 | PASS |
| ratings | 12 | 5 | 7 | 58.3333% | 2 | FAIL |

## Check results

Severity **error** = row quarantined to `data/errors/`; **warning** = row kept, logged in `data/errors/warnings.parquet`, and repaired in the preparation stage. Every quarantined row is also listed (table, source line, ids, failed checks) in `reports/validation/quarantined_records.csv`.

| table | check | column | severity | failed_rows | status | description | examples |
|---|---|---|---|---|---|---|---|
| users | bad_dtype | user_id | error | 1 | FAIL | Value cannot be parsed to the contract type | five |
| users | missing_key | user_id | error | 1 | FAIL | Identifier / rating / timestamp is null |  |
| users | missing_attribute | gender | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | bad_dtype | age | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| users | missing_attribute | age | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | bad_dtype | occupation | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| users | missing_attribute | occupation | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | missing_attribute | zip_code | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| users | id_positive | user_id | error | 0 | PASS | Identifier must be a positive integer |  |
| users | gender_valid | gender | warning | 1 | WARN | Gender must be M or F (else set to 'unknown') | X |
| users | age_code_valid | age | warning | 1 | WARN | Age must be one of the documented age-bucket codes | 99 |
| users | occupation_valid | occupation | warning | 1 | WARN | Occupation code must be within 0-20 | 33 |
| users | zip_format | zip_code | warning | 2 | WARN | ZIP code is not 5 digits (ZIP+4 truncated, others set unknown) | 94110-1234, ABCDE |
| users | duplicate_primary_key | user_id | error | 1 | FAIL | Primary key appears more than once (later copies quarantined) | 2 |
| users | row_volume | * | warning | 0 | WARN | Table row count outside the expected range | 7 rows (expected 5000-10000) |
| movies | bad_dtype | movie_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| movies | missing_key | movie_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| movies | missing_attribute | title | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| movies | missing_attribute | genres | warning | 0 | PASS | Non-key attribute is null (imputed in preparation) |  |
| movies | id_positive | movie_id | error | 0 | PASS | Identifier must be a positive integer |  |
| movies | title_has_year | title | warning | 1 | WARN | Title should end with the release year '(YYYY)' | No Year Movie |
| movies | duplicate_primary_key | movie_id | error | 0 | PASS | Primary key appears more than once (later copies quarantined) |  |
| movies | genre_vocabulary | genres | warning | 1 | WARN | Genre token outside the documented 18-genre vocabulary | Drama\|Spaghetti |
| movies | row_volume | * | warning | 0 | WARN | Table row count outside the expected range | 4 rows (expected 3000-5000) |
| ratings | bad_dtype | user_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | user_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | movie_id | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | movie_id | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | rating | error | 1 | FAIL | Value cannot be parsed to the contract type | 4.5 |
| ratings | missing_key | rating | error | 1 | FAIL | Identifier / rating / timestamp is null |  |
| ratings | bad_dtype | timestamp | error | 0 | PASS | Value cannot be parsed to the contract type |  |
| ratings | missing_key | timestamp | error | 0 | PASS | Identifier / rating / timestamp is null |  |
| ratings | id_positive | user_id | error | 0 | PASS | Identifier must be a positive integer |  |
| ratings | id_positive | movie_id | error | 0 | PASS | Identifier must be a positive integer |  |
| ratings | rating_in_scale | rating | error | 1 | FAIL | Rating must be one of the allowed values 1-5 | 7 |
| ratings | timestamp_in_range | timestamp | error | 2 | FAIL | Timestamp must be after MovieLens launch and not in the future | 4102444800, 800000000 |
| ratings | duplicate_interaction | user_id+movie_id | warning | 2 | WARN | Same (user, movie) rated more than once (latest kept) |  |
| ratings | orphan_user | user_id | error | 1 | FAIL | Rating references a user_id that is not in users | 999 |
| ratings | orphan_movie | movie_id | error | 1 | FAIL | Rating references a movie_id that is not in movies | 777 |
| ratings | rated_before_release | timestamp | warning | 0 | PASS | Rating timestamp precedes the movie's release year |  |
| ratings | row_volume | * | warning | 0 | WARN | Table row count outside the expected range | 12 rows (expected 900000-1100000) |

## Quarantined rows (what would be written to data/errors/)

### users

| _source_line | user_id | gender | age | occupation | zip_code | _error_checks |
|---|---|---|---|---|---|---|
| 4 | 2 | M | 35 | 12 | 94110 | duplicate_primary_key:user_id |
| 6 | five | F | 25 | 4 | 10001 | bad_dtype:user_id |
| 7 | <NA> | F | 25 | 4 | 10001 | missing_key:user_id |

### ratings

| _source_line | user_id | movie_id | rating | timestamp | _error_checks |
|---|---|---|---|---|---|
| 5 | 1 | 30 | 7 | 978300900 | rating_in_scale:rating |
| 6 | 2 | 20 | 4 | 4102444800 | timestamp_in_range:timestamp |
| 7 | 3 | 20 | 4 | 800000000 | timestamp_in_range:timestamp |
| 8 | 999 | 10 | 4 | 978300900 | orphan_user:user_id |
| 9 | 1 | 777 | 4 | 978300900 | orphan_movie:movie_id |
| 10 | 2 | 30 | <NA> | 978300900 | missing_key:rating |
| 11 | 3 | 10 | 4.5 | 978300900 | bad_dtype:rating |
