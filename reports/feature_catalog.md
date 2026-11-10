# Feature Catalog

Registry version `c57ff8074c25` - generated from `src/features/definitions.py`. All features are computed point-in-time: only interactions strictly before the snapshot's `as_of` are used.

## Snapshots

| snapshot | as_of (UTC) | users | items | interactions |
|---|---|---|---|---|
| tune | 2000-12-02T14:52:18 | 6040 | 3883 | 800164 |
| final | 2000-12-29T23:43:34 | 6040 | 3883 | 900188 |
| serving | 2003-02-28T17:49:51 | 6040 | 3883 | 1000209 |

## user_features

| feature | entity | type | description | source | transformation | used_by | stats (final snapshot) |
|---|---|---|---|---|---|---|---|
| user_rating_count | user_id | int | Number of ratings the user has made | clean/ratings | count(ratings) where ts < as_of | analysis | min 0 / mean 149 / max 2.03e+03; null 0.0% |
| user_avg_rating | user_id | float | Mean rating given by the user | clean/ratings | mean(rating) | ui_rating_dev_user | min 1.02 / mean 3.71 / max 5; null 0.0% |
| user_rating_std | user_id | float | Spread of the user's ratings (0 = rates everything alike) | clean/ratings | std(rating, ddof=0) | analysis | min 0 / mean 0.996 / max 1.85; null 0.0% |
| user_positive_share | user_id | float | Share of the user's ratings that are >= 4 | clean/ratings | n(rating>=4) / n(ratings) | analysis | min 0 / mean 0.622 / max 1; null 0.0% |
| user_days_since_last_interaction | user_id | float | Recency: days between the user's latest rating and as_of | clean/ratings | (as_of - max(ts)) / 86400 | analysis | min 0.000544 / mean 90.6 / max 248; null 0.5% |
| user_tenure_days | user_id | float | Days since the user's first rating | clean/ratings | (as_of - min(ts)) / 86400 | analysis | min 0.0216 / mean 103 / max 248; null 0.5% |
| user_activity_rate | user_id | float | Ratings per active day | clean/ratings | count / max(active_span_days, 1) | analysis | min 0 / mean 84.4 / max 1.16e+03; null 0.0% |
| user_preferred_genre | user_id | category | Genre with the largest share of the user's liked items | clean/ratings, clean/movies | argmax(user_genre_affinity_*) | inference (explanations) | 19 values; top 'Drama' (42%); null 0.0% |
| user_genre_affinity_<genre> | user_id | float | Share of the user's liked (>=4) items in each genre; rows sum to 1 | clean/ratings, clean/movies | sum(one_hot(genre)) over liked items / total; falls back to all ratings if none liked | hybrid | 18 columns; mean of column means 0.0553 |
| user_mainstreamness | user_id | float | Mean popularity percentile of the items the user rated (1 = only blockbusters) | clean/ratings | mean(item_popularity_pct of rated items) | analysis | min 0.447 / mean 0.844 / max 0.987; null 0.5% |
| user_gender | user_id | category | Self-reported gender (M/F/unknown) | clean/users | standardised in preparation | analysis | 2 values; top 'M' (72%); null 0.0% |
| user_age_group | user_id | category | Age bucket label (e.g. 25-34) | clean/users | code book mapping of age code | analysis | 7 values; top '25-34' (35%); null 0.0% |
| user_occupation | user_id | category | Occupation label (21 classes) | clean/users | code book mapping of occupation code | analysis | 21 values; top 'college/grad student' (13%); null 0.0% |
| user_zip_region | user_id | category | First digit of the 5-digit US ZIP code | clean/users | zip5[0]; 'unknown' when ZIP invalid | analysis | 11 values; top '9' (24%); null 0.0% |

## item_features

| feature | entity | type | description | source | transformation | used_by | stats (final snapshot) |
|---|---|---|---|---|---|---|---|
| item_rating_count | movie_id | int | Popularity: number of ratings received | clean/ratings | count(ratings) where ts < as_of | popularity, hybrid | min 0 / mean 232 / max 3.28e+03; null 0.0% |
| item_avg_rating | movie_id | float | Mean rating received | clean/ratings | mean(rating) | analysis | min 1 / mean 3.24 / max 5; null 5.3% |
| item_bayesian_avg | movie_id | float | Damped mean rating, robust for items with few ratings | clean/ratings | (sum(rating) + m * global_mean) / (count + m), m = bayesian_prior_count | popularity | min 1.78 / mean 3.41 / max 4.55; null 0.0% |
| item_popularity_rank | movie_id | int | Rank by rating count (1 = most rated) | clean/ratings | rank(count, descending, method=min) | analysis | min 1 / mean 1.93e+03 / max 3.68e+03; null 0.0% |
| item_popularity_pct | movie_id | float | Popularity percentile in (0, 1] | clean/ratings | rank(count, pct=True) | user_mainstreamness | min 0 / mean 0.474 / max 1; null 0.0% |
| item_recent_count | movie_id | int | Trend: ratings received in the trend window before as_of | clean/ratings | count(ratings) where as_of - trend_window_days <= ts < as_of | analysis | min 0 / mean 29.8 / max 436; null 0.0% |
| item_days_since_last_interaction | movie_id | float | Days since the item was last rated | clean/ratings | (as_of - max(ts)) / 86400 | analysis | min 0.000544 / mean 10.8 / max 241; null 5.3% |
| item_release_year | movie_id | int | Release year parsed from the title | clean/movies | regex '\((\d{4})\)$' on raw title | analysis | min 1.92e+03 / mean 1.99e+03 / max 2e+03; null 0.0% |
| item_age_years | movie_id | float | Years between release and as_of | clean/movies | year(as_of) - release_year | analysis | min 0 / mean 13.9 / max 81; null 0.0% |
| item_n_genres | movie_id | int | Number of genres the movie belongs to | clean/movies | len(genres) | analysis | min 1 / mean 1.65 / max 6; null 0.0% |
| item_genre_<genre> | movie_id | int | One-hot genre membership (18 genres) | clean/movies | one_hot(split(genres, '\|')) | hybrid, evaluation | 18 columns; mean of column means 0.0917 |

## interaction_features

| feature | entity | type | description | source | transformation | used_by | stats (final snapshot) |
|---|---|---|---|---|---|---|---|
| ui_rating | user_id,movie_id | int | Observed explicit rating (1-5) | clean/ratings | rating | explicit_mf, implicit_als (via interaction matrix) | min 1 / mean 3.59 / max 5; null 0.0% |
| ui_rating_dev_user | user_id,movie_id | float | Rating minus the user's mean (taste signal without user bias) | clean/ratings | rating - user_avg_rating | analysis | min -3.89 / mean -4.07e-09 / max 3.7; null 0.0% |
| ui_rating_dev_item | user_id,movie_id | float | Rating minus the item's mean (did the user like it more than others?) | clean/ratings | rating - item_avg_rating | analysis | min -3.57 / mean -1.36e-09 / max 3.52; null 0.0% |
| ui_days_before_as_of | user_id,movie_id | float | Age of the interaction in days | clean/ratings | (as_of - ts) / 86400 | analysis | min 0.000544 / mean 102 / max 248; null 0.0% |
| ui_recency_weight | user_id,movie_id | float | Exponential time-decay weight of the interaction | clean/ratings | 0.5 ** (ui_days_before_as_of / recency_half_life_days) | analysis | min 0.00325 / mean 0.236 / max 1; null 0.0% |
| ui_is_positive | user_id,movie_id | int | 1 if rating >= relevance threshold | clean/ratings | rating >= relevance_threshold | analysis | min 0 / mean 0.579 / max 1; null 0.0% |
| ui_genre_match | user_id,movie_id | float | On-demand: cosine similarity of user genre affinity and item genre vector | user_features, item_features | cos(user_genre_affinity_*, item_genre_*) - computed at request time | hybrid | computed on request |
