"""Feature registry - the single source of truth for every feature.

Each :class:`FeatureDefinition` documents *what* a feature means, *where* it
comes from and *how* it is computed. ``build.py`` materialises the features
from these definitions, ``catalog.py`` renders the catalog from them, and
``store.py`` serves them - so documentation can't drift from the code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    entity: str            # join key(s): user_id | movie_id | user_id,movie_id
    group: str             # feature view: user_features | item_features | interaction_features
    dtype: str
    description: str
    source: str            # upstream dataset(s)
    transformation: str    # how it is computed (point-in-time: only data before as_of)
    used_by: tuple[str, ...] = field(default_factory=tuple)
    n_columns: int = 1     # >1 for a feature family (e.g. one column per genre)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["used_by"] = ", ".join(self.used_by) if self.used_by else "analysis"
        return d


U, I, X = "user_features", "item_features", "interaction_features"
SRC_R = "clean/ratings"
SRC_U = "clean/users"
SRC_M = "clean/movies"

FEATURES: list[FeatureDefinition] = [
    # ---------------------------------------------------------------- user features
    FeatureDefinition("user_rating_count", "user_id", U, "int", "Number of ratings the user has made",
                      SRC_R, "count(ratings) where ts < as_of"),
    FeatureDefinition("user_avg_rating", "user_id", U, "float", "Mean rating given by the user",
                      SRC_R, "mean(rating)", ("ui_rating_dev_user",)),
    FeatureDefinition("user_rating_std", "user_id", U, "float", "Spread of the user's ratings (0 = rates everything alike)",
                      SRC_R, "std(rating, ddof=0)"),
    FeatureDefinition("user_positive_share", "user_id", U, "float", "Share of the user's ratings that are >= 4",
                      SRC_R, "n(rating>=4) / n(ratings)"),
    FeatureDefinition("user_days_since_last_interaction", "user_id", U, "float",
                      "Recency: days between the user's latest rating and as_of", SRC_R,
                      "(as_of - max(ts)) / 86400"),
    FeatureDefinition("user_tenure_days", "user_id", U, "float", "Days since the user's first rating",
                      SRC_R, "(as_of - min(ts)) / 86400"),
    FeatureDefinition("user_activity_rate", "user_id", U, "float", "Ratings per active day",
                      SRC_R, "count / max(active_span_days, 1)"),
    FeatureDefinition("user_preferred_genre", "user_id", U, "category",
                      "Genre with the largest share of the user's liked items", f"{SRC_R}, {SRC_M}",
                      "argmax(user_genre_affinity_*)", ("inference (explanations)",)),
    FeatureDefinition("user_genre_affinity_<genre>", "user_id", U, "float",
                      "Share of the user's liked (>=4) items in each genre; rows sum to 1",
                      f"{SRC_R}, {SRC_M}", "sum(one_hot(genre)) over liked items / total; "
                      "falls back to all ratings if none liked", ("hybrid",), n_columns=18),
    FeatureDefinition("user_mainstreamness", "user_id", U, "float",
                      "Mean popularity percentile of the items the user rated (1 = only blockbusters)",
                      f"{SRC_R}", "mean(item_popularity_pct of rated items)"),
    FeatureDefinition("user_gender", "user_id", U, "category", "Self-reported gender (M/F/unknown)",
                      SRC_U, "standardised in preparation"),
    FeatureDefinition("user_age_group", "user_id", U, "category", "Age bucket label (e.g. 25-34)",
                      SRC_U, "code book mapping of age code"),
    FeatureDefinition("user_occupation", "user_id", U, "category", "Occupation label (21 classes)",
                      SRC_U, "code book mapping of occupation code"),
    FeatureDefinition("user_zip_region", "user_id", U, "category", "First digit of the 5-digit US ZIP code",
                      SRC_U, "zip5[0]; 'unknown' when ZIP invalid"),
    # ---------------------------------------------------------------- item features
    FeatureDefinition("item_rating_count", "movie_id", I, "int", "Popularity: number of ratings received",
                      SRC_R, "count(ratings) where ts < as_of", ("popularity", "hybrid")),
    FeatureDefinition("item_avg_rating", "movie_id", I, "float", "Mean rating received", SRC_R, "mean(rating)"),
    FeatureDefinition("item_bayesian_avg", "movie_id", I, "float",
                      "Damped mean rating, robust for items with few ratings", SRC_R,
                      "(sum(rating) + m * global_mean) / (count + m), m = bayesian_prior_count",
                      ("popularity",)),
    FeatureDefinition("item_popularity_rank", "movie_id", I, "int", "Rank by rating count (1 = most rated)",
                      SRC_R, "rank(count, descending, method=min)"),
    FeatureDefinition("item_popularity_pct", "movie_id", I, "float", "Popularity percentile in (0, 1]",
                      SRC_R, "rank(count, pct=True)", ("user_mainstreamness",)),
    FeatureDefinition("item_recent_count", "movie_id", I, "int",
                      "Trend: ratings received in the trend window before as_of", SRC_R,
                      "count(ratings) where as_of - trend_window_days <= ts < as_of"),
    FeatureDefinition("item_days_since_last_interaction", "movie_id", I, "float",
                      "Days since the item was last rated", SRC_R, "(as_of - max(ts)) / 86400"),
    FeatureDefinition("item_release_year", "movie_id", I, "int", "Release year parsed from the title",
                      SRC_M, "regex '\\((\\d{4})\\)$' on raw title"),
    FeatureDefinition("item_age_years", "movie_id", I, "float", "Years between release and as_of",
                      SRC_M, "year(as_of) - release_year"),
    FeatureDefinition("item_n_genres", "movie_id", I, "int", "Number of genres the movie belongs to",
                      SRC_M, "len(genres)"),
    FeatureDefinition("item_genre_<genre>", "movie_id", I, "int", "One-hot genre membership (18 genres)",
                      SRC_M, "one_hot(split(genres, '|'))", ("hybrid", "evaluation"), n_columns=18),
    # ---------------------------------------------------------------- interaction features
    FeatureDefinition("ui_rating", "user_id,movie_id", X, "int", "Observed explicit rating (1-5)",
                      SRC_R, "rating", ("explicit_mf", "implicit_als (via interaction matrix)")),
    FeatureDefinition("ui_rating_dev_user", "user_id,movie_id", X, "float",
                      "Rating minus the user's mean (taste signal without user bias)", SRC_R,
                      "rating - user_avg_rating"),
    FeatureDefinition("ui_rating_dev_item", "user_id,movie_id", X, "float",
                      "Rating minus the item's mean (did the user like it more than others?)", SRC_R,
                      "rating - item_avg_rating"),
    FeatureDefinition("ui_days_before_as_of", "user_id,movie_id", X, "float", "Age of the interaction in days",
                      SRC_R, "(as_of - ts) / 86400"),
    FeatureDefinition("ui_recency_weight", "user_id,movie_id", X, "float",
                      "Exponential time-decay weight of the interaction", SRC_R,
                      "0.5 ** (ui_days_before_as_of / recency_half_life_days)"),
    FeatureDefinition("ui_is_positive", "user_id,movie_id", X, "int", "1 if rating >= relevance threshold",
                      SRC_R, "rating >= relevance_threshold"),
    FeatureDefinition("ui_genre_match", "user_id,movie_id", X, "float",
                      "On-demand: cosine similarity of user genre affinity and item genre vector",
                      "user_features, item_features",
                      "cos(user_genre_affinity_*, item_genre_*) - computed at request time",
                      ("hybrid",)),
]


def by_group(group: str) -> list[FeatureDefinition]:
    return [f for f in FEATURES if f.group == group]
