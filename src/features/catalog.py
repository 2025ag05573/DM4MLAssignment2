"""Generate the feature catalog (markdown + CSV) from the registry + snapshot stats."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.config import get_paths
from src.common.io import df_to_markdown, read_json
from src.features.build import definitions_version, expected_columns
from src.features.definitions import FEATURES
from src.features.store import FeatureStore


def _stats(series: pd.Series) -> str:
    s = series.dropna()
    nulls = 1 - len(s) / max(len(series), 1)
    if pd.api.types.is_numeric_dtype(s):
        txt = f"min {s.min():.3g} / mean {s.mean():.3g} / max {s.max():.3g}"
    else:
        vc = s.value_counts()
        txt = f"{s.nunique()} values; top '{vc.index[0]}' ({vc.iloc[0] / len(s):.0%})" if len(vc) else "-"
    return f"{txt}; null {nulls:.1%}"


def write_catalog(cfg: dict[str, Any], snapshot: str = "final") -> pd.DataFrame:
    paths = get_paths(cfg)
    store = FeatureStore(cfg)
    frames = {"user_features": store.get_user_features(snapshot=snapshot).reset_index(),
              "item_features": store.get_item_features(snapshot=snapshot).reset_index(),
              "interaction_features": store.get_interaction_features(snapshot=snapshot)}
    genres = cfg["validation"]["genres"]
    rows = []
    for f in FEATURES:
        df = frames[f.group]
        if f.n_columns > 1:
            cols = [c for c in expected_columns(f.group, genres) if c.startswith(f.name.split("<")[0])]
            stat = f"{len(cols)} columns; mean of column means {df[cols].mean().mean():.3g}"
        elif f.name in df.columns:
            stat = _stats(df[f.name])
        else:
            stat = "computed on request"
        rows.append({"feature": f.name, "group": f.group, "entity": f.entity, "type": f.dtype,
                     "description": f.description, "source": f.source,
                     "transformation": f.transformation, "used_by": f.to_dict()["used_by"],
                     f"stats ({snapshot} snapshot)": stat})
    cat = pd.DataFrame(rows)
    cat.to_csv(paths.reports / "feature_catalog.csv", index=False)
    reg = read_json(paths.features / "registry.json")
    snaps = pd.DataFrame([{"snapshot": k, "as_of (UTC)": v["as_of_iso"][:19],
                           "users": v["groups"]["user_features"]["rows"],
                           "items": v["groups"]["item_features"]["rows"],
                           "interactions": v["groups"]["interaction_features"]["rows"]}
                          for k, v in reg["snapshots"].items()])
    md = ["# Feature Catalog", "",
          f"Registry version `{definitions_version()}` - generated from `src/features/definitions.py`. "
          "All features are computed point-in-time: only interactions strictly before the snapshot's "
          "`as_of` are used.", "",
          "## Snapshots", "", df_to_markdown(snaps), ""]
    for group in ("user_features", "item_features", "interaction_features"):
        sub = cat[cat["group"] == group].drop(columns=["group"])
        md += [f"## {group}", "", df_to_markdown(sub), ""]
    (paths.reports / "feature_catalog.md").write_text("\n".join(md), encoding="utf-8")
    return cat
