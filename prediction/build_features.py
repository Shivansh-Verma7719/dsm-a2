import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser as dt_parser
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient


load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "report" / "p2" / "data" / "predictive_modelling"
FIG_ROOT = ROOT / "report" / "p2" / "figures" / "predictive_modelling"
CACHE_DIR = DATA_ROOT / "cache"


@dataclass
class BuildConfig:
    mongo_uri: str
    mongo_db: str
    graph_mode: str
    subset_name: str
    force_rebuild: bool


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return dt_parser.parse(v)
        except Exception:
            return None
    return None


def text_word_count(text: str) -> int:
    if not text:
        return 0
    return len([t for t in text.split() if t])


def bool_to_int(v: Any) -> int:
    return 1 if bool(v) else 0


def get_label(label: str, graph_mode: str) -> str:
    if graph_mode == "subset":
        m = {
            "User": "SubsetUser",
            "Business": "SubsetBusiness",
            "Review": "SubsetReview",
            "Category": "SubsetCategory",
            "FRIENDS_WITH": "SUBSET_FRIENDS_WITH",
            "IN_CATEGORY": "SUBSET_IN_CATEGORY",
        }
        return m.get(label, label)
    return label


def extract_base_from_mongo(cfg: BuildConfig, out_path: Path) -> pd.DataFrame:
    client = MongoClient(cfg.mongo_uri)
    db = client[cfg.mongo_db]

    users = {}
    for u in db.users.find(
        {},
        {
            "_id": 1,
            "review_count": 1,
            "average_stars": 1,
            "yelping_since": 1,
            "elite": 1,
            "friends": 1,
        },
    ):
        elite_val = u.get("elite", [])
        if isinstance(elite_val, str):
            elite_years = [e.strip() for e in elite_val.split(",") if e.strip() and e.strip().lower() != "none"]
        elif isinstance(elite_val, list):
            elite_years = [str(e).strip() for e in elite_val if str(e).strip()]
        else:
            elite_years = []

        friends_val = u.get("friends", [])
        if isinstance(friends_val, str):
            friends = [f.strip() for f in friends_val.split(",") if f.strip() and f.strip().lower() != "none"]
        elif isinstance(friends_val, list):
            friends = [str(f).strip() for f in friends_val if str(f).strip()]
        else:
            friends = []

        users[str(u.get("_id"))] = {
            "user_review_count": float(u.get("review_count") or 0),
            "user_average_stars": float(u.get("average_stars") or 0),
            "user_yelping_since": parse_dt(u.get("yelping_since")),
            "user_elite_years_count": float(len(elite_years)),
            "user_has_elite": float(1 if elite_years else 0),
            "user_friends_count": float(len(friends)),
        }

    businesses = {}
    for b in db.businesses.find(
        {},
        {
            "_id": 1,
            "review_count": 1,
            "stars": 1,
            "city": 1,
            "state": 1,
            "categories": 1,
        },
    ):
        cats = b.get("categories", [])
        if isinstance(cats, str):
            categories = [c.strip() for c in cats.split(",") if c.strip()]
        elif isinstance(cats, list):
            categories = [str(c).strip() for c in cats if str(c).strip()]
        else:
            categories = []

        businesses[str(b.get("_id"))] = {
            "business_review_count": float(b.get("review_count") or 0),
            "business_average_stars": float(b.get("stars") or 0),
            "business_city": str(b.get("city") or "").strip(),
            "business_state": str(b.get("state") or "").strip(),
            "business_primary_category": categories[0] if categories else "",
        }

    rows: list[dict[str, Any]] = []

    # First pass collects max date for recency feature.
    max_review_dt: datetime | None = None
    for r in db.reviews.find({}, {"date": 1}):
        rd = parse_dt(r.get("date"))
        if rd is None:
            continue
        if max_review_dt is None or rd > max_review_dt:
            max_review_dt = rd

    if max_review_dt is None:
        max_review_dt = datetime.now(timezone.utc)

    for r in db.reviews.find(
        {},
        {
            "_id": 1,
            "user_id": 1,
            "business_id": 1,
            "stars": 1,
            "date": 1,
            "text": 1,
            "useful": 1,
        },
    ):
        review_id = str(r.get("_id"))
        user_id = str(r.get("user_id") or "")
        business_id = str(r.get("business_id") or "")

        if user_id not in users or business_id not in businesses:
            continue

        u = users[user_id]
        b = businesses[business_id]

        text = str(r.get("text") or "")
        rd = parse_dt(r.get("date"))
        if rd is None:
            rd = max_review_dt

        tenure_days = (rd - u["user_yelping_since"]).days if u["user_yelping_since"] else 0
        recency_days = (max_review_dt - rd).days

        rows.append(
            {
                "review_id": review_id,
                "user_id": user_id,
                "business_id": business_id,
                "target_useful": float(r.get("useful") or 0),
                # Review-level
                "review_stars": float(r.get("stars") or 0),
                "review_text_len": float(len(text)),
                "review_word_count": float(text_word_count(text)),
                "review_exclamation_count": float(text.count("!")),
                "review_recency_days": float(max(recency_days, 0)),
                "review_year": float(rd.year),
                # User-level
                "user_review_count": u["user_review_count"],
                "user_average_stars": u["user_average_stars"],
                "user_tenure_days": float(max(tenure_days, 0)),
                "user_elite_years_count": u["user_elite_years_count"],
                "user_has_elite": u["user_has_elite"],
                "user_friends_count": u["user_friends_count"],
                # Business context
                "business_review_count": b["business_review_count"],
                "business_average_stars": b["business_average_stars"],
                "business_city": b["business_city"],
                "business_state": b["business_state"],
                "business_primary_category": b["business_primary_category"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, compression="gzip")
    return df


def extract_graph_user_features(cfg: BuildConfig, out_path: Path) -> pd.DataFrame:
    neo_uri = os.getenv("NEO4J_URL")
    neo_user = os.getenv("NEO4J_USER")
    neo_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo_uri, neo_user, neo_password]):
        raise RuntimeError("Missing NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD in environment/.env")

    user_label = get_label("User", cfg.graph_mode)
    friend_rel = get_label("FRIENDS_WITH", cfg.graph_mode)

    query = f"""
MATCH (u:{user_label})
OPTIONAL MATCH (u)-[:{friend_rel}]-(:{user_label})
WITH u, count(*) AS graph_degree
RETURN u.id AS user_id,
       graph_degree,
       toInteger(coalesce(u.p2_louvain, -1)) AS community_id;
"""

    with GraphDatabase.driver(neo_uri, auth=(neo_user, neo_password)) as driver, driver.session() as session:
        rows = [r.data() for r in session.run(query)]

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["user_id", "graph_degree", "community_id"])

    n2_profile = ROOT / "report" / "p2" / "data" / "neo4j" / "n2_louvain_community_profile.csv"
    if n2_profile.exists():
        com_df = pd.read_csv(n2_profile, usecols=["community_id", "community_size", "geo_concentration_index"])
        df["community_id"] = pd.to_numeric(df["community_id"], errors="coerce").fillna(-1).astype(int)
        df = df.merge(com_df, on="community_id", how="left")
    else:
        df["community_size"] = np.nan
        df["geo_concentration_index"] = np.nan

    df = df.rename(
        columns={
            "graph_degree": "graph_user_degree",
            "community_size": "graph_user_community_size",
            "geo_concentration_index": "graph_user_community_gci",
        }
    )
    df.to_csv(out_path, index=False, compression="gzip")
    return df


def extract_graph_business_features(out_path: Path) -> pd.DataFrame:
    n3_path = ROOT / "report" / "p2" / "data" / "neo4j" / "n3_city_category_similarity.csv"
    if not n3_path.exists():
        raise RuntimeError("Missing n3_city_category_similarity.csv. Run Neo4j NQ3 first.")

    df = pd.read_csv(n3_path)
    req_cols = {"city", "category", "mean_similarity"}
    if not req_cols.issubset(set(df.columns)):
        raise RuntimeError("n3_city_category_similarity.csv does not contain expected columns.")

    out = (
        df[["city", "category", "mean_similarity"]]
        .rename(
            columns={
                "city": "business_city",
                "category": "business_primary_category",
                "mean_similarity": "graph_business_city_category_similarity",
            }
        )
        .copy()
    )

    city_fallback = (
        out.groupby("business_city", as_index=False)["graph_business_city_category_similarity"]
        .mean()
        .rename(columns={"graph_business_city_category_similarity": "graph_business_city_similarity_fallback"})
    )

    out = out.merge(city_fallback, on="business_city", how="left")
    out.to_csv(out_path, index=False, compression="gzip")
    return out


def build_final_dataset(base_df: pd.DataFrame, user_graph_df: pd.DataFrame, biz_graph_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = base_df.copy()

    if not user_graph_df.empty:
        user_graph_df["user_id"] = user_graph_df["user_id"].astype(str)
        df = df.merge(user_graph_df, on="user_id", how="left")
    else:
        df["graph_user_degree"] = np.nan
        df["graph_user_community_size"] = np.nan
        df["graph_user_community_gci"] = np.nan

    # Many city/category combinations can repeat in NQ3 output due to multiple rows; average them.
    biz_graph = (
        biz_graph_df.groupby(["business_city", "business_primary_category"], as_index=False)
        .agg(
            graph_business_city_category_similarity=("graph_business_city_category_similarity", "mean"),
            graph_business_city_similarity_fallback=("graph_business_city_similarity_fallback", "mean"),
        )
    )

    df = df.merge(biz_graph, on=["business_city", "business_primary_category"], how="left")
    df["graph_business_city_category_similarity"] = df["graph_business_city_category_similarity"].fillna(
        df["graph_business_city_similarity_fallback"]
    )

    # Mark feature groups for traceability.
    df["has_review_features"] = 1
    df["has_user_features"] = 1
    df["has_graph_features"] = bool_to_int(
        (
            df[["graph_user_degree", "graph_user_community_size", "graph_business_city_category_similarity"]]
            .notna()
            .sum(axis=1)
            >= 2
        ).all()
    )

    df.to_csv(out_path, index=False, compression="gzip")
    return df


def run(cfg: BuildConfig) -> None:
    ensure_dirs()

    base_path = CACHE_DIR / "base_review_user_business.csv.gz"
    user_graph_path = CACHE_DIR / "graph_user_features.csv.gz"
    biz_graph_path = CACHE_DIR / "graph_business_features.csv.gz"
    final_path = DATA_ROOT / "modeling_dataset.csv.gz"
    meta_path = DATA_ROOT / "feature_build_metadata.json"

    if cfg.force_rebuild:
        for p in [base_path, user_graph_path, biz_graph_path, final_path, meta_path]:
            if p.exists():
                p.unlink()

    if base_path.exists():
        base_df = pd.read_csv(base_path)
    else:
        base_df = extract_base_from_mongo(cfg, base_path)

    if user_graph_path.exists():
        user_graph_df = pd.read_csv(user_graph_path)
    else:
        user_graph_df = extract_graph_user_features(cfg, user_graph_path)

    if biz_graph_path.exists():
        biz_graph_df = pd.read_csv(biz_graph_path)
    else:
        biz_graph_df = extract_graph_business_features(biz_graph_path)

    final_df = build_final_dataset(base_df, user_graph_df, biz_graph_df, final_path)

    feature_groups = {
        "review_level": [
            "review_stars",
            "review_text_len",
            "review_word_count",
            "review_exclamation_count",
            "review_recency_days",
            "review_year",
        ],
        "user_level": [
            "user_review_count",
            "user_average_stars",
            "user_tenure_days",
            "user_elite_years_count",
            "user_has_elite",
            "user_friends_count",
        ],
        "graph_level": [
            "graph_user_degree",
            "graph_user_community_size",
            "graph_user_community_gci",
            "graph_business_city_category_similarity",
        ],
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mongo_db": cfg.mongo_db,
        "graph_mode": cfg.graph_mode,
        "subset_name": cfg.subset_name,
        "rows": int(final_df.shape[0]),
        "columns": int(final_df.shape[1]),
        "target": "target_useful",
        "feature_groups": feature_groups,
        "cache_files": {
            "base": str(base_path),
            "user_graph": str(user_graph_path),
            "business_graph": str(biz_graph_path),
            "final": str(final_path),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Feature extraction complete.")
    print(json.dumps({"rows": metadata["rows"], "columns": metadata["columns"]}, indent=2))
    print(f"Dataset: {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached predictive modelling feature dataset")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--mongo-db", default=os.getenv("MONGO_DB", "yelp_db_subset_shared"))
    parser.add_argument("--graph-mode", choices=["full", "subset"], default="subset")
    parser.add_argument("--subset-name", default="shared_v1")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BuildConfig(
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        graph_mode=args.graph_mode,
        subset_name=args.subset_name,
        force_rebuild=args.force_rebuild,
    )
    run(cfg)
