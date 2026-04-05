import argparse
import json
import os
from datetime import datetime, timezone

import dateutil.parser
from pymongo import MongoClient


def transform_business(doc):
    doc = dict(doc)
    doc["_id"] = doc.pop("business_id")
    if isinstance(doc.get("categories"), str):
        doc["categories"] = [c.strip() for c in doc["categories"].split(",") if c.strip()]
    return doc


def transform_user(doc):
    doc = dict(doc)
    doc["_id"] = doc.pop("user_id")

    friends_val = doc.get("friends")
    if isinstance(friends_val, str):
        doc["friends"] = [f.strip() for f in friends_val.split(",") if f.strip() and f.strip().lower() != "none"]

    elite_val = doc.get("elite")
    if isinstance(elite_val, str):
        doc["elite"] = [e.strip() for e in elite_val.split(",") if e.strip() and e.strip().lower() != "none"]

    if doc.get("yelping_since"):
        try:
            doc["yelping_since"] = dateutil.parser.parse(doc["yelping_since"])
        except Exception:
            pass
    return doc


def transform_review(doc):
    doc = dict(doc)
    doc["_id"] = doc.pop("review_id")
    if doc.get("date"):
        try:
            doc["date"] = dateutil.parser.parse(doc["date"])
        except Exception:
            pass
    return doc


def transform_checkin(doc):
    doc = dict(doc)
    doc["_id"] = doc.pop("business_id")
    date_str = doc.get("date", "")
    if date_str:
        doc["dates"] = [dateutil.parser.parse(ds.strip()) for ds in date_str.split(",") if ds.strip()]
    else:
        doc["dates"] = []
    doc.pop("date", None)
    return doc


def transform_tip(doc):
    doc = dict(doc)
    if doc.get("date"):
        try:
            doc["date"] = dateutil.parser.parse(doc["date"])
        except Exception:
            pass
    return doc


def process_file(filepath, collection, transform_fn, batch_size=50000, limit=None):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping.")
        return 0

    inserted = 0
    batch = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if limit and inserted >= limit:
                break

            doc = transform_fn(json.loads(line))
            batch.append(doc)
            inserted += 1

            if len(batch) >= batch_size:
                collection.insert_many(batch, ordered=False)
                batch = []

    if batch:
        collection.insert_many(batch, ordered=False)

    return inserted


def create_indexes(db):
    db.businesses.create_index([("city", 1), ("review_count", -1)])
    db.businesses.create_index([("categories", 1), ("review_count", -1)])
    db.reviews.create_index([("business_id", 1), ("date", 1)])
    db.reviews.create_index([("user_id", 1), ("date", 1)])
    db.users.create_index([("yelping_since", 1)])


def main():
    parser = argparse.ArgumentParser(description="Load Yelp subset into separate MongoDB database")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="yelp_db_subset_shared")
    parser.add_argument("--data-dir", default="../data/subset_shared")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--drop-target",
        action="store_true",
        help="Drop target subset DB collections before load",
    )
    parser.add_argument("--subset-name", default="shared_v1")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]

    collections = ["businesses", "users", "reviews", "checkins", "tips"]
    if args.drop_target:
        print(f"Dropping target collections in DB '{args.db_name}'...")
        for coll in collections:
            db[coll].drop()

    files = [
        ("yelp_academic_dataset_business.json", "businesses", transform_business),
        ("yelp_academic_dataset_user.json", "users", transform_user),
        ("yelp_academic_dataset_review.json", "reviews", transform_review),
        ("yelp_academic_dataset_checkin.json", "checkins", transform_checkin),
        ("yelp_academic_dataset_tip.json", "tips", transform_tip),
    ]

    counts = {}
    for filename, coll_name, transform in files:
        path = os.path.join(args.data_dir, filename)
        print(f"Loading {filename} -> {args.db_name}.{coll_name}")
        count = process_file(path, db[coll_name], transform, limit=args.limit)
        counts[coll_name] = count
        print(f"  inserted: {count}")

    create_indexes(db)

    db.etl_runs.insert_one(
        {
            "run_type": "subset_load",
            "subset_name": args.subset_name,
            "loaded_at": datetime.now(timezone.utc),
            "source_data_dir": os.path.abspath(args.data_dir),
            "counts": counts,
        }
    )

    print("Subset MongoDB load complete.")
    print(f"Target DB: {args.db_name}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
