import argparse
import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


load_dotenv()


class YelpNeo4jSubsetLoader:
    def __init__(self, uri, user, password, subset_name: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.subset_name = subset_name

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            queries = [
                "CREATE CONSTRAINT subset_user_uid IF NOT EXISTS FOR (u:SubsetUser) REQUIRE u.uid IS UNIQUE",
                "CREATE CONSTRAINT subset_business_uid IF NOT EXISTS FOR (b:SubsetBusiness) REQUIRE b.uid IS UNIQUE",
                "CREATE CONSTRAINT subset_review_uid IF NOT EXISTS FOR (r:SubsetReview) REQUIRE r.uid IS UNIQUE",
                "CREATE CONSTRAINT subset_category_uid IF NOT EXISTS FOR (c:SubsetCategory) REQUIRE c.uid IS UNIQUE",
            ]
            for q in queries:
                session.run(q)

    def delete_subset_namespace(self):
        # Delete in very small chunks to stay under dbms.memory.transaction.total.max.
        with self.driver.session() as session:
            self._delete_label_in_batches(session, "SubsetReview", batch_size=1500)
            self._delete_label_in_batches(session, "SubsetUser", batch_size=1000)
            self._delete_label_in_batches(session, "SubsetBusiness", batch_size=1000)
            self._delete_label_in_batches(session, "SubsetCategory", batch_size=1000)

    def _delete_label_in_batches(self, session, label: str, batch_size: int):
        while True:
            try:
                result = session.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.subset_id = $subset_id
                    WITH n LIMIT $batch_size
                    DETACH DELETE n
                    RETURN count(n) AS deleted_count
                    """,
                    subset_id=self.subset_name,
                    batch_size=batch_size,
                )
                deleted = result.single()["deleted_count"]
            except Neo4jError as exc:
                if "MemoryPoolOutOfMemoryError" in str(exc):
                    # Fallback to much smaller chunks for extremely dense subgraphs.
                    if batch_size <= 100:
                        raise
                    batch_size = max(100, batch_size // 5)
                    continue
                raise

            if deleted == 0:
                break

    def load_businesses(self, path, limit=None, batch_size=1000):
        query = """
        UNWIND $batch AS b
        MERGE (bus:SubsetBusiness {uid: b.uid})
        SET bus.subset_id = b.subset_id,
            bus.id = b.business_id,
            bus.name = b.name,
            bus.city = b.city,
            bus.state = b.state,
            bus.stars = b.stars,
            bus.review_count = b.review_count

        WITH b, bus
        WHERE b.categories IS NOT NULL
        UNWIND b.categories AS cat_name
        WITH b, bus, trim(cat_name) AS c
        WHERE c <> ''
        MERGE (cat:SubsetCategory {uid: b.subset_id + '::' + c})
        SET cat.subset_id = b.subset_id,
            cat.name = c
        MERGE (bus)-[:SUBSET_IN_CATEGORY]->(cat)
        """
        return self._process_jsonl(path, query, self._transform_business, limit, batch_size)

    def load_users(self, path, limit=None, batch_size=1000):
        query = """
        UNWIND $batch AS u
        MERGE (user:SubsetUser {uid: u.uid})
        SET user.subset_id = u.subset_id,
            user.id = u.user_id,
            user.name = u.name,
            user.review_count = u.review_count,
            user.yelping_since = u.yelping_since,
            user.average_stars = u.average_stars

        WITH user, u
        WHERE u.friends IS NOT NULL
        UNWIND u.friends AS friend_id
        WITH user, u, trim(friend_id) AS fid
        WHERE fid <> ''
        MERGE (friend:SubsetUser {uid: u.subset_id + '::' + fid})
        ON CREATE SET friend.subset_id = u.subset_id,
                      friend.id = fid
        MERGE (user)-[:SUBSET_FRIENDS_WITH]->(friend)
        """
        return self._process_jsonl(path, query, self._transform_user, limit, batch_size)

    def load_reviews(self, path, limit=None, batch_size=1000):
        query = """
        UNWIND $batch AS r
        MERGE (rev:SubsetReview {uid: r.uid})
        SET rev.subset_id = r.subset_id,
            rev.id = r.review_id,
            rev.stars = r.stars,
            rev.date = r.date,
            rev.text = substring(r.text, 0, 500),
            rev.useful = r.useful

        MERGE (u:SubsetUser {uid: r.subset_id + '::' + r.user_id})
        ON CREATE SET u.subset_id = r.subset_id,
                      u.id = r.user_id

        MERGE (b:SubsetBusiness {uid: r.subset_id + '::' + r.business_id})
        ON CREATE SET b.subset_id = r.subset_id,
                      b.id = r.business_id

        MERGE (u)-[:SUBSET_WROTE]->(rev)
        MERGE (rev)-[:SUBSET_REVIEWS]->(b)
        """
        return self._process_jsonl(path, query, self._transform_review, limit, batch_size)

    def _transform_business(self, doc):
        categories = doc.get("categories")
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",") if c.strip()]

        return {
            "subset_id": self.subset_name,
            "uid": f"{self.subset_name}::{doc['business_id']}",
            "business_id": doc.get("business_id"),
            "name": doc.get("name"),
            "city": doc.get("city"),
            "state": doc.get("state"),
            "stars": doc.get("stars"),
            "review_count": doc.get("review_count"),
            "categories": categories or [],
        }

    def _transform_user(self, doc):
        friends_raw = doc.get("friends")
        friends = []
        if isinstance(friends_raw, str):
            friends = [f.strip() for f in friends_raw.split(",") if f.strip() and f.strip().lower() != "none"]
        elif isinstance(friends_raw, list):
            friends = [str(f).strip() for f in friends_raw if str(f).strip()]

        return {
            "subset_id": self.subset_name,
            "uid": f"{self.subset_name}::{doc['user_id']}",
            "user_id": doc.get("user_id"),
            "name": doc.get("name"),
            "review_count": doc.get("review_count"),
            "yelping_since": doc.get("yelping_since"),
            "average_stars": doc.get("average_stars"),
            "friends": friends,
        }

    def _transform_review(self, doc):
        return {
            "subset_id": self.subset_name,
            "uid": f"{self.subset_name}::{doc['review_id']}",
            "review_id": doc.get("review_id"),
            "user_id": doc.get("user_id"),
            "business_id": doc.get("business_id"),
            "stars": doc.get("stars"),
            "date": doc.get("date"),
            "text": doc.get("text", ""),
            "useful": doc.get("useful", 0),
        }

    def _process_jsonl(self, filepath, query, transform_fn, limit=None, batch_size=1000):
        if not os.path.exists(filepath):
            print(f"{filepath} not found. Skipping.")
            return 0

        count = 0
        with self.driver.session() as session, open(filepath, "r", encoding="utf-8") as f:
            batch = []
            for line in f:
                if limit and count >= limit:
                    break
                doc = json.loads(line)
                batch.append(transform_fn(doc))
                count += 1

                if len(batch) >= batch_size:
                    session.run(query, batch=batch)
                    batch = []

            if batch:
                session.run(query, batch=batch)

        return count


def main():
    parser = argparse.ArgumentParser(description="Load Yelp subset into Neo4j namespace without touching full graph")
    parser.add_argument("--data-dir", default="../data/subset_shared")
    parser.add_argument("--subset-name", default="shared_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--wipe-subset",
        action="store_true",
        help="Delete existing nodes for this subset namespace before reloading",
    )
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URL")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        print("Error: Missing NEO4J_URL, NEO4J_USER, or NEO4J_PASSWORD in .env")
        return

    loader = YelpNeo4jSubsetLoader(uri, user, password, args.subset_name)
    try:
        loader.create_constraints()

        if args.wipe_subset:
            print(f"Deleting existing namespace for subset '{args.subset_name}'...")
            loader.delete_subset_namespace()

        base = args.data_dir
        counts = {}

        counts["businesses"] = loader.load_businesses(
            os.path.join(base, "yelp_academic_dataset_business.json"),
            limit=args.limit,
        )
        print(f"Loaded subset businesses: {counts['businesses']}")

        counts["users"] = loader.load_users(
            os.path.join(base, "yelp_academic_dataset_user.json"),
            limit=args.limit,
        )
        print(f"Loaded subset users: {counts['users']}")

        counts["reviews"] = loader.load_reviews(
            os.path.join(base, "yelp_academic_dataset_review.json"),
            limit=args.limit,
        )
        print(f"Loaded subset reviews: {counts['reviews']}")

        print("Subset Neo4j load complete.")
        print(json.dumps(counts, indent=2))
    finally:
        loader.close()


if __name__ == "__main__":
    main()
