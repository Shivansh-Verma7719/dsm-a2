import json
import os
import argparse
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class YelpNeo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("Clearing existing database... (this may take a moment)")
        with self.driver.session() as session:
            while True:
                # Delete in safe batches to prevent out-of-memory errors on massive graphs
                result = session.run("""
                MATCH (n)
                WITH n LIMIT 50000
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """)
                deleted_count = result.single()["deleted_count"]
                if deleted_count == 0:
                    break
        print("Database cleared.")

    def create_indexes(self):
        print("Creating Neo4j Indexes and Constraints...")
        with self.driver.session() as session:
            queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Business) REQUIRE b.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
            ]
            for q in queries:
                session.run(q)
        print("Indexes created.")

    def load_businesses(self, filepath, limit=10000, batch_size=1000):
        query = """
        UNWIND $batch AS b
        MERGE (bus:Business {id: b.business_id})
        SET bus.name = b.name, 
            bus.city = b.city, 
            bus.state = b.state, 
            bus.stars = b.stars, 
            bus.review_count = b.review_count
        
        WITH b, bus
        WHERE b.categories IS NOT NULL
        UNWIND split(b.categories, ',') AS cat_raw
        WITH b, bus, trim(cat_raw) AS cat_name
        WHERE cat_name <> ''
        MERGE (c:Category {name: cat_name})
        MERGE (bus)-[:IN_CATEGORY]->(c)
        """
        self._process_file(filepath, query, self._transform_business, limit, batch_size, "Businesses")

    def load_users(self, filepath, limit=10000, batch_size=500):
        query = """
        UNWIND $batch AS u
        MERGE (user:User {id: u.user_id})
        SET user.name = u.name,
            user.review_count = u.review_count,
            user.yelping_since = u.yelping_since,
            user.average_stars = u.average_stars
            
        WITH user, u
        WHERE u.friends IS NOT NULL
        UNWIND split(u.friends, ',') AS friend_raw
        WITH user, trim(friend_raw) AS friend_id
        WHERE friend_id <> '' AND friend_id <> 'None'
        MERGE (friend:User {id: friend_id})
        MERGE (user)-[:FRIENDS_WITH]->(friend)
        """
        self._process_file(filepath, query, self._transform_user, limit, batch_size, "Users")

    def load_reviews(self, filepath, limit=50000, batch_size=1000):
        query = """
        UNWIND $batch AS r
        MERGE (rev:Review {id: r.review_id})
        SET rev.stars = r.stars,
            rev.date = r.date,
            rev.text = substring(r.text, 0, 500),
            rev.useful = r.useful
            
        MERGE (u:User {id: r.user_id})
        MERGE (b:Business {id: r.business_id})
        
        MERGE (u)-[:WROTE]->(rev)
        MERGE (rev)-[:REVIEWS]->(b)
        """
        self._process_file(filepath, query, self._transform_review, limit, batch_size, "Reviews")

    def _transform_business(self, doc):
        return doc

    def _transform_user(self, doc):
        return doc

    def _transform_review(self, doc):
        return doc

    def _process_file(self, filepath, query, transform_fn, limit, batch_size, name):
        if not os.path.exists(filepath):
            print(f"{filepath} not found. Skipping {name}.")
            return

        print(f"Loading {name} into Neo4j...")
        with self.driver.session() as session:
            batch = []
            count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and count >= limit:
                        break
                    doc = json.loads(line)
                    batch.append(transform_fn(doc))
                    count += 1
                    
                    if len(batch) >= batch_size:
                        session.run(query, batch=batch)
                        batch = []
                        print(f"Loaded {count} {name}...")
                        
            if batch:
                session.run(query, batch=batch)
                print(f"Loaded {count} {name}...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../data', help='Path to JSON data')
    parser.add_argument('--limit', type=int, default=None, help='Subset limit for each file (leave empty for full dataset)')
    args = parser.parse_args()

    uri = os.getenv('NEO4J_URL')
    user = os.getenv('NEO4J_USER')
    password = os.getenv('NEO4J_PASSWORD')

    if not all([uri, user, password]):
        print("Error: Missing NEO4J_URL, NEO4J_USER, or NEO4J_PASSWORD in .env")
        return

    loader = YelpNeo4jLoader(uri, user, password)
    try:
        loader.clear_database()
        loader.create_indexes()
        
        # Load order: businesses, users, reviews
        loader.load_businesses(os.path.join(args.data_dir, 'yelp_academic_dataset_business.json'), limit=args.limit)
        loader.load_users(os.path.join(args.data_dir, 'yelp_academic_dataset_user.json'), limit=args.limit)
        loader.load_reviews(os.path.join(args.data_dir, 'yelp_academic_dataset_review.json'), limit=args.limit)
        
        print("Neo4j ETL successfully completed.")
    finally:
        loader.close()

if __name__ == '__main__':
    main()
