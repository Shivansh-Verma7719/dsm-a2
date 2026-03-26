import json
import os
import sys
from pymongo import MongoClient
import argparse
import dateutil.parser

def get_db():
    client = MongoClient('mongodb://localhost:27017/')
    return client['yelp_db']

def drop_collections(db):
    print("Dropping existing collections...")
    for coll in ['businesses', 'users', 'reviews', 'checkins', 'tips']:
        db[coll].drop()

def process_file(filepath, db, collection_name, transform_fn, batch_size=50000, limit=None):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Skipping.")
        return

    print(f"Processing {filepath} into {collection_name} collection...")
    collection = db[collection_name]
    batch = []
    count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
                
            doc = json.loads(line)
            doc = transform_fn(doc)
            
            if doc:
                batch.append(doc)
                count += 1
                
            if len(batch) >= batch_size:
                collection.insert_many(batch)
                print(f"Inserted {count} documents into {collection_name}...")
                batch = []

    if batch:
        collection.insert_many(batch)
        print(f"Inserted {count} documents into {collection_name}...")

    print(f"Finished processing {filepath}. Total loaded: {count}\n")

def transform_business(doc):
    doc['_id'] = doc.pop('business_id')
    
    if isinstance(doc.get('categories'), str):
        doc['categories'] = [c.strip() for c in doc['categories'].split(',')]
    return doc

def transform_user(doc):
    doc['_id'] = doc.pop('user_id')
    # Normalize string-encoded arrays used in Yelp JSON.
    friends_val = doc.get('friends')
    if isinstance(friends_val, str):
        doc['friends'] = [f.strip() for f in friends_val.split(',') if f.strip() and f.strip().lower() != 'none']

    elite_val = doc.get('elite')
    if isinstance(elite_val, str):
        doc['elite'] = [e.strip() for e in elite_val.split(',') if e.strip() and e.strip().lower() != 'none']

    if doc.get('yelping_since'):
        try:
            doc['yelping_since'] = dateutil.parser.parse(doc['yelping_since'])
        except Exception:
            pass
    return doc

def transform_review(doc):
    doc['_id'] = doc.pop('review_id')
    if doc.get('date'):
        try:
            doc['date'] = dateutil.parser.parse(doc['date'])
        except Exception:
            pass
    return doc

def transform_checkin(doc):
    doc['_id'] = doc.pop('business_id')
    date_str = doc.get('date', '')
    if date_str:
        doc['dates'] = [dateutil.parser.parse(ds.strip()) for ds in date_str.split(',') if ds.strip()]
    else:
        doc['dates'] = []
    doc.pop('date', None)
    return doc

def transform_tip(doc):
    if doc.get('date'):
        try:
            doc['date'] = dateutil.parser.parse(doc['date'])
        except Exception:
            pass
    return doc

def create_indexes(db):
    print("Creating indexes...")
    # Business indexes
    db.businesses.create_index([("city", 1), ("review_count", -1)])
    db.businesses.create_index([("categories", 1), ("review_count", -1)])
    
    # Review indexes
    db.reviews.create_index([("business_id", 1), ("date", 1)])
    db.reviews.create_index([("user_id", 1), ("date", 1)])

    # User index for tenure-bucket style analysis
    db.users.create_index([("yelping_since", 1)])
    
    # Checkins automatically indexed on _id
    print("Indexes created.")

def main():
    parser = argparse.ArgumentParser(description="Load Yelp Dataset into MongoDB")
    parser.add_argument('--data-dir', default='../data', help='Path to dataset directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of documents per file for testing')
    args = parser.parse_args()

    db = get_db()
    drop_collections(db)

    data_dir = args.data_dir
    files_to_process = [
        (os.path.join(data_dir, 'yelp_academic_dataset_business.json'), 'businesses', transform_business),
        (os.path.join(data_dir, 'yelp_academic_dataset_user.json'), 'users', transform_user),
        (os.path.join(data_dir, 'yelp_academic_dataset_review.json'), 'reviews', transform_review),
        (os.path.join(data_dir, 'yelp_academic_dataset_checkin.json'), 'checkins', transform_checkin),
        (os.path.join(data_dir, 'yelp_academic_dataset_tip.json'), 'tips', transform_tip)
    ]

    for filepath, collection, transform_fn in files_to_process:
        process_file(filepath, db, collection, transform_fn, limit=args.limit)

    create_indexes(db)
    print("ETL complete.")

if __name__ == '__main__':
    main()
