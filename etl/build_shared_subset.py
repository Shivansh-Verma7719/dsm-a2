import argparse
import json
from collections import defaultdict
from pathlib import Path


BUSINESS_FILE = "yelp_academic_dataset_business.json"
USER_FILE = "yelp_academic_dataset_user.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
CHECKIN_FILE = "yelp_academic_dataset_checkin.json"
TIP_FILE = "yelp_academic_dataset_tip.json"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_categories(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [c.strip() for c in raw.split(",") if c.strip()]
    return []


def allocate_state_quotas(candidates_by_state, target_businesses, max_states, min_per_state):
    ranked_states = sorted(
        candidates_by_state.keys(),
        key=lambda s: len(candidates_by_state[s]),
        reverse=True,
    )[:max_states]

    if not ranked_states:
        return {}

    total_in_scope = sum(len(candidates_by_state[s]) for s in ranked_states)
    quotas = {}

    for state in ranked_states:
        available = len(candidates_by_state[state])
        proportional = int(target_businesses * (available / total_in_scope))
        q = max(min_per_state, proportional)
        quotas[state] = min(q, available)

    current = sum(quotas.values())
    if current > target_businesses:
        for state in sorted(quotas.keys(), key=lambda s: quotas[s], reverse=True):
            if current <= target_businesses:
                break
            reducible = max(0, quotas[state] - min_per_state)
            if reducible == 0:
                continue
            cut = min(reducible, current - target_businesses)
            quotas[state] -= cut
            current -= cut
    elif current < target_businesses:
        for state in sorted(ranked_states, key=lambda s: len(candidates_by_state[s]) - quotas[s], reverse=True):
            if current >= target_businesses:
                break
            room = len(candidates_by_state[state]) - quotas[state]
            if room <= 0:
                continue
            add = min(room, target_businesses - current)
            quotas[state] += add
            current += add

    return quotas


def select_diverse_businesses(state_candidates, quota):
    by_category = defaultdict(list)
    for biz in state_candidates:
        primary = biz["primary_category"]
        by_category[primary].append(biz)

    for cat in by_category:
        by_category[cat].sort(key=lambda b: b["review_count"], reverse=True)

    categories = sorted(by_category.keys(), key=lambda c: len(by_category[c]), reverse=True)
    chosen = []

    while len(chosen) < quota:
        picked_this_round = 0
        for cat in categories:
            if len(chosen) >= quota:
                break
            if not by_category[cat]:
                continue
            chosen.append(by_category[cat].pop(0))
            picked_this_round += 1
        if picked_this_round == 0:
            break

    return chosen


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_subset(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    business_path = data_dir / BUSINESS_FILE
    user_path = data_dir / USER_FILE
    review_path = data_dir / REVIEW_FILE
    checkin_path = data_dir / CHECKIN_FILE
    tip_path = data_dir / TIP_FILE

    print("[1/7] Reading check-in business ids...")
    checkin_business_ids = set()
    for doc in iter_jsonl(checkin_path):
        bid = doc.get("business_id")
        if bid:
            checkin_business_ids.add(bid)

    print("[2/7] Scanning business candidates...")
    candidates_by_state = defaultdict(list)
    all_business_docs = {}
    for b in iter_jsonl(business_path):
        bid = b.get("business_id")
        if not bid:
            continue

        all_business_docs[bid] = b

        state = (b.get("state") or "").strip()
        city = (b.get("city") or "").strip()
        review_count = int(b.get("review_count") or 0)
        categories = normalize_categories(b.get("categories"))

        if not state or not city or not categories:
            continue
        if review_count < args.min_business_reviews:
            continue
        if args.require_checkin and bid not in checkin_business_ids:
            continue

        candidates_by_state[state].append(
            {
                "business_id": bid,
                "state": state,
                "city": city,
                "review_count": review_count,
                "categories": categories,
                "primary_category": categories[0],
            }
        )

    quotas = allocate_state_quotas(
        candidates_by_state,
        target_businesses=args.target_businesses,
        max_states=args.max_states,
        min_per_state=args.min_businesses_per_state,
    )

    print("[3/7] Selecting businesses with state/category diversity...")
    selected_business_ids = set()
    selected_business_rows = []
    for state, quota in quotas.items():
        state_candidates = candidates_by_state[state]
        chosen = select_diverse_businesses(state_candidates, quota)
        for c in chosen:
            bid = c["business_id"]
            selected_business_ids.add(bid)
            selected_business_rows.append(all_business_docs[bid])

    print(f"    Selected businesses: {len(selected_business_ids)}")

    print("[4/7] Selecting reviews and referenced users...")
    per_business_review_counter = defaultdict(int)
    selected_user_ids = set()
    selected_reviews = []
    for r in iter_jsonl(review_path):
        bid = r.get("business_id")
        if bid not in selected_business_ids:
            continue
        if per_business_review_counter[bid] >= args.max_reviews_per_business:
            continue

        selected_reviews.append(r)
        per_business_review_counter[bid] += 1

        uid = r.get("user_id")
        if uid:
            selected_user_ids.add(uid)

    print(f"    Selected reviews: {len(selected_reviews)}")
    print(f"    Referenced users: {len(selected_user_ids)}")

    print("[5/7] Extracting users and pruning friend lists to subset users...")
    selected_users = []
    for u in iter_jsonl(user_path):
        uid = u.get("user_id")
        if uid not in selected_user_ids:
            continue

        friends = u.get("friends")
        if isinstance(friends, str):
            friend_list = [f.strip() for f in friends.split(",") if f.strip() and f.strip().lower() != "none"]
            friend_list = [f for f in friend_list if f in selected_user_ids]
            u["friends"] = ",".join(friend_list)

        selected_users.append(u)

    print(f"    Selected users: {len(selected_users)}")

    print("[6/7] Extracting check-ins and tips for selected businesses...")
    selected_checkins = [c for c in iter_jsonl(checkin_path) if c.get("business_id") in selected_business_ids]
    selected_tips = [t for t in iter_jsonl(tip_path) if t.get("business_id") in selected_business_ids]

    print(f"    Selected checkins: {len(selected_checkins)}")
    print(f"    Selected tips: {len(selected_tips)}")

    print("[7/7] Writing subset files...")
    write_jsonl(out_dir / BUSINESS_FILE, selected_business_rows)
    write_jsonl(out_dir / USER_FILE, selected_users)
    write_jsonl(out_dir / REVIEW_FILE, selected_reviews)
    write_jsonl(out_dir / CHECKIN_FILE, selected_checkins)
    write_jsonl(out_dir / TIP_FILE, selected_tips)

    states_summary = sorted(
        ((s, q) for s, q in quotas.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    manifest = {
        "subset_name": args.subset_name,
        "created_from": str(data_dir),
        "output_dir": str(out_dir),
        "parameters": {
            "target_businesses": args.target_businesses,
            "max_states": args.max_states,
            "min_businesses_per_state": args.min_businesses_per_state,
            "min_business_reviews": args.min_business_reviews,
            "max_reviews_per_business": args.max_reviews_per_business,
            "require_checkin": args.require_checkin,
        },
        "counts": {
            "businesses": len(selected_business_rows),
            "reviews": len(selected_reviews),
            "users": len(selected_users),
            "checkins": len(selected_checkins),
            "tips": len(selected_tips),
        },
        "state_business_allocations": [{"state": s, "businesses": q} for s, q in states_summary],
    }

    with (out_dir / "subset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Subset generation complete.")
    print(json.dumps(manifest["counts"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build one shared Yelp subset for MongoDB, Neo4j, and predictive modelling. "
            "The subset is materialized as smaller JSONL files."
        )
    )
    parser.add_argument("--data-dir", default="../data", help="Directory containing full Yelp JSONL files")
    parser.add_argument(
        "--out-dir",
        default="../data/subset_shared",
        help="Directory where subset JSONL files will be written",
    )
    parser.add_argument("--subset-name", default="shared_v1", help="Name recorded in subset manifest")
    parser.add_argument("--target-businesses", type=int, default=6000, help="Approximate businesses to keep")
    parser.add_argument("--max-states", type=int, default=12, help="Maximum states to include")
    parser.add_argument(
        "--min-businesses-per-state",
        type=int,
        default=250,
        help="Minimum businesses per selected state (if available)",
    )
    parser.add_argument(
        "--min-business-reviews",
        type=int,
        default=15,
        help="Minimum business.review_count required for candidate selection",
    )
    parser.add_argument(
        "--max-reviews-per-business",
        type=int,
        default=120,
        help="Cap reviews kept per selected business",
    )
    parser.add_argument(
        "--require-checkin",
        action="store_true",
        help="Only select businesses that have check-in records",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_subset(parse_args())
