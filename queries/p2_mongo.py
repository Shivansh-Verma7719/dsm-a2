import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymongo import MongoClient
from rich.console import Console
from rich.panel import Panel


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report" / "p2"
FIG_ROOT = OUT_DIR / "figures"
DATA_ROOT = OUT_DIR / "data"
FIG_DIR = FIG_ROOT / "mongo"
DATA_DIR = DATA_ROOT / "mongo"
QUERIES_TXT = ROOT / "queries" / "mongo_part2_queries.txt"
console = Console()


def run_aggregate(db, collection_name: str, pipeline: list, step_label: str):
    collection = getattr(db, collection_name)
    with console.status(f"[bold cyan]{step_label}[/bold cyan]", spinner="dots"):
        started = time.perf_counter()
        rows = list(collection.aggregate(pipeline, allowDiskUse=True))
        elapsed = time.perf_counter() - started
    console.log(f"[green]Done[/green] {step_label} -> {len(rows)} rows in {elapsed:.2f}s")
    return rows


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (FIG_ROOT / "neo4j").mkdir(parents=True, exist_ok=True)
    (FIG_ROOT / "predictive_modelling").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "neo4j").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "predictive_modelling").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB", "yelp_db")
    client = MongoClient(uri)
    return client[db_name]


def ensure_indexes(db) -> None:
    """Create indexes used by heavy joins/grouping stages in Part 2 queries."""
    console.log("[cyan]Ensuring MongoDB indexes for Part 2 queries...[/cyan]")
    db.reviews.create_index("user_id")
    db.reviews.create_index([("business_id", 1), ("date", 1)])
    db.users.create_index("_id")
    db.businesses.create_index("_id")
    db.businesses.create_index("categories")
    db.checkins.create_index("_id")
    db.tips.create_index("business_id")
    console.log("[green]Index check complete.[/green]")


def dump_query_block(fh, title: str, collection_name: str, pipeline: list, note: str = "") -> None:
    fh.write("// ==========================================\n")
    fh.write(f"// {title}\n")
    if note:
        fh.write(f"// {note}\n")
    fh.write("// ==========================================\n\n")
    fh.write(f"db.{collection_name}.aggregate(\n")
    fh.write(json.dumps(pipeline, indent=4))
    fh.write("\n);\n\n")


def query1_cohort_analysis(db, fh):
    console.rule("[bold]Query 1: Cohort analysis")
    stats_pipeline = [
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "u",
            }
        },
        {"$unwind": "$u"},
        {"$match": {"u.yelping_since": {"$type": "date"}}},
        {
            "$addFields": {
                "cohort_year": {"$year": "$u.yelping_since"},
                "review_char_len": {"$strLenCP": {"$ifNull": ["$text", ""]}},
            }
        },
        {
            "$group": {
                "_id": "$cohort_year",
                "mean_star_rating": {"$avg": "$stars"},
                "stddev_star_rating": {"$stdDevPop": "$stars"},
                "mean_review_char_length": {"$avg": "$review_char_len"},
                "mean_useful_votes_per_review": {"$avg": "$useful"},
                "review_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    star_mix_pipeline = [
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "u",
            }
        },
        {"$unwind": "$u"},
        {"$match": {"u.yelping_since": {"$type": "date"}}},
        {
            "$addFields": {
                "cohort_year": {"$year": "$u.yelping_since"},
                "star_int": {"$toInt": "$stars"},
            }
        },
        {
            "$group": {
                "_id": {"cohort_year": "$cohort_year", "star": "$star_int"},
                "count": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.cohort_year",
                "total_reviews": {"$sum": "$count"},
                "star_buckets": {
                    "$push": {
                        "star": "$_id.star",
                        "count": "$count",
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 1,
                "total_reviews": 1,
                "star_buckets": 1,
            }
        },
        {"$sort": {"_id": 1}},
    ]

    dump_query_block(
        fh,
        "P2-Q1A. Cohort Stats (mean/std stars, length, useful)",
        "reviews",
        stats_pipeline,
    )
    dump_query_block(
        fh,
        "P2-Q1B. Cohort Star Distribution (1-5 proportions)",
        "reviews",
        star_mix_pipeline,
    )

    stats_rows = run_aggregate(
        db,
        "reviews",
        stats_pipeline,
        "Q1A aggregate: cohort-level stats",
    )
    star_rows = run_aggregate(
        db,
        "reviews",
        star_mix_pipeline,
        "Q1B aggregate: cohort star distribution",
    )

    stats_df = pd.DataFrame(stats_rows).rename(columns={"_id": "cohort_year"})

    dist_records = []
    for row in star_rows:
        cohort = row["_id"]
        total = row["total_reviews"]
        bucket_map = {item["star"]: item["count"] for item in row.get("star_buckets", [])}
        for star in [1, 2, 3, 4, 5]:
            cnt = bucket_map.get(star, 0)
            dist_records.append(
                {
                    "cohort_year": cohort,
                    "star": star,
                    "count": cnt,
                    "proportion": (cnt / total) if total else 0.0,
                }
            )
    dist_df = pd.DataFrame(dist_records)

    stats_df.to_csv(DATA_DIR / "q1_cohort_stats.csv", index=False)
    dist_df.to_csv(DATA_DIR / "q1_cohort_star_proportions.csv", index=False)

    highest_mean_star = stats_df.sort_values("mean_star_rating", ascending=False).iloc[0]
    highest_useful = stats_df.sort_values("mean_useful_votes_per_review", ascending=False).iloc[0]

    summary = {
        "highest_mean_star_cohort": int(highest_mean_star["cohort_year"]),
        "highest_mean_star_value": float(highest_mean_star["mean_star_rating"]),
        "highest_useful_cohort": int(highest_useful["cohort_year"]),
        "highest_useful_value": float(highest_useful["mean_useful_votes_per_review"]),
        "cohort_count": int(stats_df.shape[0]),
    }

    with (DATA_DIR / "q1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    heat = dist_df.pivot(index="cohort_year", columns="star", values="proportion").sort_index()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, cmap="YlGnBu", linewidths=0.2)
    plt.title("Q1: Star Proportions by Cohort Year")
    plt.xlabel("Star Rating")
    plt.ylabel("Cohort Year")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_cohort_star_heatmap.png", dpi=220)
    plt.close()

    trend_df = stats_df.sort_values("cohort_year")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        trend_df["cohort_year"],
        trend_df["mean_star_rating"],
        color="#1f77b4",
        marker="o",
        label="Mean Star Rating",
    )
    ax1.set_ylabel("Mean Star Rating", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Cohort Year")

    ax2 = ax1.twinx()
    ax2.plot(
        trend_df["cohort_year"],
        trend_df["mean_useful_votes_per_review"],
        color="#d62728",
        marker="s",
        label="Mean Useful Votes/Review",
    )
    ax2.set_ylabel("Mean Useful Votes per Review", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    plt.title("Q1: Cohort Means for Stars and Useful Votes")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "q1_cohort_metric_lines.png", dpi=220)
    plt.close()


def query2_mom_consistency(db, fh):
    console.rule("[bold]Query 2: MoM consistency")
    monthly_pipeline = [
        {
            "$lookup": {
                "from": "businesses",
                "localField": "business_id",
                "foreignField": "_id",
                "as": "b",
            }
        },
        {"$unwind": "$b"},
        {"$match": {"b.categories": {"$type": "array"}}},
        {"$unwind": "$b.categories"},
        {
            "$group": {
                "_id": {
                    "category": "$b.categories",
                    "year": {"$year": "$date"},
                    "month": {"$month": "$date"},
                },
                "avg_star": {"$avg": "$stars"},
                "review_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.category": 1, "_id.year": 1, "_id.month": 1}},
        {
            "$group": {
                "_id": "$_id.category",
                "total_reviews": {"$sum": "$review_count"},
                "monthly": {
                    "$push": {
                        "year": "$_id.year",
                        "month": "$_id.month",
                        "avg_star": "$avg_star",
                        "review_count": "$review_count",
                    }
                },
            }
        },
        {"$match": {"total_reviews": {"$gte": 500}}},
        {"$sort": {"total_reviews": -1}},
    ]

    dump_query_block(
        fh,
        "P2-Q2. Monthly Category Ratings + Trend Consistency Base",
        "reviews",
        monthly_pipeline,
    )

    rows = run_aggregate(
        db,
        "reviews",
        monthly_pipeline,
        "Q2 aggregate: monthly category ratings",
    )

    records = []
    line_records = []
    for row in rows:
        category = row["_id"]
        monthly = row.get("monthly", [])
        if len(monthly) < 2:
            continue

        inc = 0
        dec = 0
        for i in range(1, len(monthly)):
            delta = monthly[i]["avg_star"] - monthly[i - 1]["avg_star"]
            if delta > 0:
                inc += 1
            elif delta < 0:
                dec += 1

        pair_count = len(monthly) - 1
        records.append(
            {
                "category": category,
                "total_reviews": row["total_reviews"],
                "month_points": len(monthly),
                "increase_ratio": inc / pair_count,
                "decrease_ratio": dec / pair_count,
            }
        )

        for point in monthly:
            line_records.append(
                {
                    "category": category,
                    "date": pd.Timestamp(point["year"], point["month"], 1),
                    "avg_star": point["avg_star"],
                }
            )

    result_df = pd.DataFrame(records).sort_values("increase_ratio", ascending=False)
    line_df = pd.DataFrame(line_records)

    top_up = result_df.sort_values("increase_ratio", ascending=False).head(3)
    top_down = result_df.sort_values("decrease_ratio", ascending=False).head(3)

    result_df.to_csv(DATA_DIR / "q2_category_mom_consistency.csv", index=False)
    top_up.to_csv(DATA_DIR / "q2_top3_upward.csv", index=False)
    top_down.to_csv(DATA_DIR / "q2_top3_downward.csv", index=False)

    plt.figure(figsize=(10, 6))
    top_plot = top_up[["category", "increase_ratio"]].copy()
    top_plot["direction"] = "Upward"
    down_plot = top_down[["category", "decrease_ratio"]].copy()
    down_plot = down_plot.rename(columns={"decrease_ratio": "increase_ratio"})
    down_plot["direction"] = "Downward"
    bar_df = pd.concat([top_plot, down_plot], ignore_index=True)

    sns.barplot(data=bar_df, x="increase_ratio", y="category", hue="direction", orient="h")
    plt.title("Q2: Most Consistent Upward/Downward Categories")
    plt.xlabel("Consistency Ratio")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q2_trend_consistency_bars.png", dpi=220)
    plt.close()

    chosen_categories = list(top_up["category"]) + list(top_down["category"])
    plot_line = line_df[line_df["category"].isin(chosen_categories)].copy()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_line, x="date", y="avg_star", hue="category")
    plt.title("Q2: Monthly Average Star Trajectories (Top Up/Down Categories)")
    plt.xlabel("Month")
    plt.ylabel("Average Star")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q2_selected_category_monthly_lines.png", dpi=220)
    plt.close()


def query3_checkin_class_crosstab(db, fh):
    console.rule("[bold]Query 3: Check-in class cross-tab")
    checkin_base_pipeline = [
        {
            "$project": {
                "business_id": "$_id",
                "checkin_count": {"$size": {"$ifNull": ["$dates", []]}},
            }
        }
    ]

    top10_category_pipeline = [
        {"$match": {"categories": {"$type": "array"}}},
        {"$unwind": "$categories"},
        {
            "$group": {
                "_id": "$categories",
                "total_review_count": {"$sum": "$review_count"},
            }
        },
        {"$sort": {"total_review_count": -1}},
        {"$limit": 10},
    ]

    dump_query_block(
        fh,
        "P2-Q3A. Check-in Counts per Business (for quartiles)",
        "checkins",
        checkin_base_pipeline,
    )
    dump_query_block(
        fh,
        "P2-Q3B. Top-10 Categories by Total Review Count",
        "businesses",
        top10_category_pipeline,
    )

    checkin_rows = run_aggregate(
        db,
        "checkins",
        checkin_base_pipeline,
        "Q3A aggregate: check-in counts per business",
    )
    checkins_df = pd.DataFrame(checkin_rows)
    q1 = float(checkins_df["checkin_count"].quantile(0.25))
    q3 = float(checkins_df["checkin_count"].quantile(0.75))

    top10 = run_aggregate(
        db,
        "businesses",
        top10_category_pipeline,
        "Q3B aggregate: top-10 categories by review count",
    )
    top10_categories = [r["_id"] for r in top10]

    crosstab_pipeline = [
        {"$match": {"categories": {"$type": "array"}}},
        {"$match": {"categories": {"$in": top10_categories}}},
        {
            "$lookup": {
                "from": "checkins",
                "localField": "_id",
                "foreignField": "_id",
                "as": "c",
            }
        },
        {
            "$lookup": {
                "from": "tips",
                "let": {"bid": "$_id"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$business_id", "$$bid"]}}},
                    {"$group": {"_id": None, "tip_count": {"$sum": 1}}},
                ],
                "as": "tip_info",
            }
        },
        {
            "$addFields": {
                "checkin_count": {
                    "$ifNull": [
                        {
                            "$let": {
                                "vars": {"c1": {"$arrayElemAt": ["$c", 0]}},
                                "in": {"$size": {"$ifNull": ["$$c1.dates", []]}},
                            }
                        },
                        0,
                    ]
                },
                "tip_count": {
                    "$ifNull": [{"$arrayElemAt": ["$tip_info.tip_count", 0]}, 0]
                },
            }
        },
        {
            "$addFields": {
                "checkin_class": {
                    "$switch": {
                        "branches": [
                            {"case": {"$lte": ["$checkin_count", q1]}, "then": "low"},
                            {"case": {"$gt": ["$checkin_count", q3]}, "then": "high"},
                        ],
                        "default": "medium",
                    }
                }
            }
        },
        {"$unwind": "$categories"},
        {"$match": {"categories": {"$in": top10_categories}}},
        {
            "$group": {
                "_id": {"category": "$categories", "checkin_class": "$checkin_class"},
                "business_count": {"$sum": 1},
                "mean_star_rating": {"$avg": "$stars"},
                "mean_review_count": {"$avg": "$review_count"},
                "total_tips": {"$sum": "$tip_count"},
                "total_reviews": {"$sum": "$review_count"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "category": "$_id.category",
                "checkin_class": "$_id.checkin_class",
                "business_count": 1,
                "mean_star_rating": 1,
                "mean_review_count": 1,
                "tips_to_reviews_ratio": {
                    "$cond": [
                        {"$gt": ["$total_reviews", 0]},
                        {"$divide": ["$total_tips", "$total_reviews"]},
                        0,
                    ]
                },
            }
        },
        {"$sort": {"category": 1, "checkin_class": 1}},
    ]

    dump_query_block(
        fh,
        "P2-Q3C. Cross-tab by Check-in Class and Top-10 Categories",
        "businesses",
        crosstab_pipeline,
        note=f"Quartile cutoffs from check-ins: Q1={q1:.2f}, Q3={q3:.2f}",
    )

    crosstab_rows = run_aggregate(
        db,
        "businesses",
        crosstab_pipeline,
        "Q3C aggregate: cross-tab by class and category",
    )
    cross_df = pd.DataFrame(crosstab_rows)
    cross_df.to_csv(DATA_DIR / "q3_checkin_class_crosstab.csv", index=False)

    summary = {
        "q1_checkin_cutoff": q1,
        "q3_checkin_cutoff": q3,
        "top10_categories": top10_categories,
    }
    with (DATA_DIR / "q3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    heat = cross_df.pivot(
        index="category",
        columns="checkin_class",
        values="tips_to_reviews_ratio",
    )
    heat = heat.reindex(columns=["low", "medium", "high"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, cmap="OrRd", annot=False, linewidths=0.2)
    plt.title("Q3: Tips-to-Reviews Ratio by Check-in Class and Category")
    plt.xlabel("Check-in Frequency Class")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_tips_reviews_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=cross_df,
        x="checkin_class",
        y="mean_star_rating",
        hue="category",
        estimator=np.mean,
        errorbar=None,
    )
    plt.title("Q3: Mean Star Rating by Check-in Class (Top-10 Categories)")
    plt.xlabel("Check-in Frequency Class")
    plt.ylabel("Mean Star Rating")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Category")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q3_mean_star_by_class.png", dpi=220)
    plt.close()


def main():
    setup_dirs()
    sns.set_theme(style="whitegrid")

    console.print(
        Panel.fit(
            "DSM Assignment 2 Part 2\nMongoDB Querying Pipeline",
            title="Execution Start",
            border_style="cyan",
        )
    )

    db = get_db()
    ensure_indexes(db)
    with QUERIES_TXT.open("w", encoding="utf-8") as fh:
        fh.write("// DSM Assignment 2 Part 2 - MongoDB Querying\n")
        fh.write("// Generated by queries/p2_mongo.py\n\n")

        query1_cohort_analysis(db, fh)
        query2_mom_consistency(db, fh)
        query3_checkin_class_crosstab(db, fh)

    console.print(f"[bold green]Wrote query pipelines to:[/bold green] {QUERIES_TXT}")
    console.print(f"[bold green]Wrote result tables to:[/bold green] {DATA_DIR}")
    console.print(f"[bold green]Wrote figures to:[/bold green] {FIG_DIR}")


if __name__ == "__main__":
    main()