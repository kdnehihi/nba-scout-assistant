from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.pipelines.data import run_online_data_update_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for one incremental data refresh."""
    parser = argparse.ArgumentParser(
        description="Update NBA Scout Assistant inference data from BALLDONTLIE."
    )
    parser.add_argument("--data-dir", default="data", help="Project data root.")
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive YYYY-MM-DD cutoff; defaults to today.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=2,
        help="Previously processed dates to refetch for source corrections.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the online update and print its persisted outputs."""
    load_dotenv()
    args = parse_args()
    result = run_online_data_update_pipeline(
        data_dir=args.data_dir,
        end_date=args.end_date,
        overlap_days=args.overlap_days,
    )
    requested_range = result.fetch.date_range
    print("Requested range:", requested_range)
    print("Raw API rows:", result.fetch.raw_row_count)
    print("Mapped incoming rows:", result.upsert.incoming_rows)
    print("Inserted Silver rows:", result.upsert.inserted_rows)
    print("Replaced Silver rows:", result.upsert.replaced_rows)
    print("Unmatched API rows:", result.upsert.unmatched_rows)
    print("Bronze snapshot:", result.bronze_snapshot_path)
    print("Silver game logs:", result.silver_game_logs_path)
    for name, path in result.gold_output_paths.items():
        print(f"Gold {name}:", path)
    print("Checkpoint:", result.checkpoint_path)


if __name__ == "__main__":
    main()
