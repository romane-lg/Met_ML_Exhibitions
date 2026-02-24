from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
THEME_QUERIES: dict[str, list[str]] = {
    "portraits_people": ["portrait", "self portrait", "woman", "man", "face"],
    "landscape_nature": ["landscape", "garden", "river", "mountain", "tree"],
    "religion_myth": ["religious", "saint", "mythology", "biblical", "deity"],
    "architecture_city": ["architecture", "temple", "palace", "city", "building"],
    "objects_decorative": ["vase", "ceramic", "textile", "furniture", "jewelry"],
    "abstract_patterns": ["abstract", "pattern", "geometric", "ornament", "symbolic"],
}
OUTPUT_COLUMNS = [
    "objectID",
    "title",
    "artist",
    "department",
    "objectDate",
    "medium",
    "description",
    "image_path",
    "source_theme_bucket",
    "source_query",
]


@dataclass
class CollectStats:
    skipped_invalid_meta: int = 0
    skipped_missing_image_url: int = 0
    skipped_failed_image_download: int = 0
    skipped_duplicate: int = 0
    skipped_artist_cap: int = 0
    skipped_department_cap: int = 0
    request_failures: int = 0
    object_parse_failures: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect balanced MET data by theme buckets with safe retries and reporting."
    )
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--max-per-artist", type=int, default=20)
    parser.add_argument("--max-per-department", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-every", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-csv", type=str, default="data/raw/met_data.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--report-json", type=str, default="data/raw/collection_report.json")
    return parser.parse_args()


def request_json(
    session: requests.Session,
    url: str,
    timeout_seconds: float,
    max_retries: int,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(0.25 * (2**attempt), 2.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError("request_json failed with no error details.")


def request_search_ids(
    session: requests.Session,
    query: str,
    timeout_seconds: float,
    max_retries: int,
) -> list[int]:
    search_url = f"{BASE_URL}/search"
    params = {"hasImages": True, "q": query}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(search_url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            object_ids = payload.get("objectIDs") or []
            return [int(v) for v in object_ids if v is not None]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(0.25 * (2**attempt), 2.0))
    if last_error is not None:
        raise last_error
    return []


def is_valid_text(value: object) -> bool:
    return bool(str(value or "").strip())


def fetch_object(
    session: requests.Session,
    object_id: int,
    timeout_seconds: float,
    max_retries: int,
) -> dict:
    return request_json(
        session,
        f"{BASE_URL}/objects/{object_id}",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def download_image(
    session: requests.Session,
    img_url: str,
    img_path: Path,
    timeout_seconds: float,
    max_retries: int,
) -> bool:
    if img_path.exists():
        return True
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.get(img_url, timeout=timeout_seconds)
            response.raise_for_status()
            img_path.write_bytes(response.content)
            return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(0.25 * (2**attempt), 2.0))
    if last_error is not None:
        return False
    return False


def main() -> None:
    args = parse_args()
    target_size = max(1, int(args.target_size))
    rng = random.Random(args.seed)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    theme_names = list(THEME_QUERIES.keys())
    per_theme_quota = max(1, target_size // max(1, len(theme_names)))
    per_theme_limit = {theme: per_theme_quota for theme in theme_names}
    remaining = target_size - (per_theme_quota * len(theme_names))
    for theme in theme_names[:remaining]:
        per_theme_limit[theme] += 1

    stats = CollectStats()
    records: list[dict[str, object]] = []
    seen_object_ids: set[int] = set()
    artist_counts: Counter[str] = Counter()
    department_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()

    session = requests.Session()
    processed = 0

    for theme_name, queries in THEME_QUERIES.items():
        if len(records) >= target_size:
            break
        theme_target = per_theme_limit[theme_name]
        if theme_target <= 0:
            continue

        query_pool = list(queries)
        rng.shuffle(query_pool)

        for query in query_pool:
            if len(records) >= target_size or theme_counts[theme_name] >= theme_target:
                break
            try:
                candidate_ids = request_search_ids(
                    session=session,
                    query=query,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )
            except Exception:  # noqa: BLE001
                stats.request_failures += 1
                continue

            rng.shuffle(candidate_ids)
            for object_id in candidate_ids:
                if len(records) >= target_size or theme_counts[theme_name] >= theme_target:
                    break
                processed += 1
                if args.sleep_every > 0 and processed % args.sleep_every == 0:
                    time.sleep(args.sleep_seconds)
                if object_id in seen_object_ids:
                    stats.skipped_duplicate += 1
                    continue
                try:
                    obj = fetch_object(
                        session=session,
                        object_id=object_id,
                        timeout_seconds=args.timeout_seconds,
                        max_retries=args.max_retries,
                    )
                except Exception:  # noqa: BLE001
                    stats.request_failures += 1
                    continue
                if not isinstance(obj, dict):
                    stats.object_parse_failures += 1
                    continue

                title = str(obj.get("title") or "").strip()
                department = str(obj.get("department") or "").strip()
                artist = str(obj.get("artistDisplayName") or "").strip()
                img_url = str(obj.get("primaryImage") or "").strip()

                if not is_valid_text(title) or not is_valid_text(department):
                    stats.skipped_invalid_meta += 1
                    continue
                if not img_url:
                    stats.skipped_missing_image_url += 1
                    continue

                artist_key = artist.lower() if artist else "<unknown>"
                department_key = department.lower()
                if artist_counts[artist_key] >= max(1, args.max_per_artist):
                    stats.skipped_artist_cap += 1
                    continue
                if department_counts[department_key] >= max(1, args.max_per_department):
                    stats.skipped_department_cap += 1
                    continue

                img_path = images_dir / f"{object_id}.jpg"
                if not download_image(
                    session=session,
                    img_url=img_url,
                    img_path=img_path,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                ):
                    stats.skipped_failed_image_download += 1
                    continue

                record = {
                    "objectID": int(object_id),
                    "title": title,
                    "artist": artist,
                    "department": department,
                    "objectDate": str(obj.get("objectDate") or ""),
                    "medium": str(obj.get("medium") or ""),
                    "description": str(obj.get("creditLine") or ""),
                    "image_path": str(img_path),
                    "source_theme_bucket": theme_name,
                    "source_query": query,
                }
                records.append(record)
                seen_object_ids.add(int(object_id))
                artist_counts[artist_key] += 1
                department_counts[department_key] += 1
                theme_counts[theme_name] += 1
                query_counts[query] += 1

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.drop_duplicates(subset="objectID", keep="first")
        frame = frame.reindex(columns=OUTPUT_COLUMNS)
    frame.to_csv(output_path, index=False)

    report_payload = {
        "target_size": target_size,
        "final_size": int(len(frame)),
        "theme_counts": dict(sorted(theme_counts.items())),
        "query_counts": dict(sorted(query_counts.items())),
        "department_counts_top20": dict(Counter(department_counts).most_common(20)),
        "stats": {
            "skipped_invalid_meta": stats.skipped_invalid_meta,
            "skipped_missing_image_url": stats.skipped_missing_image_url,
            "skipped_failed_image_download": stats.skipped_failed_image_download,
            "skipped_duplicate": stats.skipped_duplicate,
            "skipped_artist_cap": stats.skipped_artist_cap,
            "skipped_department_cap": stats.skipped_department_cap,
            "request_failures": stats.request_failures,
            "object_parse_failures": stats.object_parse_failures,
        },
        "output_csv": str(output_path),
        "images_dir": str(images_dir),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved metadata: {output_path} ({len(frame)} rows)")
    print(f"Saved collection report: {report_path}")
    print(f"Theme counts: {dict(sorted(theme_counts.items()))}")


if __name__ == "__main__":
    main()
