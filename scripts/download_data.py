import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
IMG_DIR = Path("data/raw/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 1000
OUTPUT_CSV = Path("data/raw/met_data.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TIMEOUT_OBJ = 15
TIMEOUT_IMG = 30
MAX_RETRIES = 3
SLEEP_EVERY = 10          # sleep every N attempted objects
SLEEP_SECONDS = 0.8
COOLDOWN_403_THRESHOLD_1 = 10
COOLDOWN_403_SECONDS_1 = 60
COOLDOWN_403_THRESHOLD_2 = 20
COOLDOWN_403_SECONDS_2 = 180

def get_json_with_retries(session: requests.Session, url: str, timeout: int, max_retries: int):
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # 403 is typically a hard deny for this object/request context; retrying just increases noise.
            if status == 403:
                raise
            last_err = e
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
        except Exception as e:
            last_err = e
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
    raise last_err

def download_with_retries(session: requests.Session, url: str, path: Path, timeout: int, max_retries: int) -> bool:
    if path.exists():
        return True
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            path.write_bytes(r.content)
            return True
        except Exception as e:
            last_err = e
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
    return False

# 0) Resume: load already-valid rows (CSV row + matching image file) so we skip them.
records = []
seen_ids = set()
if OUTPUT_CSV.exists():
    try:
        existing = pd.read_csv(OUTPUT_CSV, dtype={"objectID": int})
        for _, row in existing.iterrows():
            oid = int(row["objectID"])
            img_path = IMG_DIR / f"{oid}.jpg"
            if img_path.exists():
                records.append(row.to_dict())
                seen_ids.add(oid)
        print(f"Resuming: {len(records)} already-valid rows loaded from existing CSV (will skip these).")
    except Exception as e:
        print(f"Could not load existing CSV for resume ({e}); starting fresh.")
        records = []
        seen_ids = set()

# 1) Search ALL object IDs that "have images"
# Note: still not perfect, so we validate per-object later.
search_params = {"hasImages": True, "q": "*"}
with requests.Session() as session:
    resp = session.get(f"{BASE_URL}/search", params=search_params, timeout=TIMEOUT_OBJ)
    resp.raise_for_status()
    payload = resp.json()
    object_ids = payload.get("objectIDs") or []

print(f"Found {len(object_ids)} candidate objects with images")

with requests.Session() as session:
    attempted = 0
    idx = 0
    consecutive_403 = 0

    while len(records) < TARGET and idx < len(object_ids):
        object_id = object_ids[idx]
        idx += 1

        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)

        attempted += 1
        if SLEEP_EVERY > 0 and attempted % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

        try:
            obj = get_json_with_retries(
                session,
                f"{BASE_URL}/objects/{object_id}",
                timeout=TIMEOUT_OBJ,
                max_retries=MAX_RETRIES,
            )

            # prefer primaryImage, but fall back to primaryImageSmall
            img_url = (obj.get("primaryImage") or "").strip() or (obj.get("primaryImageSmall") or "").strip()
            if not img_url:
                continue

            img_path = IMG_DIR / f"{object_id}.jpg"
            ok = download_with_retries(
                session,
                img_url,
                img_path,
                timeout=TIMEOUT_IMG,
                max_retries=MAX_RETRIES,
            )
            if not ok:
                continue

            records.append({
                "objectID": int(object_id),
                "title": obj.get("title"),
                "artist": obj.get("artistDisplayName"),
                "department": obj.get("department"),
                "objectDate": obj.get("objectDate"),
                "medium": obj.get("medium"),
                "image_path": str(img_path)
            })
            consecutive_403 = 0
            if len(records) % 25 == 0:
                print(f"Collected {len(records)}/{TARGET}... (attempted {attempted})")

        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 403:
                consecutive_403 += 1
                if consecutive_403 == COOLDOWN_403_THRESHOLD_1:
                    print(
                        f"Hit {COOLDOWN_403_THRESHOLD_1} consecutive 403s. "
                        f"Cooling down for {COOLDOWN_403_SECONDS_1}s..."
                    )
                    time.sleep(COOLDOWN_403_SECONDS_1)
                elif consecutive_403 == COOLDOWN_403_THRESHOLD_2:
                    print(
                        f"Hit {COOLDOWN_403_THRESHOLD_2} consecutive 403s. "
                        f"Cooling down for {COOLDOWN_403_SECONDS_2}s..."
                    )
                    time.sleep(COOLDOWN_403_SECONDS_2)
                elif consecutive_403 % 10 == 0:
                    print(f"Skipping 403 object {object_id} (streak={consecutive_403})")
            else:
                consecutive_403 = 0
                print(f"Skipping object {object_id}: HTTP {status}")
        except Exception as e:
            consecutive_403 = 0
            # Keep going until we reach TARGET successes
            print(f"Skipping object {object_id}: {e}")
            continue

# 3) Export metadata with strict CSV-to-image consistency checks.
df = pd.DataFrame(records)
if not df.empty:
    df["objectID"] = df["objectID"].astype(str)
    df = df.drop_duplicates(subset=["objectID"], keep="first").copy()
    df = df[df["objectID"].apply(lambda oid: (IMG_DIR / f"{oid}.jpg").exists())].copy()
    df["objectID"] = df["objectID"].astype(int)
    df["image_path"] = df["objectID"].apply(lambda oid: str(IMG_DIR / f"{oid}.jpg"))

# Safety check: ensure 1000 rows if possible
if len(df) < TARGET:
    print(f"Warning: only collected {len(df)} valid image rows before running out of candidate IDs.")
else:
    df = df.iloc[:TARGET].copy()

df.to_csv(OUTPUT_CSV, index=False)

# Remove any image files in IMG_DIR that are not referenced in the CSV.
csv_ids = set(df["objectID"].astype(str).tolist())
removed = 0
for img_file in IMG_DIR.glob("*.jpg"):
    if img_file.stem not in csv_ids:
        img_file.unlink()
        removed += 1
if removed:
    print(f"Removed {removed} orphan image(s) not present in the CSV.")

image_count = len(list(IMG_DIR.glob("*.jpg")))
print(
    f"Saved {OUTPUT_CSV} with {len(df)} rows. "
    f"Current image files in {IMG_DIR}: {image_count}."
)
