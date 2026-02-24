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

# 1) Search ALL object IDs that "have images"
# Note: still not perfect, so we validate per-object later.
search_params = {"hasImages": True, "q": "*"}
with requests.Session() as session:
    resp = session.get(f"{BASE_URL}/search", params=search_params, timeout=TIMEOUT_OBJ)
    resp.raise_for_status()
    payload = resp.json()
    object_ids = payload.get("objectIDs") or []

print(f"Found {len(object_ids)} candidate objects with images")

records = []
seen_ids = set()

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

# 3) Export metadata (exactly len(records) rows)
df = pd.DataFrame(records)

# Safety check: ensure 1000 rows if possible
if len(df) < TARGET:
    print(f"Warning: only collected {len(df)} images before running out of candidate IDs.")
else:
    df = df.iloc[:TARGET].copy()

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {OUTPUT_CSV} with {len(df)} rows and downloaded {len(df)} images to {IMG_DIR}.")
