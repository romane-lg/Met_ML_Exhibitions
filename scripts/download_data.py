import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
IMG_DIR = Path("data/raw/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Get object IDs with images
search_params = {"hasImages": True, "q": "*"}
response = requests.get(f"{BASE_URL}/search", params=search_params)
response.raise_for_status()

object_ids = response.json()["objectIDs"]
print(f"Found {len(object_ids)} objects with images")

records = []

# 2. Fetch object data + download images
for i, object_id in enumerate(object_ids[:500]):  # start small
    try:
        obj = requests.get(f"{BASE_URL}/objects/{object_id}").json()

        img_url = obj.get("primaryImage")
        if not img_url:
            continue

        img_path = IMG_DIR / f"{object_id}.jpg"

        # Download image only once
        if not img_path.exists():
            img_data = requests.get(img_url).content
            img_path.write_bytes(img_data)

        records.append({
            "objectID": object_id,
            "title": obj.get("title"),
            "artist": obj.get("artistDisplayName"),
            "department": obj.get("department"),
            "objectDate": obj.get("objectDate"),
            "medium": obj.get("medium"),
            "image_path": str(img_path)
        })

        # Be polite to the API
        if i % 50 == 0:
            time.sleep(0.2)

    except Exception as e:
        print(f"Skipping object {object_id}: {e}")

# 3. Export metadata
df = pd.DataFrame(records)
output_path = Path("data/raw/met_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Saved {output_path} and downloaded {len(records)} images to {IMG_DIR}.")
