from __future__ import annotations

import argparse
from pathlib import Path


def is_supported_image_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    signatures = [
        b"\xff\xd8\xff",  # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a",  # GIF
        b"GIF89a",  # GIF
        b"BM",  # BMP
        b"II*\x00",  # TIFF LE
        b"MM\x00*",  # TIFF BE
    ]
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return True
    return any(header.startswith(sig) for sig in signatures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find/remove invalid image files by signature.")
    parser.add_argument("--images-dir", default="data/raw/images", help="Directory containing image files.")
    parser.add_argument("--delete", action="store_true", help="Delete invalid files. Default is dry-run.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    invalid: list[Path] = []
    for path in images_dir.iterdir():
        if path.is_file() and not is_supported_image_file(path):
            invalid.append(path)

    print(f"checked_dir={images_dir}")
    print(f"invalid_count={len(invalid)}")
    for path in invalid:
        print(f"invalid: {path} ({path.stat().st_size} bytes)")

    if args.delete and invalid:
        for path in invalid:
            path.unlink(missing_ok=True)
            print(f"deleted: {path}")
        print("cleanup_complete=true")
    elif args.delete:
        print("cleanup_complete=true (nothing to delete)")
    else:
        print("dry_run=true")


if __name__ == "__main__":
    main()
