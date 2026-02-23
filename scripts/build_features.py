from __future__ import annotations

import argparse
import logging

from src.features.build_pipeline import (
    atomic_pickle_dump,
    build_combined_embeddings_payload,
    build_numeric_feature_matrix,
    build_text,
    extract_metadata_numeric_features,
    is_supported_image_file,
    resolve_image_path,
    run_build,
    tokenize_local,
)

__all__ = [
    "atomic_pickle_dump",
    "build_combined_embeddings_payload",
    "build_numeric_feature_matrix",
    "build_text",
    "extract_metadata_numeric_features",
    "is_supported_image_file",
    "resolve_image_path",
    "run_build",
    "tokenize_local",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--pca-max-components", type=int, default=256)
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--vision-weight", type=float, default=1.0)
    parser.add_argument("--numeric-weight", type=float, default=1.0)
    args = parser.parse_args()
    run_build(
        limit=args.limit,
        force=args.force,
        offline=args.offline,
        pca_variance=args.pca_variance,
        pca_max_components=args.pca_max_components,
        text_weight=args.text_weight,
        vision_weight=args.vision_weight,
        numeric_weight=args.numeric_weight,
    )


if __name__ == "__main__":
    main()
