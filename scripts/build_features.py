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
    parser.add_argument("--embedding-backend", choices=["tfidf", "clip"], default="tfidf")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--clip-device", type=str, default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--clip-text-weight", type=float, default=0.5)
    parser.add_argument("--clip-image-weight", type=float, default=0.5)
    parser.add_argument("--clip-retrieval-weight", type=float, default=0.8)
    parser.add_argument("--clip-lexical-weight", type=float, default=0.2)
    parser.add_argument("--clip-prompt-ensemble", action="store_true", default=True)
    parser.add_argument("--no-clip-prompt-ensemble", dest="clip_prompt_ensemble", action="store_false")
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
        embedding_backend=args.embedding_backend,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        clip_device=args.clip_device,
        clip_batch_size=args.clip_batch_size,
        clip_text_weight=args.clip_text_weight,
        clip_image_weight=args.clip_image_weight,
        clip_retrieval_weight=args.clip_retrieval_weight,
        clip_lexical_weight=args.clip_lexical_weight,
        clip_prompt_ensemble=args.clip_prompt_ensemble,
    )


if __name__ == "__main__":
    main()
