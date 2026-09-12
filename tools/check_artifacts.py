import argparse
import hashlib
from pathlib import Path

import torch

from download_kge_checkpoint import FILENAME, SHA256


DATASETS = [
    f"{target}_10shots_{fold}"
    for target in ("ago", "blocker", "e-")
    for fold in range(5)
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    required = [
        root / "var_data/entities.dict",
        root / "var_data/relations.dict",
        root / "var_data/ent_id2smiles.csv",
        root / "var_data/ent_id2seq.csv",
        root / "var_features/pretrained_self/pretrained_self_drug_embed.pt",
        root / "var_features/pretrained_self/pretrained_self_protein_embed.pt",
    ]
    split_names = (
        "train.tsv",
        "train_neg.tsv",
        "valid.tsv",
        "valid_neg.tsv",
        "test.tsv",
        "test_neg.tsv",
    )
    required.extend(
        root / "var_data" / dataset / name
        for dataset in DATASETS
        for name in split_names
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n  " + "\n  ".join(missing))
    empty = [str(path) for path in required if path.stat().st_size == 0]
    if empty:
        raise SystemExit("Empty required artifacts:\n  " + "\n  ".join(empty))

    feature_specs = (
        (root / "var_features/pretrained_self/pretrained_self_drug_embed.pt", 768),
        (root / "var_features/pretrained_self/pretrained_self_protein_embed.pt", 1280),
    )
    for feature_path, expected_dim in feature_specs:
        saved = torch.load(feature_path, map_location="cpu")
        embeddings = saved.get("embeddings", saved) if isinstance(saved, dict) else saved
        if not embeddings:
            raise SystemExit(f"No embeddings found in {feature_path}")
        vector = next(iter(embeddings.values()))
        if vector.numel() != expected_dim:
            raise SystemExit(
                f"Embedding dimension mismatch in {feature_path}: "
                f"expected {expected_dim}, got {vector.numel()}"
            )

    checkpoint = root / "var_models" / FILENAME
    if not checkpoint.is_file():
        raise SystemExit(f"Missing KG checkpoint: {checkpoint}")
    actual = file_sha256(checkpoint)
    if actual != SHA256:
        raise SystemExit(f"KG checkpoint checksum mismatch: {actual}")
    print(f"[check] all artifacts are compatible; KG SHA-256={actual}")


if __name__ == "__main__":
    main()
