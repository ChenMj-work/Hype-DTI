import argparse
import csv
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def read_id2text(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ent_id = int(row[0])
            except ValueError:
                continue
            text = row[1].strip()
            if text:
                rows.append((ent_id, text))
    return rows


def mean_pool(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return pooled


def normalize_protein_sequence(seq):
    seq = seq.replace(" ", "").replace("\n", "")
    return " ".join(list(seq))


def encode_rows(rows, model_name, output_path, batch_size, max_length, device, protein_mode=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        saved = torch.load(output_path, map_location="cpu")
        embeddings = saved.get("embeddings", saved) if isinstance(saved, dict) else saved
    else:
        embeddings = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    pending = [(ent_id, text) for ent_id, text in rows if ent_id not in embeddings]
    print(f"[precompute] {model_name} pending {len(pending)}/{len(rows)} -> {output_path}")
    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            ids = [ent_id for ent_id, _ in batch]
            texts = [normalize_protein_sequence(text) if protein_mode else text for _, text in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs, encoded["attention_mask"]).detach().cpu()
            for ent_id, vector in zip(ids, pooled):
                embeddings[int(ent_id)] = vector.float()
            if (start // batch_size) % 20 == 0:
                torch.save({"model": model_name, "embeddings": embeddings}, output_path)
                print(f"[precompute] saved {len(embeddings)}/{len(rows)}")
    torch.save({"model": model_name, "embeddings": embeddings}, output_path)
    print(f"[precompute] done {len(embeddings)}/{len(rows)} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="var_data")
    parser.add_argument("--output_dir", default="var_features/pretrained_self")
    parser.add_argument("--drug_model", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--protein_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--protein_batch_size", type=int, default=2)
    parser.add_argument("--drug_max_length", type=int, default=256)
    parser.add_argument("--protein_max_length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_drug", action="store_true")
    parser.add_argument("--skip_protein", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    smiles_path = data_root / "ent_id2smiles.csv"
    seq_path = data_root / "ent_id2seq.csv"
    if not args.skip_drug:
        encode_rows(
            read_id2text(smiles_path),
            args.drug_model,
            output_dir / "pretrained_self_drug_embed.pt",
            args.batch_size,
            args.drug_max_length,
            args.device,
            protein_mode=False,
        )
    if not args.skip_protein:
        encode_rows(
            read_id2text(seq_path),
            args.protein_model,
            output_dir / "pretrained_self_protein_embed.pt",
            args.protein_batch_size,
            args.protein_max_length,
            args.device,
            protein_mode=True,
        )


if __name__ == "__main__":
    main()
