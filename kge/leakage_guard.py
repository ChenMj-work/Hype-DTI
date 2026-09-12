"""Runtime guards for pseudo-labelled DTI training samples.

This module intentionally depends only on the Python standard library so that
the guard cannot perturb model state, CUDA state, or random-number streams.
"""

from __future__ import absolute_import, division, print_function

import csv
import os


COLD_DATASET_MARKERS = ("cold", "coldstart", "cold_start", "_cs_", "-cs-")


def _pair(triple):
    return int(triple[0]), int(triple[2])


def is_cold_dataset(dataset):
    name = str(dataset).lower()
    return any(marker in name for marker in COLD_DATASET_MARKERS)


def filter_protected_candidates(
    candidates,
    dataset,
    train_pos,
    train_neg,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
):
    """Remove evaluation pairs and, for cold splits, evaluation-only entities."""
    protected_pairs = {
        _pair(triple)
        for triple in list(valid_pos) + list(valid_neg) + list(test_pos) + list(test_neg)
    }
    train_all = list(train_pos) + list(train_neg)
    eval_all = list(valid_pos) + list(valid_neg) + list(test_pos) + list(test_neg)
    train_drugs = {int(triple[0]) for triple in train_all}
    train_targets = {int(triple[2]) for triple in train_all}
    heldout_drugs = {int(triple[0]) for triple in eval_all} - train_drugs
    heldout_targets = {int(triple[2]) for triple in eval_all} - train_targets
    enforce_cold = is_cold_dataset(dataset)

    kept = []
    pair_removed = 0
    cold_entity_removed = 0
    for triple in candidates:
        if _pair(triple) in protected_pairs:
            pair_removed += 1
            continue
        if enforce_cold and (
            int(triple[0]) in heldout_drugs or int(triple[2]) in heldout_targets
        ):
            cold_entity_removed += 1
            continue
        kept.append(triple)
    return kept, {
        "input_count": len(candidates),
        "output_count": len(kept),
        "evaluation_pair_removed": pair_removed,
        "cold_entity_removed": cold_entity_removed,
        "cold_entity_guard": enforce_cold,
    }


def audit_training_selection(
    dataset,
    stage,
    selected_samples,
    train_pos,
    train_neg,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    id2entity,
    id2relation,
    output_path,
    enforce_cold_entities=None,
):
    """Write an audit trail and raise before optimization on leakage.

    ``selected_samples`` is an iterable of ``(kind, triple, score)``. The
    exact drug-target pair is protected regardless of relation id. In a cold
    dataset, selected samples containing an evaluation-only entity are also
    rejected.
    """
    protected = {
        "valid_pos": {_pair(triple) for triple in valid_pos},
        "valid_neg": {_pair(triple) for triple in valid_neg},
        "test_pos": {_pair(triple) for triple in test_pos},
        "test_neg": {_pair(triple) for triple in test_neg},
    }
    train_all = list(train_pos) + list(train_neg)
    eval_all = list(valid_pos) + list(valid_neg) + list(test_pos) + list(test_neg)
    train_drugs = {int(triple[0]) for triple in train_all}
    train_targets = {int(triple[2]) for triple in train_all}
    heldout_drugs = {int(triple[0]) for triple in eval_all} - train_drugs
    heldout_targets = {int(triple[2]) for triple in eval_all} - train_targets

    if enforce_cold_entities is None:
        enforce_cold_entities = is_cold_dataset(dataset)

    rows = []
    violations = []
    for rank, (kind, triple, score) in enumerate(selected_samples):
        head_id, relation_id, tail_id = (int(value) for value in triple)
        pair = (head_id, tail_id)
        overlap_splits = [name for name, pairs in protected.items() if pair in pairs]
        heldout_drug = head_id in heldout_drugs
        heldout_target = tail_id in heldout_targets
        violation_types = []
        if overlap_splits:
            violation_types.append("evaluation_pair")
        if enforce_cold_entities and (heldout_drug or heldout_target):
            violation_types.append("cold_entity")
        if violation_types:
            violations.append((pair, violation_types))
        rows.append({
            "dataset": dataset,
            "stage": stage,
            "kind": kind,
            "rank": rank,
            "head_id": head_id,
            "relation_id": relation_id,
            "tail_id": tail_id,
            "head": id2entity[head_id],
            "relation": id2relation[relation_id],
            "tail": id2entity[tail_id],
            "score": float(score),
            "overlap_splits": ";".join(overlap_splits),
            "heldout_drug": int(heldout_drug),
            "heldout_target": int(heldout_target),
            "cold_entity_guard": int(bool(enforce_cold_entities)),
            "status": "FAIL" if violation_types else "PASS",
            "violation_types": ";".join(violation_types),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "dataset", "stage", "kind", "rank", "head_id", "relation_id",
        "tail_id", "head", "relation", "tail", "score",
        "overlap_splits", "heldout_drug", "heldout_target",
        "cold_entity_guard", "status", "violation_types",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if violations:
        preview = ", ".join(
            "{}:{}".format(pair, "/".join(types))
            for pair, types in violations[:5]
        )
        raise RuntimeError(
            "KGE leakage guard rejected {} pseudo-labelled samples at stage {}. "
            "See {}. First violations: {}".format(
                len(violations), stage, output_path, preview
            )
        )

    print(
        "[leakage-guard] stage={} selected={} evaluation_pair_hits=0 "
        "cold_entity_guard={} saved={}".format(
            stage, len(rows), bool(enforce_cold_entities), output_path
        )
    )
    return rows
