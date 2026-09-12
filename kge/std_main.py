"""
a forward of model (assuming batch size is 1) can compute the score for a positive triplet, or scores of all related
negative triplets(certainly containing the positive triplet itself)
There is no use of train_neg.txt
derive from my previous file kge_self.py
"""

# !/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gc
import math

import datetime

import json
import csv

from torch.utils.data import DataLoader


from utils_mine import *
from esm_model import *
from few_dataloader import *
from full_model_config import add_full_model_args
from leakage_guard import audit_training_selection, filter_protected_candidates
import itertools
import random
import time
from collections import defaultdict




def main(args, start_time):
    def ensure_diag_dir(args):
        diag_dir = os.path.join(args.pkl_path, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        return diag_dir

    def clone_state_dict(model):
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def restore_state_dict(model, state):
        device_state = {name: value.to(args.device) for name, value in state.items()}
        model.load_state_dict(device_state)

    def metric_saturated(aupr, auc, args):
        return aupr >= args.gate_saturation_level and auc >= args.gate_saturation_level

    def acc_guard_metric_not_contradicted(candidate, self_auc, self_aupr, rel_auc, rel_aupr):
        if candidate == "self":
            return not (self_auc < rel_auc and self_aupr < rel_aupr)
        return not (rel_auc < self_auc and rel_aupr < self_aupr)

    def choose_gate_anchor(expert_valid, args, valid_n=None):
        self_acc, self_auc, self_aupr = expert_valid["self"]
        rel_acc, rel_auc, rel_aupr = expert_valid["relation"]
        decisions = list()

        def adaptive_gap(metric_name, base_gap, self_value, rel_value):
            if not args.gate_anchor_adaptive_gap or not valid_n:
                return base_gap
            pooled = min(max((self_value + rel_value) / 2.0, 1e-6), 1.0 - 1e-6)
            scale = {
                "aupr": args.gate_anchor_aupr_se_scale,
                "auc": args.gate_anchor_auc_se_scale,
                "acc": args.gate_anchor_acc_se_scale,
            }.get(metric_name, args.gate_anchor_gap_se_scale)
            uncertainty_gap = scale * math.sqrt(pooled * (1.0 - pooled) / max(1, valid_n))
            return max(base_gap, uncertainty_gap)

        metric_specs = [
            ("aupr", self_aupr, rel_aupr, args.gate_anchor_aupr_gap),
            ("auc", self_auc, rel_auc, args.gate_anchor_auc_gap),
            ("acc", self_acc, rel_acc, args.gate_anchor_acc_gap),
        ]
        anchor_name = None
        selected_by = None
        for name, self_value, rel_value, gap in metric_specs:
            diff = self_value - rel_value
            used_gap = adaptive_gap(name, gap, self_value, rel_value)
            winner = "tie"
            if diff > used_gap:
                winner = "self"
            elif -diff > used_gap:
                winner = "relation"
            decisions.append({
                "metric": name,
                "self": self_value,
                "relation": rel_value,
                "diff": diff,
                "gap": used_gap,
                "base_gap": gap,
                "winner": winner,
            })
            if anchor_name is None and winner != "tie":
                anchor_name = winner
                selected_by = name

        if anchor_name is None:
            anchor_name = args.gate_anchor_default
            if anchor_name not in ("self", "relation"):
                anchor_name = "self" if self_aupr >= rel_aupr else "relation"
                selected_by = "aupr_tiebreak"
            else:
                selected_by = "default_tiebreak"
        acc_diff = self_acc - rel_acc
        anchor_acc_gap = args.gate_anchor_acc_guard_gap
        anchor_acc_guard_metric_gap = 0.0
        if args.gate_anchor_acc_guard and valid_n:
            anchor_acc_gap = max(
                args.gate_anchor_acc_guard_gap,
                args.gate_anchor_acc_guard_se_scale / math.sqrt(max(1, valid_n)),
            )
            anchor_acc_guard_metric_gap = args.gate_anchor_acc_guard_metric_scale * anchor_acc_gap
            if anchor_name == "self" and acc_diff < -anchor_acc_gap:
                aupr_adv = self_aupr - rel_aupr
                auc_adv = self_auc - rel_auc
                if (
                    max(aupr_adv, auc_adv) < anchor_acc_guard_metric_gap and
                    acc_guard_metric_not_contradicted("relation", self_auc, self_aupr, rel_auc, rel_aupr)
                ):
                    anchor_name = "relation"
                    selected_by = "acc_guard"
            elif anchor_name == "relation" and acc_diff > anchor_acc_gap:
                aupr_adv = rel_aupr - self_aupr
                auc_adv = rel_auc - self_auc
                if (
                    max(aupr_adv, auc_adv) < anchor_acc_guard_metric_gap and
                    acc_guard_metric_not_contradicted("self", self_auc, self_aupr, rel_auc, rel_aupr)
                ):
                    anchor_name = "self"
                    selected_by = "acc_guard"

        saturation_guard = False
        saturation_acc_gap = args.gate_saturation_acc_gap
        if args.gate_saturation_guard:
            if (
                anchor_name == "self" and
                metric_saturated(self_aupr, self_auc, args) and
                self_acc < rel_acc - saturation_acc_gap
            ):
                anchor_name = "relation"
                selected_by = "saturation_acc_guard"
                saturation_guard = True
            elif (
                anchor_name == "relation" and
                metric_saturated(rel_aupr, rel_auc, args) and
                rel_acc < self_acc - saturation_acc_gap
            ):
                anchor_name = "self"
                selected_by = "saturation_acc_guard"
                saturation_guard = True

        return anchor_name, {
            "selected_by": selected_by,
            "decisions": decisions,
            "valid_n": valid_n,
            "adaptive_gap": args.gate_anchor_adaptive_gap,
            "acc_diff": acc_diff,
            "acc_guard_gap": anchor_acc_gap,
            "acc_guard_metric_gap": anchor_acc_guard_metric_gap,
            "acc_guard_self_metric_ok": acc_guard_metric_not_contradicted("self", self_auc, self_aupr, rel_auc, rel_aupr),
            "acc_guard_relation_metric_ok": acc_guard_metric_not_contradicted("relation", self_auc, self_aupr, rel_auc, rel_aupr),
            "saturation_guard": saturation_guard,
            "saturation_level": args.gate_saturation_level,
            "saturation_acc_gap": saturation_acc_gap,
        }

    def choose_gate_fallback_expert(expert_valid, args, valid_n=None):
        self_acc, self_auc, self_aupr = expert_valid["self"]
        rel_acc, rel_auc, rel_aupr = expert_valid["relation"]
        diff = self_aupr - rel_aupr
        base_gap = args.gate_fallback_expert_aupr_gap
        if args.gate_fallback_expert_adaptive_gap and valid_n:
            pooled = min(max((self_aupr + rel_aupr) / 2.0, 1e-6), 1.0 - 1e-6)
            adaptive_gap = args.gate_fallback_expert_se_scale * math.sqrt(
                pooled * (1.0 - pooled) / max(1, valid_n)
            )
            used_gap = max(base_gap, adaptive_gap)
        else:
            used_gap = base_gap
        if diff > used_gap:
            selected = "self"
            selected_by = "aupr"
        elif -diff > used_gap:
            selected = "relation"
            selected_by = "aupr"
        else:
            if diff > 0:
                selected = "self"
                selected_by = "aupr_soft_tiebreak"
            elif diff < 0:
                selected = "relation"
                selected_by = "aupr_soft_tiebreak"
            else:
                selected = args.gate_fallback_expert_default
                selected_by = "default_tiebreak"
        if selected not in ("self", "relation"):
            selected = "self" if self_aupr >= rel_aupr else "relation"
            selected_by = "aupr_tiebreak"
        acc_diff = self_acc - rel_acc
        if args.gate_fallback_expert_acc_guard and valid_n:
            acc_gap = max(
                args.gate_fallback_expert_acc_gap,
                args.gate_fallback_expert_acc_se_scale / math.sqrt(max(1, valid_n)),
            )
            aupr_override_gap = max(
                used_gap,
                args.gate_fallback_expert_acc_guard_aupr_scale * acc_gap,
            )
            allow_acc_tiebreak = abs(diff) <= args.gate_fallback_expert_acc_tiebreak_aupr_tol
            if (
                selected == "self" and
                acc_diff < -acc_gap and
                allow_acc_tiebreak and
                diff < aupr_override_gap and
                acc_guard_metric_not_contradicted("relation", self_auc, self_aupr, rel_auc, rel_aupr)
            ):
                selected = "relation"
                selected_by = "acc_guard"
            elif (
                selected == "relation" and
                acc_diff > acc_gap and
                allow_acc_tiebreak and
                -diff < aupr_override_gap and
                acc_guard_metric_not_contradicted("self", self_auc, self_aupr, rel_auc, rel_aupr)
            ):
                selected = "self"
                selected_by = "acc_guard"
        else:
            acc_gap = args.gate_fallback_expert_acc_gap
            aupr_override_gap = used_gap
            allow_acc_tiebreak = False
        saturation_guard = False
        saturation_acc_gap = args.gate_saturation_acc_gap
        if args.gate_saturation_guard:
            if (
                selected == "self" and
                metric_saturated(self_aupr, self_auc, args) and
                self_acc < rel_acc - saturation_acc_gap
            ):
                selected = "relation"
                selected_by = "saturation_acc_guard"
                saturation_guard = True
            elif (
                selected == "relation" and
                metric_saturated(rel_aupr, rel_auc, args) and
                rel_acc < self_acc - saturation_acc_gap
            ):
                selected = "self"
                selected_by = "saturation_acc_guard"
                saturation_guard = True
        return selected, {
            "selected_by": selected_by,
            "diff": diff,
            "gap": used_gap,
            "acc_diff": acc_diff,
            "acc_gap": acc_gap,
            "acc_guard_aupr_gap": aupr_override_gap,
            "acc_tiebreak_aupr_tol": args.gate_fallback_expert_acc_tiebreak_aupr_tol,
            "acc_tiebreak_allowed": allow_acc_tiebreak,
            "self_aupr": self_aupr,
            "rel_aupr": rel_aupr,
            "self_acc": self_acc,
            "rel_acc": rel_acc,
            "self_metric_ok": acc_guard_metric_not_contradicted("self", self_auc, self_aupr, rel_auc, rel_aupr),
            "relation_metric_ok": acc_guard_metric_not_contradicted("relation", self_auc, self_aupr, rel_auc, rel_aupr),
            "valid_n": valid_n,
            "saturation_guard": saturation_guard,
            "saturation_level": args.gate_saturation_level,
            "saturation_acc_gap": saturation_acc_gap,
        }

    def compute_binary_metrics(labels, probs, threshold=0.5):
        tp = fp = tn = fn = 0
        for label, prob in zip(labels, probs):
            pred = 1 if prob >= threshold else 0
            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 0 and pred == 0:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        acc = (tp + tn) / len(labels) if labels else 0.0
        return {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "acc": acc,
        }

    def triples_to_names(triples, id2entity, id2relation):
        return [(id2entity[h], id2relation[r], id2entity[t]) for h, r, t in triples]

    def write_json(obj, filepath):
        with open(filepath, "w", encoding="utf-8") as fout:
            json.dump(obj, fout, indent=2)

    def save_prediction_diagnostics(stage, split, model, dataloader, triples, labels, args, id2entity, id2relation):
        probs = model.evaluate(dataloader, args, if_metric=False)
        labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        diag_dir = ensure_diag_dir(args)
        prefix = f"{start_time}__{stage}__{split}"
        csv_path = os.path.join(diag_dir, f"{prefix}.csv")
        named_triples = triples_to_names(triples, id2entity, id2relation)
        with open(csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["idx", "head_id", "relation_id", "tail_id", "head", "relation", "tail", "label", "prob_pos"])
            for idx, ((h, r, t), (head, rel, tail), label, prob) in enumerate(zip(triples, named_triples, labels, probs)):
                writer.writerow([idx, h, r, t, head, rel, tail, label, prob])

        summary = {
            "stage": stage,
            "split": split,
            "dataset": args.dataset,
            "num_samples": len(labels),
            "num_pos": int(sum(labels)),
            "num_neg": int(len(labels) - sum(labels)),
            "mean_prob_pos": float(np.mean(probs)) if probs else 0.0,
            "mean_prob_true_pos": float(np.mean([p for p, y in zip(probs, labels) if y == 1])) if any(y == 1 for y in labels) else 0.0,
            "mean_prob_true_neg": float(np.mean([p for p, y in zip(probs, labels) if y == 0])) if any(y == 0 for y in labels) else 0.0,
            "confusion_0_5": compute_binary_metrics(labels, probs, 0.5),
        }
        if len(set(labels)) > 1:
            summary["auc"] = float(roc_auc_score(labels, probs))
            summary["aupr"] = float(average_precision_score(labels, probs))
        json_path = os.path.join(diag_dir, f"{prefix}.summary.json")
        write_json(summary, json_path)
        print(f"[diagnostics] {stage} {split}: AUC={summary.get('auc', 0):.4f}, AUPR={summary.get('aupr', 0):.4f}, "
              f"FP={summary['confusion_0_5']['fp']}, FN={summary['confusion_0_5']['fn']}")
        print(f"[diagnostics] saved {csv_path}")
        print(f"[diagnostics] saved {json_path}")
        return summary

    def save_pseudo_label_diagnostics(stage, triple_preds, selected_pos, selected_neg, args, id2entity, id2relation):
        diag_dir = ensure_diag_dir(args)
        prefix = f"{start_time}__{stage}"
        summary = {
            "stage": stage,
            "dataset": args.dataset,
            "candidate_count": len(triple_preds) if triple_preds else 0,
            "selected_pos_count": len(selected_pos),
            "selected_neg_count": len(selected_neg),
            "selected_pos_score_min": float(min([s for _, s in selected_pos])) if selected_pos else None,
            "selected_pos_score_max": float(max([s for _, s in selected_pos])) if selected_pos else None,
            "selected_pos_score_mean": float(np.mean([s for _, s in selected_pos])) if selected_pos else None,
            "selected_neg_score_min": float(min([s for _, s in selected_neg])) if selected_neg else None,
            "selected_neg_score_max": float(max([s for _, s in selected_neg])) if selected_neg else None,
            "selected_neg_score_mean": float(np.mean([s for _, s in selected_neg])) if selected_neg else None,
        }
        summary_path = os.path.join(diag_dir, f"{prefix}.pseudo.summary.json")
        write_json(summary, summary_path)
        csv_path = os.path.join(diag_dir, f"{prefix}.pseudo.selected.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["kind", "rank", "head_id", "relation_id", "tail_id", "head", "relation", "tail", "score"])
            for kind, samples in (("pos", selected_pos), ("neg", selected_neg)):
                for rank, ((h, r, t), score) in enumerate(samples):
                    head, rel, tail = triples_to_names([(h, r, t)], id2entity, id2relation)[0]
                    writer.writerow([kind, rank, h, r, t, head, rel, tail, score])
        print(f"[diagnostics] pseudo labels saved {csv_path}")

    def pick_with_diversity(candidates, count, head_limit, tail_limit, used_triples=None, head_counts=None, tail_counts=None):
        picked = []
        if used_triples is None:
            used_triples = set()
        if head_counts is None:
            head_counts = defaultdict(int)
        if tail_counts is None:
            tail_counts = defaultdict(int)
        for item in candidates:
            tri = item[0]
            head, _, tail = tri
            if tri in used_triples:
                continue
            if head_counts[head] >= head_limit or tail_counts[tail] >= tail_limit:
                continue
            picked.append(item)
            used_triples.add(tri)
            head_counts[head] += 1
            tail_counts[tail] += 1
            if len(picked) >= count:
                break
        if len(picked) < count:
            for item in candidates:
                tri = item[0]
                if tri in used_triples:
                    continue
                picked.append(item)
                used_triples.add(tri)
                if len(picked) >= count:
                    break
        return picked

    def select_reliable_relation_to_self(triple_preds, pos_count, neg_count, args):
        head_limit = args.pseudo_head_limit
        tail_limit = args.pseudo_tail_limit
        pos_candidates = [(tri, score, score) for tri, score in triple_preds if score >= args.r2s_pos_threshold]
        neg_candidates = [(tri, score, 1 - score) for tri, score in reversed(triple_preds) if score <= args.r2s_neg_threshold]
        if len(pos_candidates) < pos_count:
            pos_candidates = [(tri, score, score) for tri, score in triple_preds]
        if len(neg_candidates) < neg_count:
            neg_candidates = [(tri, score, 1 - score) for tri, score in reversed(triple_preds)]
        selected_pos = pick_with_diversity(pos_candidates, pos_count, head_limit, tail_limit)
        selected_neg = pick_with_diversity(neg_candidates, neg_count, head_limit, tail_limit)
        print(f"[pseudo] relation->self reliable selection: pos={len(selected_pos)}/{pos_count}, neg={len(selected_neg)}/{neg_count}, "
              f"pos_threshold={args.r2s_pos_threshold}, neg_threshold={args.r2s_neg_threshold}")
        return [(tri, score) for tri, score, _ in selected_pos], [(tri, score) for tri, score, _ in selected_neg]

    def score_triples_with_relation_model(rel_model, triples, args, ent_id2smiles, ent_id2seq):
        dataloader = DataLoader(RelationScoreDataset(triples),
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=max(1, args.cpu_num // 100),
                                collate_fn=RelationScoreDataset.collate_fn)
        return rel_model.evaluate(dataloader, args, if_metric=False)

    def select_hybrid_self_to_relation(triple_preds, rel_model, pos_count, neg_count, args, ent_id2smiles, ent_id2seq):
        triples = [tri for tri, _ in triple_preds]
        relation_scores = score_triples_with_relation_model(rel_model, triples, args, ent_id2smiles, ent_id2seq)
        items = [(tri, self_score, rel_score) for (tri, self_score), rel_score in zip(triple_preds, relation_scores)]

        agree_pos_count = int(round(pos_count * args.hybrid_agreement_ratio))
        explore_pos_count = pos_count - agree_pos_count
        agree_neg_count = int(round(neg_count * args.hybrid_agreement_ratio))
        explore_neg_count = neg_count - agree_neg_count

        head_limit = args.pseudo_head_limit
        tail_limit = args.pseudo_tail_limit
        agree_pos = sorted(((tri, ss, ss * rs) for tri, ss, rs in items), key=lambda x: x[2], reverse=True)
        explore_pos = sorted(((tri, ss, ss * (1 - rs)) for tri, ss, rs in items), key=lambda x: x[2], reverse=True)
        agree_neg = sorted(((tri, ss, (1 - ss) * (1 - rs)) for tri, ss, rs in items), key=lambda x: x[2], reverse=True)
        explore_neg = sorted(((tri, ss, (1 - ss) * rs) for tri, ss, rs in items), key=lambda x: x[2], reverse=True)

        used_pos = set()
        pos_head_counts = defaultdict(int)
        pos_tail_counts = defaultdict(int)
        selected_agree_pos = pick_with_diversity(agree_pos, agree_pos_count, head_limit, tail_limit,
                                                 used_pos, pos_head_counts, pos_tail_counts)
        selected_explore_pos = pick_with_diversity(explore_pos, explore_pos_count, head_limit, tail_limit,
                                                   used_pos, pos_head_counts, pos_tail_counts)

        used_neg = set()
        neg_head_counts = defaultdict(int)
        neg_tail_counts = defaultdict(int)
        selected_agree_neg = pick_with_diversity(agree_neg, agree_neg_count, head_limit, tail_limit,
                                                 used_neg, neg_head_counts, neg_tail_counts)
        selected_explore_neg = pick_with_diversity(explore_neg, explore_neg_count, head_limit, tail_limit,
                                                   used_neg, neg_head_counts, neg_tail_counts)

        selected_pos = [(tri, ss) for tri, ss, _ in selected_agree_pos + selected_explore_pos]
        selected_neg = [(tri, ss) for tri, ss, _ in selected_agree_neg + selected_explore_neg]
        pos_weights = [args.agreement_pseudo_weight] * len(selected_agree_pos) + [args.exploration_pseudo_weight] * len(selected_explore_pos)
        neg_weights = [args.agreement_pseudo_weight] * len(selected_agree_neg) + [args.exploration_pseudo_weight] * len(selected_explore_neg)
        print(f"[pseudo] self->relation hybrid selection: pos agreement/explore={len(selected_agree_pos)}/{len(selected_explore_pos)}, "
              f"neg agreement/explore={len(selected_agree_neg)}/{len(selected_explore_neg)}, ratio={args.hybrid_agreement_ratio}")
        return selected_pos, selected_neg, pos_weights, neg_weights

    def select_distill_self_to_relation(triple_preds, rel_model, args, ent_id2smiles, ent_id2seq):
        triples = [tri for tri, _ in triple_preds]
        relation_scores = score_triples_with_relation_model(rel_model, triples, args, ent_id2smiles, ent_id2seq)
        items = [(tri, float(self_score), float(rel_score))
                 for (tri, self_score), rel_score in zip(triple_preds, relation_scores)]

        pos_count = max(0, int(len(triple_preds) * args.s2r_distill_pos_ratio))
        neg_count = max(0, int(len(triple_preds) * args.s2r_distill_neg_ratio))
        agree_count = max(0, int(len(triple_preds) * args.s2r_distill_agreement_ratio))
        head_limit = args.pseudo_head_limit
        tail_limit = args.pseudo_tail_limit

        useful_pos = sorted(
            ((tri, ss, ss * (1.0 - abs(rs - 0.5) * 2.0)) for tri, ss, rs in items
             if ss >= args.s2r_distill_pos_threshold and abs(rs - 0.5) <= args.s2r_relation_uncertain_width),
            key=lambda x: x[2], reverse=True)
        useful_neg = sorted(
            ((tri, ss, (1.0 - ss) * (1.0 - abs(rs - 0.5) * 2.0)) for tri, ss, rs in items
             if ss <= args.s2r_distill_neg_threshold and abs(rs - 0.5) <= args.s2r_relation_uncertain_width),
            key=lambda x: x[2], reverse=True)
        agree = sorted(
            ((tri, ss, max(ss * rs, (1.0 - ss) * (1.0 - rs))) for tri, ss, rs in items
             if (ss >= args.s2r_distill_pos_threshold and rs >= args.s2r_distill_pos_threshold) or
                (ss <= args.s2r_distill_neg_threshold and rs <= args.s2r_distill_neg_threshold)),
            key=lambda x: x[2], reverse=True)

        soft_head_counts = defaultdict(int)
        soft_tail_counts = defaultdict(int)
        soft_used = set()
        selected_pos = pick_with_diversity(useful_pos, pos_count, head_limit, tail_limit,
                                           soft_used, soft_head_counts, soft_tail_counts)
        selected_neg = pick_with_diversity(useful_neg, neg_count, head_limit, tail_limit,
                                           soft_used, soft_head_counts, soft_tail_counts)
        selected_agree = pick_with_diversity(agree, agree_count, head_limit, tail_limit,
                                             soft_used, soft_head_counts, soft_tail_counts)
        selected_soft = selected_pos + selected_neg + selected_agree

        pos_preview = [(tri, ss) for tri, ss, _ in selected_pos + selected_agree if ss >= 0.5]
        neg_preview = [(tri, ss) for tri, ss, _ in selected_neg + selected_agree if ss < 0.5]
        print(f"[distill] self->relation selected: useful_pos={len(selected_pos)}/{pos_count}, "
              f"useful_neg={len(selected_neg)}/{neg_count}, agreement={len(selected_agree)}/{agree_count}, "
              f"soft_weight={args.s2r_distill_weight}")
        return [(tri, -1, ss) for tri, ss, _ in selected_soft], pos_preview, neg_preview
    class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""

        def __init__(self, directory, save, save_thre, save_mul, fun, patience=15,
                     ):
            """

            """
            self.patience = patience  #15
            self.counter = 0
            self.best_score = float('-inf')
            self.early_stop = False
            self.directory = directory
            self.best_model_path = None
            self.trace_fun = print
            self.save = save
            self.save_thre = save_thre
            self.save_mul = save_mul
            self.fun = fun

        def __call__(self, score, model):
            if (1 - score) >= (1 - self.best_score) * 0.97:
                if_opt = False
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.trace_fun("Training early stopped!")
            else:
                if_opt = True
                self.best_score = score
                self.counter = 0
                self.trace_fun("*****************new opt*******************")
                if self.save and score>self.save_thre:
                    if self.save_mul:
                        save2file(model, self.directory, start_time, info=self.fun + "_" + str(score))
                    else:
                        save2file(model, self.directory, start_time, info=self.fun)
                    self.trace_fun("model saved")


            return self.early_stop, if_opt



    def load_model_to_device(model_path, device, load_kge_model):
        print("load model from disk...")
        kwargs, state = torch.load(f"{model_path}/../{load_kge_model}", map_location=device)
        model = KGEModel(**kwargs)
        if "entity_embedding" not in state:
            raise ValueError("The KG checkpoint does not contain entity_embedding.")
        if state["entity_embedding"].shape != model.entity_embedding.shape:
            raise ValueError(
                "KG entity_embedding shape mismatch: "
                f"checkpoint={tuple(state['entity_embedding'].shape)}, "
                f"expected={tuple(model.entity_embedding.shape)}"
            )
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                "The KG checkpoint is missing required model state: "
                + ", ".join(incompatible.missing_keys)
            )
        return model.to(device)

    def load_pretrained_self_embeddings(args):
        drug_path = args.self_drug_embed_path
        protein_path = args.self_protein_embed_path
        if not drug_path:
            drug_path = os.path.join(args.data_path, "../pretrained_self_drug_embed.pt")
        if not protein_path:
            protein_path = os.path.join(args.data_path, "../pretrained_self_protein_embed.pt")
        drug_path = os.path.normpath(drug_path)
        protein_path = os.path.normpath(protein_path)
        drug_raw = torch.load(drug_path, map_location="cpu")
        protein_raw = torch.load(protein_path, map_location="cpu")

        def normalize(raw):
            if isinstance(raw, dict) and "embeddings" in raw:
                raw = raw["embeddings"]
            return {int(k): torch.as_tensor(v, dtype=torch.float32) for k, v in raw.items()}

        drug_embeds = normalize(drug_raw)
        protein_embeds = normalize(protein_raw)
        drug_dim = next(iter(drug_embeds.values())).numel()
        protein_dim = next(iter(protein_embeds.values())).numel()
        print(f"[self-embed] loaded drug embeddings {len(drug_embeds)} from {drug_path}, dim={drug_dim}")
        print(f"[self-embed] loaded protein embeddings {len(protein_embeds)} from {protein_path}, dim={protein_dim}")
        return drug_embeds, protein_embeds, drug_dim, protein_dim

    class RelationScoreDataset(torch.utils.data.Dataset):
        def __init__(self, pos_triples, neg_triples=None, sample_weights=None):
            if neg_triples is None:
                neg_triples = list()
            self.samples = list()
            total_len = len(pos_triples) + len(neg_triples)
            if sample_weights is None:
                sample_weights = [1.0] * total_len
            if len(sample_weights) != total_len:
                raise ValueError("sample_weights length must match pos_triples + neg_triples")
            for i, tri in enumerate(pos_triples + neg_triples):
                label = 1 if i < len(pos_triples) else 0
                self.samples.append((tri, label, sample_weights[i]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            tri, label, weight = self.samples[idx]
            head, _, tail = tri
            return 0, 0, label, head, tail, weight

        @staticmethod
        def collate_fn(batch):
            labels = torch.tensor([_[2] for _ in batch])
            heads = torch.tensor([_[3] for _ in batch])
            tails = torch.tensor([_[4] for _ in batch])
            weights = torch.tensor([_[5] for _ in batch], dtype=torch.float32)
            return None, None, labels, heads, tails, weights









    def gen_psd_label(fun, args, model=None):

        def gen_cartesian_triples(relation2id, ent_id2seq, ent_id2smiles, cartesian_ratio):
            cmp_with_smiles, gene_with_seq = tuple(ent_id2smiles.keys()), tuple(ent_id2seq.keys())
            cmp_gene_cartesian_product = list(itertools.product(cmp_with_smiles, (relation2id["DTI"],), gene_with_seq))
            print(len(cmp_gene_cartesian_product))
            print(cmp_gene_cartesian_product[0:5])
            cmp_gene_cartesian_product = random.sample(cmp_gene_cartesian_product,
                                                       int(len(cmp_gene_cartesian_product) * cartesian_ratio))
            return cmp_gene_cartesian_product

        if fun == "psd_label_kge":
            psd_ent_id2smiles = ent_id2smiles
            psd_ent_id2seq = ent_id2seq
            psd_dataset_cls = SelfDataset
        else:
            psd_self_ent_id2smiles, psd_self_ent_id2seq, _, _ = load_pretrained_self_embeddings(args)
            psd_ent_id2smiles = {ent_id: feat for ent_id, feat in ent_id2smiles.items()
                                 if ent_id in psd_self_ent_id2smiles}
            psd_ent_id2seq = {ent_id: feat for ent_id, feat in ent_id2seq.items()
                              if ent_id in psd_self_ent_id2seq}
            psd_self_ent_id2smiles = {ent_id: psd_self_ent_id2smiles[ent_id]
                                      for ent_id in psd_ent_id2smiles}
            psd_self_ent_id2seq = {ent_id: psd_self_ent_id2seq[ent_id]
                                   for ent_id in psd_ent_id2seq}
            psd_dataset_cls = HybridSelfDataset
            print(f"[hybrid-self] pseudo candidates use feature intersection: "
                  f"drugs={len(psd_ent_id2smiles)}, proteins={len(psd_ent_id2seq)}")

        cartesian_ratio = args.kge_cartesian_ratio if fun == "psd_label_kge" else args.self_cartesian_ratio
        cartesian_triples = gen_cartesian_triples(relation2id, psd_ent_id2seq, psd_ent_id2smiles, cartesian_ratio)

        if args.filter_protected_pseudo_candidates:
            train_pos = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id, relation2id)
            train_neg = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id, relation2id)
            valid_pos = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
            valid_neg = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)
            test_pos = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
            test_neg = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)
            cartesian_triples, filter_stats = filter_protected_candidates(
                cartesian_triples,
                args.dataset,
                train_pos,
                train_neg,
                valid_pos,
                valid_neg,
                test_pos,
                test_neg,
            )
            print("[leakage-guard] protected candidate filtering: {}".format(filter_stats))

        if psd_dataset_cls is HybridSelfDataset:
            psd_dataset = psd_dataset_cls(cartesian_triples, list(), psd_ent_id2smiles, psd_ent_id2seq,
                                          psd_self_ent_id2smiles, psd_self_ent_id2seq)
        else:
            psd_dataset = psd_dataset_cls(cartesian_triples, list(), psd_ent_id2smiles, psd_ent_id2seq)

        test_cartesian_dataloader = DataLoader(
            psd_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=max(1, args.cpu_num // 100),
            collate_fn=psd_dataset_cls.collate_fn,
        )
        preds = model.evaluate(test_cartesian_dataloader, args, if_metric=False)

        print("End evaluate~~~~~~~~~~~")
        triple_preds = [(cartesian_triples[i], preds[i]) for i in range(len(preds))]
        triple_preds.sort(key=lambda x: x[1], reverse=True)
        if args.save:
            save2file(triple_preds, args.pkl_path, start_time, info=fun)
        return triple_preds



    def train_module(fun, args, triple_preds=None, kge_model=None, self_model=None, rel_model=None, max_ephs=999999):
        def read_psd_triples(true_triple_len, psd_use_ratio, balance_psd, triple_preds, balance_ratio):
            psd_use_num = int(len(triple_preds) * psd_use_ratio)
            selected_pos = list(triple_preds[i] for i in range(psd_use_num))
            psd_triples = list(tri_pred[0] for tri_pred in selected_pos)
            if balance_psd:
                balance_times = psd_use_num / true_triple_len
                if balance_times < 1:
                    balance_times = 1
                    print("psd triples are fewer than real triples")
                else:
                    balance_times = int(balance_times*balance_ratio)
            else:
                balance_times = 1
            selected_neg = list(random.sample(triple_preds[int(-0.5 * len(triple_preds)):],
                                              psd_use_num + (balance_times - 1) * true_triple_len))
            psd_neg_triples = list(tri_pred[0] for tri_pred in selected_neg)
            return psd_triples, psd_neg_triples, balance_times, selected_pos, selected_neg



        train_real_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id,
                                         relation2id)
        train_real_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
                                             relation2id)
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
        valid_neg_triples = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)
        test_neg_triples = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)
        train_weights = None
        distill_samples = None
        selected_pos = list()
        selected_neg = list()

        if fun in ("train_self", "train_relation"):



            if args.use_reliable_pseudo and fun == "train_self":
                psd_use_num = int(len(triple_preds) * args.rel_psd_use_ratio)
                selected_pos, selected_neg = select_reliable_relation_to_self(triple_preds, psd_use_num, psd_use_num, args)
                psd_triples = [tri for tri, _ in selected_pos]
                psd_neg_triples = [tri for tri, _ in selected_neg]
                balance_times = max(1, int((psd_use_num / len(train_real_triples)) * args.rel_balance_ratio)) if args.balance_psd else 1
                train_weights = (
                    [args.real_sample_weight] * (len(train_real_triples) * balance_times) +
                    [args.r2s_pseudo_weight] * len(psd_triples) +
                    [args.real_sample_weight] * len(train_real_neg_triples) +
                    [args.r2s_pseudo_weight] * len(psd_neg_triples)
                )
            elif args.s2r_distill and fun == "train_relation" and rel_model is not None:
                selected_distill, selected_pos, selected_neg = select_distill_self_to_relation(
                    triple_preds, rel_model, args, ent_id2smiles, ent_id2seq)
                distill_samples = selected_distill
                psd_triples = list()
                psd_neg_triples = list()
                balance_times = 1
                train_weights = (
                    [args.real_sample_weight] * len(train_real_triples) +
                    [score for _, _, score in selected_distill] +
                    [args.real_sample_weight] * len(train_real_neg_triples)
                )
            elif args.use_reliable_pseudo and fun == "train_relation" and rel_model is not None:
                psd_use_num = int(len(triple_preds) * args.self_psd_use_ratio)
                neg_use_num = psd_use_num
                if args.balance_psd:
                    balance_times_tmp = psd_use_num / len(train_real_triples)
                    if balance_times_tmp < 1:
                        balance_times_tmp = 1
                    else:
                        balance_times_tmp = int(balance_times_tmp * args.self_balance_ratio)
                    neg_use_num = psd_use_num + (balance_times_tmp - 1) * len(train_real_triples)
                selected_pos, selected_neg, pos_weights, neg_weights = select_hybrid_self_to_relation(
                    triple_preds, rel_model, psd_use_num, neg_use_num, args, ent_id2smiles, ent_id2seq)
                psd_triples = [tri for tri, _ in selected_pos]
                psd_neg_triples = [tri for tri, _ in selected_neg]
                balance_times = max(1, int((psd_use_num / len(train_real_triples)) * args.self_balance_ratio)) if args.balance_psd else 1
                train_weights = (
                    [args.real_sample_weight] * (len(train_real_triples) * balance_times) +
                    pos_weights +
                    [args.real_sample_weight] * len(train_real_neg_triples) +
                    neg_weights
                )
            else:
                psd_triples, psd_neg_triples, balance_times, selected_pos, selected_neg = read_psd_triples(len(train_real_triples),
                                                                               args.self_psd_use_ratio if fun=="train_relation" else args.rel_psd_use_ratio,
                                                                               args.balance_psd,
                                                                               triple_preds,
                                                                               args.self_balance_ratio if fun == "train_relation" else args.rel_balance_ratio
                                                                               )
            should_save_module_diag = args.save_diagnostics and (args.diagnostics_all_modules or fun == "train_relation")
            if should_save_module_diag:
                save_pseudo_label_diagnostics(fun, triple_preds, selected_pos, selected_neg, args, id2entity, id2relation)



            train_triples = (train_real_triples * balance_times) + psd_triples
            train_neg_triples = train_real_neg_triples + psd_neg_triples
            if distill_samples is not None:
                train_triples = train_triples + [tri for tri, _, _ in distill_samples]



        elif fun == "train_projector":
            # train_triples = train_real_triples
            # train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id, relation2id)
            # pass

            psd_triples, psd_neg_triples, balance_times, selected_pos, selected_neg = read_psd_triples(
                                                                           len(train_real_triples), args.gate_psd_use_ratio,
                                                                           args.balance_psd, triple_preds, args.gate_balance_ratio)
            should_save_module_diag = args.save_diagnostics and (args.diagnostics_all_modules or fun == "train_relation")
            if should_save_module_diag:
                save_pseudo_label_diagnostics(fun, triple_preds, selected_pos, selected_neg, args, id2entity, id2relation)

            train_triples = (train_real_triples * balance_times) + psd_triples
            train_neg_triples = train_real_neg_triples + psd_neg_triples
            train_weights = (
                [args.real_sample_weight] * (len(train_real_triples) * balance_times) +
                [args.gate_pseudo_sample_weight] * len(psd_triples) +
                [args.real_sample_weight] * len(train_real_neg_triples) +
                [args.gate_pseudo_sample_weight] * len(psd_neg_triples)
            )


        # elif fun in ("train_projector_pure", "train_relation_pure", "train_self_pure"):
            # aug_times = 542 // len(train_real_triples)  # The number 542 has no special meaning. 500, 510, etc are both ok.
            # if aug_times == 0:
            #     train_triples = train_real_triples
            #     train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
            #                                     relation2id)
            # else:
            #     train_triples = train_real_triples * aug_times
            #     train_neg_triples = read_triple(f"{args.data_path}/../full_drugcentral/train_neg.tsv", entity2id, relation2id)
            #     train_neg_triples = train_neg_triples[:len(train_triples)]
        elif fun in ("train_projector_pure", "train_relation_pure", "train_self_pure"):
            train_triples = train_real_triples
            train_neg_triples = train_real_neg_triples
        else:
            raise ValueError

        if fun in ("train_self", "train_relation", "train_projector"):
            if distill_samples is not None:
                audit_samples = [
                    ("soft", triple, score)
                    for triple, _, score in distill_samples
                ]
            else:
                audit_samples = (
                    [("pos", triple, score) for triple, score in selected_pos] +
                    [("neg", triple, score) for triple, score in selected_neg]
                )
            audit_path = os.path.join(
                ensure_diag_dir(args),
                f"{start_time}__{fun}__leakage_guard.csv",
            )
            audit_training_selection(
                dataset=args.dataset,
                stage=fun,
                selected_samples=audit_samples,
                train_pos=train_real_triples,
                train_neg=train_real_neg_triples,
                valid_pos=valid_triples,
                valid_neg=valid_neg_triples,
                test_pos=test_triples,
                test_neg=test_neg_triples,
                id2entity=id2entity,
                id2relation=id2relation,
                output_path=audit_path,
            )

        uses_self_features = fun in (
            "train_self",
            "train_self_pure",
            "train_projector",
            "train_projector_pure",
        )
        dataset_cls = HybridSelfDataset if uses_self_features else SelfDataset

        def make_dataset(pos_triples, neg_triples, sample_weights=None):
            if dataset_cls is HybridSelfDataset:
                return dataset_cls(pos_triples, neg_triples, ent_id2smiles, ent_id2seq,
                                   self_ent_id2smiles, self_ent_id2seq, sample_weights)
            return dataset_cls(pos_triples, neg_triples, ent_id2smiles, ent_id2seq, sample_weights)

        if distill_samples is not None:
            train_dataset = make_dataset(train_triples, train_neg_triples, train_weights)
            base_len = len(train_triples) - len(distill_samples)
            for i in range(base_len, len(train_triples)):
                tri, _, weight = distill_samples[i - base_len]
                train_dataset.samples[i] = (tri, -1, weight)
        else:
            train_dataset = make_dataset(train_triples, train_neg_triples, train_weights)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=max(1, args.cpu_num // 100),
                                      collate_fn=dataset_cls.collate_fn,
                                      )
        valid_dataloader = DataLoader(make_dataset(valid_triples, valid_neg_triples),
                                      batch_size=args.train_batch_size,
                                      shuffle=False,
                                      num_workers=max(1, args.cpu_num // 100),
                                      collate_fn=dataset_cls.collate_fn
                                      )
        test_dataloader = DataLoader(make_dataset(test_triples, test_neg_triples),
                                     batch_size=args.train_batch_size,
                                     shuffle=False,
                                     num_workers=max(1, args.cpu_num // 100),
                                     collate_fn=dataset_cls.collate_fn
                                     )

        print(f"len of train set: {len(train_triples)}")
        print(f"len of train neg set: {len(train_neg_triples)}")
        valid_eval_triples = valid_triples + valid_neg_triples
        test_eval_triples = test_triples + test_neg_triples
        valid_labels = [1] * len(valid_triples) + [0] * len(valid_neg_triples)
        test_labels = [1] * len(test_triples) + [0] * len(test_neg_triples)
        gate_expert_valid = None

        if fun in ("train_relation", "train_relation_pure"):
            if not rel_model:
                model = RelationModel(args.hidden_dim, kge_model.entity_embedding).to(args.device)
            else:
                model = rel_model

            # if fun == "train_relation_pure":
            #     model.freeze_embeddings()
            # else:
            #     model.unfreeze_embeddings()
            model.freeze_embeddings()

        elif fun in ("train_self", "train_self_pure"):
            model = HybridSelfModel(
                args.hidden_dim,
                esm_model,
                args.self_drug_embed_dim,
                args.self_protein_embed_dim,
                args.self_embed_dropout,
            ).to(args.device)
        else:
            rel_model.freeze_embeddings()
            model = GateModel(rel_model, self_model, args.double_layer).to(args.device)
            model.deployment_selection = {"predictor": "gate", "rule": "gate"}
            model.freeze()
            model.zero_grad(set_to_none=True)
            if "train_projector" in fun:
                print("[anchor-gate] legacy_fusion enabled: using old-style Gate fusion "
                      "with complement/local-cap constraints disabled; "
                      "fallback/relation/self logic is unchanged.")
            if "train_projector" in fun:
                self_acc, self_auc, self_aupr, rel_acc, rel_auc, rel_aupr = model.evaluate_experts(valid_dataloader, args)
                gate_expert_valid = {
                    "self": (self_acc, self_auc, self_aupr),
                    "relation": (rel_acc, rel_auc, rel_aupr),
                }
                gate_anchor_name, gate_anchor_stats = choose_gate_anchor(gate_expert_valid, args, valid_n=len(valid_labels))
                anchor_prior_is_self = gate_anchor_name == "self"
                model.set_anchor_mode("safe_free", anchor_prior_is_self=anchor_prior_is_self)
                anchor_decision_text = ", ".join(
                    f"{item['metric']}={item['winner']}"
                    f"(diff={item['diff']:.4f},gap={item['gap']:.4f})"
                    for item in gate_anchor_stats["decisions"]
                )
                print(f"[anchor-gate] anchor_select strategy=hierarchical, "
                      f"selected_by={gate_anchor_stats['selected_by']}, "
                      f"anchor={gate_anchor_name}, decisions=[{anchor_decision_text}], "
                      f"acc_diff={gate_anchor_stats['acc_diff']:.4f}, "
                      f"acc_guard_gap={gate_anchor_stats['acc_guard_gap']:.4f}, "
                      f"acc_guard_metric_gap={gate_anchor_stats['acc_guard_metric_gap']:.4f}, "
                      f"acc_guard_self_metric_ok={gate_anchor_stats['acc_guard_self_metric_ok']}, "
                      f"acc_guard_relation_metric_ok={gate_anchor_stats['acc_guard_relation_metric_ok']}")
                print(f"[anchor-gate] valid experts: self_aupr={self_aupr:.4f}, rel_aupr={rel_aupr:.4f}, "
                      f"gap={abs(self_aupr - rel_aupr):.4f}, gate_type=safe_free, "
                      f"prior={'self' if anchor_prior_is_self else 'relation'}")

        rollback_state = None
        rollback_valid_aupr = None
        rollback_valid_auc = None
        rollback_valid_acc = None
        rollback_test_aupr = None
        should_track_rollback = args.relation_valid_rollback and fun == "train_relation"
        should_save_module_diag = args.save_diagnostics and (args.diagnostics_all_modules or fun == "train_relation")
        if should_save_module_diag:
            before_valid_summary = save_prediction_diagnostics(f"{fun}_before", "valid", model, valid_dataloader,
                                                               valid_eval_triples, valid_labels, args, id2entity, id2relation)
            before_test_summary = save_prediction_diagnostics(f"{fun}_before", "test", model, test_dataloader,
                                                              test_eval_triples, test_labels, args, id2entity, id2relation)
            if should_track_rollback:
                rollback_valid_aupr = before_valid_summary.get("aupr")
                rollback_valid_auc = before_valid_summary.get("auc")
                rollback_valid_acc = before_valid_summary.get("confusion_0_5", {}).get("acc")
                rollback_test_aupr = before_test_summary.get("aupr")
        elif should_track_rollback:
            rollback_valid_acc, rollback_valid_auc, rollback_valid_aupr = model.evaluate(valid_dataloader, args)
            _, _, rollback_test_aupr = model.evaluate(test_dataloader, args)

        if should_track_rollback:
            rollback_state = clone_state_dict(model)
            print(f"[rollback] relation before tune valid_aupr={rollback_valid_aupr:.4f}, test_aupr={rollback_test_aupr:.4f}")


        # if "train_projector" in fun:
        #     learning_rate = 1e-2
        # else:
        #     learning_rate = 1e-3
        if "train_projector" in fun and "full" not in args.data_path:
            weight_decay = args.gate_weight_decay
            learning_rate = args.gate_learning_rate
            # if_save_opt = args.save_opt
        else:
            weight_decay = args.module_weight_decay
            learning_rate = args.module_learning_rate
            # if_save_opt = args.save_opt

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        # if fun not in ["train_projector", "train_projector_pure"]:
        #     if_save = args.save_opt
        # else:
        #     if_save = False
        # if_save = args.save_opt  # DEBUG

        if "train_projector" in fun:
            es_pat = 5
        elif "train_self" in fun:
            es_pat = 5
        else:
            es_pat = 2
        es = EarlyStopping(args.model_path, args.save_opt, args.save_thre, args.save_mul, fun, patience=es_pat)



        opt_test_aupr = float("-inf")
        test_acc, test_auc, test_aupr = None, None, None
        best_gate_valid_aupr = float("-inf")
        best_gate_valid_auc = float("-inf")
        best_gate_valid_acc = float("-inf")
        best_valid_aupr = float("-inf")
        best_valid_state = None
        distill_weight = args.s2r_distill_weight
        distill_best_valid_aupr = rollback_valid_aupr if should_track_rollback and args.s2r_distill else float("-inf")
        distill_best_state = clone_state_dict(model) if should_track_rollback and args.s2r_distill else None
        # acc_save, auc_save, aupr_save = float("-inf"), float("-inf"), float("-inf")
        for i in range(max_ephs):
            if args.s2r_distill and fun == "train_relation" and hasattr(model, "train_distill_epoch"):
                loss = model.train_distill_epoch(train_dataloader, optimizer, args, distill_weight)
            else:
                loss = model.train_epoch(train_dataloader, loss_fn, optimizer, args)
            print(f"Epoch {i} loss: {loss:>7f}")
            acc, auc, aupr = model.evaluate(valid_dataloader, args)
            print(f"valid acc:{acc:.4f}, auc:{auc:.4f}, aupr:{aupr:.4f}")
            if "train_projector" in fun and aupr > best_gate_valid_aupr:
                best_gate_valid_aupr = aupr
                best_gate_valid_auc = auc
                best_gate_valid_acc = acc
            if args.s2r_distill and fun == "train_relation" and should_track_rollback:
                if aupr >= distill_best_valid_aupr + args.s2r_distill_valid_delta:
                    distill_best_valid_aupr = aupr
                    distill_best_state = clone_state_dict(model)
                    distill_weight = min(args.s2r_distill_weight_max, distill_weight * args.s2r_distill_weight_up)
                    print(f"[distill] keep epoch {i}: best_valid_aupr={distill_best_valid_aupr:.4f}, next_weight={distill_weight:.4f}")
                else:
                    if distill_best_state is not None:
                        restore_state_dict(model, distill_best_state)
                    distill_weight = max(args.s2r_distill_weight_min, distill_weight * args.s2r_distill_weight_down)
                    print(f"[distill] rollback epoch {i}: valid_aupr={aupr:.4f}, best_valid_aupr={distill_best_valid_aupr:.4f}, next_weight={distill_weight:.4f}")
            if args.use_reliable_pseudo and args.best_valid_self and "train_self" in fun and aupr > best_valid_aupr:
                best_valid_aupr = aupr
                best_valid_state = clone_state_dict(model)
                print(f"[best-valid] update {fun}: valid_aupr={best_valid_aupr:.4f}")


            if_stop, if_opt = es(aupr, model)
            if (if_opt and "train_projector" in fun) or (if_stop and "train_projector" not in fun):
                test_acc, test_auc, test_aupr = model.evaluate(test_dataloader, args)
                print(f"test acc:{test_acc:.4f}, auc:{test_auc:.4f}, aupr:{test_aupr:.4f}#")
                if test_aupr > opt_test_aupr:
                    opt_test_aupr = test_aupr

            # if if_opt or if_stop:

            if if_stop and ("train_projector" in fun or max_ephs == 999999):
                break

        if args.use_reliable_pseudo and args.best_valid_self and "train_self" in fun and best_valid_state is not None:
            restore_state_dict(model, best_valid_state)
            acc, auc, aupr = model.evaluate(valid_dataloader, args)
            test_acc, test_auc, test_aupr = model.evaluate(test_dataloader, args)
            opt_test_aupr = max(opt_test_aupr, test_aupr)
            print(f"[best-valid] restored {fun}: valid acc:{acc:.4f}, auc:{auc:.4f}, aupr:{aupr:.4f}")
            print(f"[best-valid] restored {fun}: test acc:{test_acc:.4f}, auc:{test_auc:.4f}, aupr:{test_aupr:.4f}#")

        after_valid_aupr = None
        after_valid_auc = None
        after_valid_acc = None
        after_test_aupr = None
        if should_save_module_diag:
            after_valid_summary = save_prediction_diagnostics(f"{fun}_after", "valid", model, valid_dataloader,
                                                              valid_eval_triples, valid_labels, args, id2entity, id2relation)
            after_test_summary = save_prediction_diagnostics(f"{fun}_after", "test", model, test_dataloader,
                                                             test_eval_triples, test_labels, args, id2entity, id2relation)
            after_valid_aupr = after_valid_summary.get("aupr")
            after_valid_auc = after_valid_summary.get("auc")
            after_valid_acc = after_valid_summary.get("confusion_0_5", {}).get("acc")
            after_test_aupr = after_test_summary.get("aupr")
        elif should_track_rollback:
            after_valid_acc, after_valid_auc, after_valid_aupr = model.evaluate(valid_dataloader, args)
            _, _, after_test_aupr = model.evaluate(test_dataloader, args)

        if should_track_rollback:
            tolerance = args.rollback_min_delta
            rollback_valid_n = None
            if should_save_module_diag:
                rollback_valid_n = before_valid_summary.get("num_samples")
            if args.rollback_adaptive_min_delta and rollback_valid_n:
                valid_p = min(max(rollback_valid_aupr if rollback_valid_aupr is not None else 0.5, 1e-6), 1.0 - 1e-6)
                se_delta = args.rollback_valid_se_scale * math.sqrt(valid_p * (1.0 - valid_p) / max(1, rollback_valid_n))
                tolerance = max(tolerance, se_delta)
            keep_tuned = after_valid_aupr is not None and rollback_valid_aupr is not None and after_valid_aupr >= rollback_valid_aupr + tolerance
            robust_reasons = []
            if args.relation_robust_rollback:
                before_valid_auc = before_valid_summary.get("auc") if should_save_module_diag else rollback_valid_auc
                before_valid_acc = before_valid_summary.get("confusion_0_5", {}).get("acc") if should_save_module_diag else rollback_valid_acc
                if before_valid_auc is not None and after_valid_auc is not None:
                    min_auc = before_valid_auc - args.rollback_auc_tolerance
                    if after_valid_auc < min_auc:
                        keep_tuned = False
                        robust_reasons.append(f"auc_drop:{after_valid_auc:.4f}<{min_auc:.4f}")
                if before_valid_acc is not None and after_valid_acc is not None:
                    min_acc = before_valid_acc - args.rollback_acc_tolerance
                    if after_valid_acc < min_acc:
                        keep_tuned = False
                        robust_reasons.append(f"acc_drop:{after_valid_acc:.4f}<{min_acc:.4f}")
                if args.rollback_require_valid_fp_not_worse and should_save_module_diag:
                    before_fp = before_valid_summary.get("confusion_0_5", {}).get("fp")
                    after_fp = after_valid_summary.get("confusion_0_5", {}).get("fp")
                    if before_fp is not None and after_fp is not None and after_fp > before_fp + args.rollback_fp_tolerance:
                        keep_tuned = False
                        robust_reasons.append(f"fp_worse:{after_fp}>{before_fp}+{args.rollback_fp_tolerance}")
            decision = "keep_tuned" if keep_tuned else "rollback_to_before"
            print(f"[rollback] relation after tune valid_aupr={after_valid_aupr:.4f}, test_aupr={after_test_aupr:.4f}")
            if args.rollback_adaptive_min_delta:
                print(f"[rollback] adaptive_min_delta={tolerance:.6f}, valid_n={rollback_valid_n}, "
                      f"se_scale={args.rollback_valid_se_scale:.4f}")
            if args.relation_robust_rollback:
                reason_text = ",".join(robust_reasons) if robust_reasons else "passed"
                print(f"[rollback] robust valid_auc before={before_valid_auc:.4f}, after={after_valid_auc:.4f}, "
                      f"valid_acc before={before_valid_acc:.4f}, after={after_valid_acc:.4f}, reason={reason_text}")
            print(f"[rollback] decision={decision}, min_delta={tolerance:.6f}")
            if not keep_tuned:
                restore_state_dict(model, rollback_state)
                test_acc, test_auc, test_aupr = model.evaluate(test_dataloader, args)
                opt_test_aupr = rollback_test_aupr if rollback_test_aupr is not None else test_aupr
                print(f"[rollback] restored relation before tune. restored test acc:{test_acc:.4f}, auc:{test_auc:.4f}, aupr:{test_aupr:.4f}#")
            else:
                opt_test_aupr = after_test_aupr if after_test_aupr is not None else opt_test_aupr

            if should_save_module_diag:
                save_prediction_diagnostics(f"{fun}_final_{decision}", "valid", model, valid_dataloader,
                                            valid_eval_triples, valid_labels, args, id2entity, id2relation)
                save_prediction_diagnostics(f"{fun}_final_{decision}", "test", model, test_dataloader,
                                            test_eval_triples, test_labels, args, id2entity, id2relation)
                rollback_summary = {
                    "dataset": args.dataset,
                    "stage": fun,
                    "decision": decision,
                    "min_delta": tolerance,
                    "base_min_delta": args.rollback_min_delta,
                    "rollback_adaptive_min_delta": args.rollback_adaptive_min_delta,
                    "rollback_valid_n": rollback_valid_n,
                    "rollback_valid_se_scale": args.rollback_valid_se_scale,
                    "before_valid_aupr": rollback_valid_aupr,
                    "after_valid_aupr": after_valid_aupr,
                    "before_valid_auc": before_valid_summary.get("auc") if should_save_module_diag else rollback_valid_auc,
                    "after_valid_auc": after_valid_auc,
                    "before_valid_acc": before_valid_summary.get("confusion_0_5", {}).get("acc") if should_save_module_diag else rollback_valid_acc,
                    "after_valid_acc": after_valid_acc,
                    "before_test_aupr": rollback_test_aupr,
                    "after_test_aupr": after_test_aupr,
                    "final_is_rollback": not keep_tuned,
                    "relation_robust_rollback": args.relation_robust_rollback,
                    "robust_reasons": robust_reasons,
                    "pseudo_ratio": args.self_psd_use_ratio,
                    "self_balance_ratio": args.self_balance_ratio,
                    "module_learning_rate": args.module_learning_rate,
                    "module_weight_decay": args.module_weight_decay,
                    "use_reliable_pseudo": args.use_reliable_pseudo,
                    "hybrid_agreement_ratio": args.hybrid_agreement_ratio,
                    "agreement_pseudo_weight": args.agreement_pseudo_weight,
                    "exploration_pseudo_weight": args.exploration_pseudo_weight,
                    "r2s_pseudo_weight": args.r2s_pseudo_weight,
                    "s2r_distill": args.s2r_distill,
                    "s2r_distill_weight": args.s2r_distill_weight,
                    "s2r_ranking_weight": args.s2r_ranking_weight,
                    "seed": args.seed,
                }
                summary_path = os.path.join(ensure_diag_dir(args), f"{start_time}__{fun}.rollback.summary.json")
                write_json(rollback_summary, summary_path)
                print(f"[rollback] summary saved {summary_path}")

        gate_fallback_enabled = args.gate_accept_delta is not None or args.gate_fallback_margin >= 0
        if "train_projector" in fun and gate_expert_valid is not None and gate_fallback_enabled:
            best_expert_name, fallback_expert_stats = choose_gate_fallback_expert(
                gate_expert_valid, args, valid_n=len(valid_labels)
            )
            _, gate_anchor_stats = choose_gate_anchor(gate_expert_valid, args, valid_n=len(valid_labels))
            best_expert_acc, best_expert_auc, best_expert_valid = gate_expert_valid[best_expert_name]
            gate_accept_gain = None
            gate_aupr_gain = best_gate_valid_aupr - best_expert_valid
            gate_auc_gain = best_gate_valid_auc - best_expert_auc
            gate_pareto_ok = False
            if args.gate_accept_delta is not None:
                valid_n_for_accept = max(1, len(valid_labels))
                adaptive_accept_delta = max(
                    args.gate_accept_delta,
                    args.gate_accept_se_scale / math.sqrt(valid_n_for_accept)
                )
                gate_auc_ok = best_gate_valid_auc >= best_expert_auc - args.gate_auc_tolerance
                gate_acc_guard_gap = max(
                    args.gate_accept_acc_guard_gap,
                    args.gate_accept_acc_guard_se_scale / math.sqrt(valid_n_for_accept)
                )
                gate_acc_ok = (
                    (not args.gate_accept_acc_guard) or
                    best_gate_valid_acc >= best_expert_acc - gate_acc_guard_gap
                )
                gate_saturation_accept_ok = True
                gate_saturation_noninferior_ok = False
                gate_saturation_ref = max(
                    best_gate_valid_aupr,
                    best_gate_valid_auc,
                    best_expert_valid,
                    best_expert_auc,
                )
                if (
                    args.gate_saturation_guard and
                    metric_saturated(best_gate_valid_aupr, best_gate_valid_auc, args)
                ):
                    gate_saturation_accept_ok = (
                        best_gate_valid_acc > best_expert_acc + args.gate_saturation_acc_gap
                    )
                if args.gate_accept_saturation_noninferior:
                    gate_saturation_noninferior_ok = (
                        gate_saturation_ref >= args.gate_accept_saturation_noninferior_level and
                        best_gate_valid_aupr >= best_expert_valid - args.gate_accept_saturation_aupr_tolerance and
                        best_gate_valid_auc >= best_expert_auc - args.gate_auc_tolerance and
                        gate_acc_ok
                    )
                pareto_expert_gap_threshold = max(
                    args.gate_accept_pareto_min_expert_aupr_gap,
                    args.gate_accept_pareto_expert_gap_scale / math.sqrt(valid_n_for_accept)
                )
                gate_pareto_ok = (
                    args.gate_accept_pareto_if_noninferior and
                    abs(fallback_expert_stats["diff"]) >= pareto_expert_gap_threshold and
                    best_gate_valid_aupr >= best_expert_valid - args.gate_accept_pareto_aupr_tolerance and
                    gate_auc_ok
                )
                if args.gate_accept_metric == "combined":
                    if args.gate_accept_adaptive_weights:
                        gate_aupr_ref = min(max((best_gate_valid_aupr + best_expert_valid) / 2.0, 1e-6), 1.0 - 1e-6)
                        gate_auc_ref = min(max((best_gate_valid_auc + best_expert_auc) / 2.0, 1e-6), 1.0 - 1e-6)
                        aupr_se = math.sqrt(gate_aupr_ref * (1.0 - gate_aupr_ref) / valid_n_for_accept)
                        auc_se = math.sqrt(gate_auc_ref * (1.0 - gate_auc_ref) / valid_n_for_accept)
                        aupr_w = 1.0 / max(aupr_se, 1e-8)
                        auc_w = 1.0 / max(auc_se, 1e-8)
                    else:
                        aupr_w = args.gate_accept_aupr_weight
                        auc_w = args.gate_accept_auc_weight
                    weight_sum = max(1e-8, aupr_w + auc_w)
                    aupr_w = aupr_w / weight_sum
                    auc_w = auc_w / weight_sum
                    gate_accept_gain = aupr_w * gate_aupr_gain + auc_w * gate_auc_gain
                    gate_aupr_ok = gate_accept_gain >= adaptive_accept_delta
                    fallback_rule = "combined_adaptive_accept_delta"
                else:
                    gate_accept_gain = gate_aupr_gain
                    gate_aupr_ok = best_gate_valid_aupr >= best_expert_valid + adaptive_accept_delta
                    fallback_rule = "adaptive_accept_delta"
                use_fallback = not (gate_aupr_ok and gate_auc_ok and gate_acc_ok and gate_saturation_accept_ok)
                if use_fallback and gate_pareto_ok:
                    use_fallback = False
                    fallback_rule = f"{fallback_rule}+pareto_valid_noninferior"
                if use_fallback and gate_saturation_noninferior_ok:
                    use_fallback = False
                    fallback_rule = f"{fallback_rule}+saturation_valid_noninferior"
                if not gate_acc_ok:
                    use_fallback = True
                    fallback_rule = f"{fallback_rule}+acc_guard"
                if not gate_saturation_accept_ok:
                    use_fallback = True
                    fallback_rule = f"{fallback_rule}+saturation_acc_guard"
            else:
                valid_n_for_accept = max(1, len(valid_labels))
                adaptive_accept_delta = None
                gate_auc_ok = True
                gate_acc_guard_gap = None
                gate_acc_ok = True
                gate_saturation_accept_ok = True
                gate_saturation_noninferior_ok = False
                gate_saturation_ref = None
                gate_aupr_ok = best_gate_valid_aupr + args.gate_fallback_margin >= best_expert_valid
                gate_accept_gain = gate_aupr_gain
                gate_pareto_ok = False
                pareto_expert_gap_threshold = None
                use_fallback = best_gate_valid_aupr + args.gate_fallback_margin < best_expert_valid
                fallback_rule = "fallback_margin"
            raw_gate_opt_test_aupr = opt_test_aupr
            if use_fallback:
                test_self_acc, test_self_auc, test_self_aupr, test_rel_acc, test_rel_auc, test_rel_aupr = model.evaluate_experts(test_dataloader, args)
                gate_expert_test = {
                    "self": (test_self_acc, test_self_auc, test_self_aupr),
                    "relation": (test_rel_acc, test_rel_auc, test_rel_aupr),
                }
                test_acc, test_auc, test_aupr = gate_expert_test[best_expert_name]
                opt_test_aupr = test_aupr
            model.deployment_selection = {
                "predictor": best_expert_name if use_fallback else "gate",
                "rule": fallback_rule,
                "anchor_prior_is_self": bool(model.anchor_prior_is_self),
            }
            accept_weight_text = ""
            if args.gate_accept_delta is not None and args.gate_accept_metric == "combined":
                accept_weight_text = (
                    f"accept_weights=({aupr_w:.3f},{auc_w:.3f}), "
                    f"adaptive_weights={args.gate_accept_adaptive_weights}, "
                )
            print("[gate-fallback] "
                  f"best_gate_valid_aupr={best_gate_valid_aupr:.4f}, "
                  f"best_gate_valid_auc={best_gate_valid_auc:.4f}, "
                  f"best_gate_valid_acc={best_gate_valid_acc:.4f}, "
                  f"best_expert={best_expert_name}, best_expert_valid_aupr={best_expert_valid:.4f}, "
                  f"best_expert_valid_auc={best_expert_auc:.4f}, "
                  f"best_expert_valid_acc={best_expert_acc:.4f}, "
                  f"anchor_selected_by={gate_anchor_stats['selected_by']}, "
                  f"fallback_selected_by={fallback_expert_stats['selected_by']}, "
                  f"fallback_aupr_diff={fallback_expert_stats['diff']:.4f}, "
                  f"fallback_aupr_gap={fallback_expert_stats['gap']:.4f}, "
                  f"fallback_acc_diff={fallback_expert_stats['acc_diff']:.4f}, "
                  f"fallback_acc_gap={fallback_expert_stats['acc_gap']:.4f}, "
                  f"fallback_acc_guard_aupr_gap={fallback_expert_stats['acc_guard_aupr_gap']:.4f}, "
                  f"fallback_acc_tiebreak_aupr_tol={fallback_expert_stats['acc_tiebreak_aupr_tol']:.4f}, "
                  f"fallback_acc_tiebreak_allowed={fallback_expert_stats['acc_tiebreak_allowed']}, "
                  f"fallback_self_metric_ok={fallback_expert_stats['self_metric_ok']}, "
                  f"fallback_relation_metric_ok={fallback_expert_stats['relation_metric_ok']}, "
                  f"margin={args.gate_fallback_margin:.4f}, "
                  f"accept_delta={args.gate_accept_delta}, "
                  f"accept_metric={args.gate_accept_metric}, "
                  f"accept_gain={gate_accept_gain:.4f}, "
                  f"aupr_gain={gate_aupr_gain:.4f}, auc_gain={gate_auc_gain:.4f}, "
                  f"{accept_weight_text}"
                  f"adaptive_accept_delta={adaptive_accept_delta}, "
                  f"accept_valid_n={valid_n_for_accept}, "
                  f"accept_se_scale={args.gate_accept_se_scale:.4f}, "
                  f"auc_tolerance={args.gate_auc_tolerance:.4f}, "
                  f"pareto_ok={gate_pareto_ok}, "
                  f"pareto_aupr_tol={args.gate_accept_pareto_aupr_tolerance:.4f}, "
                  f"pareto_expert_gap_threshold={pareto_expert_gap_threshold}, "
                  f"pareto_gap_scale={args.gate_accept_pareto_expert_gap_scale:.4f}, "
                  f"gate_acc_ok={gate_acc_ok}, gate_acc_guard_gap={gate_acc_guard_gap}, "
                  f"gate_saturation_ok={gate_saturation_accept_ok}, "
                  f"gate_saturation_noninferior_ok={gate_saturation_noninferior_ok}, "
                  f"gate_saturation_ref={gate_saturation_ref}, "
                  f"saturation_noninferior_level={args.gate_accept_saturation_noninferior_level:.4f}, "
                  f"saturation_aupr_tol={args.gate_accept_saturation_aupr_tolerance:.4f}, "
                  f"saturation_guard={args.gate_saturation_guard}, "
                  f"saturation_level={args.gate_saturation_level:.4f}, "
                  f"saturation_acc_gap={args.gate_saturation_acc_gap:.4f}, "
                  f"fallback_saturation_guard={fallback_expert_stats['saturation_guard']}, "
                  f"anchor_saturation_guard={gate_anchor_stats['saturation_guard']}, "
                  f"gate_aupr_ok={gate_aupr_ok}, gate_auc_ok={gate_auc_ok}, rule={fallback_rule}, "
                  f"decision={'fallback' if use_fallback else 'gate'}, "
                  f"raw_gate_opt_test_aupr={raw_gate_opt_test_aupr:.4f}, "
                  f"final_test_aupr={opt_test_aupr:.4f}")

        if args.save:
            if "train_projector" in fun:
                save_full_model_checkpoint(model, args, start_time)
            else:
                save2file(model, args.model_path, start_time, info=fun)
        if test_acc is None or test_auc is None or test_aupr is None:
            test_acc, test_auc, test_aupr = model.evaluate(test_dataloader, args)
            opt_test_aupr = max(opt_test_aupr, test_aupr)
            print(f"[final-eval] {fun}: test acc:{test_acc:.4f}, auc:{test_auc:.4f}, aupr:{test_aupr:.4f}#")
        current_time = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
        print(f"End training at time {current_time}")
        print(f"opt test aupr {opt_test_aupr:.4f}")


        return model, test_acc, test_auc, test_aupr


    def set_random_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seeds(args.seed)


    gc.collect()
    torch.cuda.empty_cache()

    if hasattr(args, "device") and args.device is not None:
        args.device = torch.device(f"cuda:{args.device}" if int(args.device) >= 0 else "cpu")
    elif hasattr(args, "device_num") and args.device_num >= 0:
        args.device = torch.device(f"cuda:{args.device_num}")
    else:
        args.device = torch.device("cpu")
    print(f"[config] device={args.device}")

    args.data_path = f"var_data/{args.dataset}"
    args.model_path = f"var_models/{args.dataset}"
    args.pkl_path = f"var_pkls/{args.dataset}"
    os.makedirs(args.pkl_path, exist_ok=True)
    config_path = os.path.join(ensure_diag_dir(args), f"{start_time}__run_config.json")
    serializable_args = {key: str(value) if key == "device" else value for key, value in vars(args).items()}
    write_json(serializable_args, config_path)
    print(f"[diagnostics] run config saved {config_path}")

    print("{args.rel_psd_use_ratio} {args.self_psd_use_ratio} {args.kge_cartesian_ratio} {args.self_cartesian_ratio}")
    print(f"{args.rel_psd_use_ratio} {args.self_psd_use_ratio} {args.kge_cartesian_ratio} {args.self_cartesian_ratio}")


    import esm

    entity2id = dict()
    id2entity = dict()
    with open(os.path.join(f"{args.data_path}/../", 'entities.dict')) as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    relation2id = dict()
    id2relation = dict()
    with open(os.path.join(f"{args.data_path}/../", 'relations.dict')) as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model = esm_model.to(args.device)
    esm_converter = esm_alphabet.get_batch_converter()
    ent_id2seq, ent_id2smiles = prepare_self_esm_data(f"{args.data_path}/../ent_id2seq.csv",
                                                      f"{args.data_path}/../ent_id2smiles.csv", esm_converter)
    self_ent_id2smiles, self_ent_id2seq, drug_dim, protein_dim = load_pretrained_self_embeddings(args)
    args.self_drug_embed_dim = drug_dim
    args.self_protein_embed_dim = protein_dim

    if not args.load_kge_model:
        raise ValueError("Hype-DTI requires --load_kge_model.")
    print("**************************load kge model******************************")
    kge_model = load_model_to_device(args.model_path, args.device, args.load_kge_model)

    print("***********************Start train relation projector******************************")
    rel_model, _, _, _ = train_module(
        "train_relation_pure", args, kge_model=kge_model, max_ephs=args.rel_train_ephs
    )

    print("***********************Start generating psd label kge************************")
    kge_triple_preds = gen_psd_label("psd_label_kge", args, rel_model)

    print("************************Start training self model*****************************")
    self_model, _, _, _ = train_module(
        "train_self", args, kge_triple_preds, max_ephs=args.self_train_ephs
    )

    print("************************Start generating psd label self*****************************")
    self_triple_preds = gen_psd_label("psd_label_self", args, self_model)

    print("***********************Start tune relation projector******************************")
    rel_model, _, _, _ = train_module(
        "train_relation", args, self_triple_preds, rel_model=rel_model, max_ephs=args.rel_tune_ephs
    )

    print("***********************Start train gating model******************************")
    triple_preds = cut_long_preds(kge_triple_preds, self_triple_preds)
    train_module(
        "train_projector",
        args,
        triple_preds=triple_preds,
        rel_model=rel_model,
        self_model=self_model,
        max_ephs=args.gate_train_ephs,
    )

    print("*************************exit*****************************")
# TILL HERE

if __name__ == '__main__':


    start_time = str(datetime.datetime.now())[0:-4].replace(' ', '_').replace(":", "_")

    parser = argparse.ArgumentParser()

    parser = add_full_model_args(parser)
    args = parser.parse_args()





    main(args, start_time)
