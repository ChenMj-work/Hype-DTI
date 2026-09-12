from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os
import random
import sys

from rdkit import Chem
import networkx as nx

from dgl import DGLGraph


import numpy as np
import torch

import dgl

from esm_model import *

def cut_long_preds(seq1, seq2):
    if len(seq1) > len(seq2):
        # seq1 = [seq1[i] for i in range(len(seq2))]
        seq1 = random.sample(seq1, len(seq2))
    else:
        # seq2 = [seq2[i] for i in range(len(seq1))]
        seq2 = random.sample(seq2, len(seq1))

    seq = seq1 + seq2
    seq.sort(key=lambda x: x[1], reverse=True)
    return seq


def read_triple(file_path, entity2id, relation2id):
    """
    Read triples and map them into ids.
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            etth = entity2id[h]
            rlt = relation2id[r]
            ettt = entity2id[t]
            triples.append((etth, rlt, ettt))
    return triples



def prepare_self_esm_data(ent_id2seq_path, ent_id2smiles_path, esm_converter):
    def load_id2seq(filepath):
        data = list()
        long_seq_index = list()
        seq_1023_index = list()
        with open(filepath, "r") as fin:
            csv_rdr = csv.reader(fin)
            for i, line in enumerate(csv_rdr):
                seq = line[1]
                length = len(line[1])
                label = line[0]
                if length > 1022:
                    data.append((label, seq[int(length / 2) - 511:int(length / 2) + 511]))
                    if length == 1023:
                        seq_1023_index.append(i)
                    else:
                        long_seq_index.append(i)
                else:
                    data.append((label, seq))


        _, _, batch_tokens = esm_converter(data)

        # if the sequence has its length exceeding the max length, no start and finish token should be added
        for index in seq_1023_index:
            batch_tokens[index, -1] = 1
        for index in long_seq_index:
            batch_tokens[index, -1] = 1
            batch_tokens[index, 0] = 1

        ent_id2seq = dict()
        for i, (idx, _) in enumerate(data):
            ent_id2seq[int(idx)] = batch_tokens[i].tolist()

        return ent_id2seq



    def load_id2smiles(filepath):
        result = dict()
        with open(filepath, "r") as fin:
            csv_rdr = csv.reader(fin)
            for line in csv_rdr:
                result[int(line[0])] = line[1]
        return result

    def smiles2graph(smile):
        def atom_features(atom):
            def one_of_k_encoding_unk(x, allowable_set):
                """Maps inputs not in the allowable set to the last element."""
                if x not in allowable_set:
                    x = allowable_set[-1]
                return list(map(lambda s: x == s, allowable_set))

            def one_of_k_encoding(x, allowable_set):
                if x not in allowable_set:
                    raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
                return list(map(lambda s: x == s, allowable_set))

            return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                                  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                                                   'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                                                   'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                                                   'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                                                   'Unknown']) +
                            one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                            [atom.GetIsAromatic()])

        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None, None, None
        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return c_size, features, edge_index

    def get_graph_features(c_id, smiles2graph):
        h = list()
        c_size, features, edge_index = smiles2graph[c_id]
        g = DGLGraph()
        g.add_nodes(c_size)
        if edge_index:
            edge_index = np.array(edge_index)
            g.add_edges(edge_index[:, 0], edge_index[:, 1])

        for f in features:
            h.append(f)
        # g.ndata['x'] = torch.from_numpy(np.array(features)).float()  # CHGH
        g = dgl.add_self_loop(g)

        deg = g.in_degrees().float().clamp(min=1)  # in_degree is 入度. clamp is 夹紧
        norm = torch.pow(deg, -0.5)  # for Equ6
        g.ndata['d'] = norm  # set features called "d" for all nodes
        return g, h

    ent_id2seq = load_id2seq(ent_id2seq_path)




    ent_id2smiles = load_id2smiles(ent_id2smiles_path)
    invalid_smiles_drug_ids = list()
    for drug_id, smiles in ent_id2smiles.items():
        c_size, features, edge_index = smiles2graph(smiles)
        if c_size is None and features is None and edge_index is None:
            invalid_smiles_drug_ids.append(
                drug_id)  # the iteration mechanism of python does not support del element in loop
        else:
            ent_id2smiles[drug_id] = (c_size, features, edge_index)

    for drug_id in invalid_smiles_drug_ids:
        ent_id2smiles.pop(drug_id)

    for drug_id in ent_id2smiles:
        smiles_g, smiles_h = get_graph_features(
            drug_id, ent_id2smiles)
        ent_id2smiles[drug_id] = (smiles_g, smiles_h)

    return ent_id2seq, ent_id2smiles



def save2file(obj, save_directory, start_time, info=""):
    file_path = start_time
    for i in range(0, len(sys.argv)):
        file_path += f"__{sys.argv[i]}"

    file_path = (file_path.replace("/", "SLH").replace("__--", "--"))[0:80]
    file_path = f"{save_directory}/{file_path}"
    file_path = file_path + "_" + info
    if isinstance(obj, torch.nn.Module):
        file_path += ".pth"
    else:
        file_path += ".pickle"
    print(file_path)
    with open(file_path, "wb") as fout:
        if isinstance(obj, torch.nn.Module):
            st_dict = obj.state_dict()
            if isinstance(obj, SelfModel) or obj.__class__.__name__ in {"SelfEmbedModel", "HybridSelfModel"}:
                self_states = dict()
                for name, value in st_dict.items():
                    if name.startswith("esm_model.") or ".esm_model." in name:
                        continue
                    self_states[name] = value
                st_dict = self_states
            torch.save((obj.kwargs, st_dict), fout)


        else:
            pickle.dump(obj, fout)

    return file_path


def save_full_model_checkpoint(model, args, start_time):
    """Save both experts, the gate, frozen config, and deployment selection."""
    os.makedirs(args.model_path, exist_ok=True)
    state = {
        name: value
        for name, value in model.state_dict().items()
        if ".esm_model." not in name and not name.startswith("esm_model.")
    }
    config = {
        key: str(value) if key == "device" else value
        for key, value in vars(args).items()
    }
    payload = {
        "format_version": 1,
        "model_name": "Hype-DTI",
        "config": config,
        "relation_kwargs": dict(model.relation_model.kwargs),
        "self_kwargs": dict(model.self_model.kwargs),
        "gate_kwargs": dict(model.kwargs),
        "deployment_selection": getattr(
            model,
            "deployment_selection",
            {"predictor": "gate", "rule": "gate"},
        ),
        "state_dict": state,
    }
    path = os.path.join(
        args.model_path,
        f"{start_time}__{args.dataset}__hype_dti_full.pth",
    )
    torch.save(payload, path)
    print(f"[checkpoint] full model saved {path}")
    return path


def load_full_model_checkpoint(path, esm_model, device):
    """Reconstruct a complete Hype-DTI gate model from a release checkpoint."""
    payload = torch.load(path, map_location=device)
    if payload.get("format_version") != 1 or payload.get("model_name") != "Hype-DTI":
        raise ValueError("Unsupported Hype-DTI checkpoint format.")

    state = payload["state_dict"]
    embedding_key = "relation_model.entity_embeddings"
    if embedding_key not in state:
        raise ValueError(f"Checkpoint is missing {embedding_key}.")

    relation_kwargs = payload["relation_kwargs"]
    relation_model = RelationModel(
        entity_dim=relation_kwargs["entity_dim"],
        entity_embeddings=state[embedding_key].detach().clone(),
    )
    self_kwargs = payload["self_kwargs"]
    if self_kwargs.get("self_encoder") != "hybrid_embed":
        raise ValueError("The release loader expects the hybrid_embed self expert.")
    self_model = HybridSelfModel(
        self_kwargs["hidden_dim"],
        esm_model,
        self_kwargs["drug_embed_dim"],
        self_kwargs["protein_embed_dim"],
        self_kwargs.get("self_embed_dropout", 0.2),
    )
    gate_kwargs = payload["gate_kwargs"]
    model = GateModel(
        relation_model,
        self_model,
        gate_kwargs["if_double_layer"],
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if ".esm_model." not in name]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "Incompatible full-model checkpoint: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.deployment_selection = payload.get(
        "deployment_selection",
        {"predictor": "gate", "rule": "gate"},
    )
    model.set_anchor_mode(
        "safe_free",
        anchor_prior_is_self=model.deployment_selection.get("anchor_prior_is_self"),
    )
    return model.to(device), payload
