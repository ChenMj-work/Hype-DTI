from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as nnfun
from dgl.nn.pytorch import GraphConv
from dgllife.model import MLPNodeReadout

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score




class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.kwargs = {"model_name": model_name, "nentity":nentity, "nrelation":nrelation,
                       "hidden_dim": hidden_dim, "gamma":gamma,
                       "double_entity_embedding": double_entity_embedding,
                       "double_relation_embedding": double_relation_embedding
                       }
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')






    def forward(self, sample, module_name, mode='single'):  # HERE
        """
        train and test dataset can have different neg sample size

        the format of sample can be different in different module and mode
        """
        if mode in ["single", "single-head-batch", "single-tail-batch"]:
            if module_name in ["molecule_train", "protein_train", "fuse_test", "molecule_test"]:
                sample, self_data = sample
            elif module_name == "relation":
                pass
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            head = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            if module_name == "fuse_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head_seq_batch = self.encode_proteins(self_data["head_seq_batch"]).unsqueeze(1)
                tail_seq_batch = self.encode_proteins(self_data["tail_seq_batch"]).unsqueeze(1)

                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))
                head[self_data["if_head_seq_batch"]] = self.protein_mlp(
                    torch.cat((head[self_data["if_head_seq_batch"]], head_seq_batch), dim=2))
                tail[self_data["if_tail_seq_batch"]] = self.protein_mlp(
                    torch.cat((tail[self_data["if_tail_seq_batch"]], tail_seq_batch), dim=2))


            elif module_name == "molecule_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))


            elif module_name == "molecule_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_compounds(self_data).unsqueeze(1)
                    tail = self.molecule_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_compounds(self_data).unsqueeze(1)
                    head = self.molecule_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)


            elif module_name == "protein_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_proteins(self_data).unsqueeze(1)
                    tail = self.protein_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_proteins(self_data).unsqueeze(1)
                    head = self.protein_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)




        elif mode == "head-batch":  # positive entities are tails
            if module_name == "relation":
                tail_part, head_part = sample  # 16 * 3, 16 * 97238


                tail = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                tail_part, head_part, smiles_batch = sample
                tail = self.encode_compounds(smiles_batch).unsqueeze(1)
                tail_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.molecule_mlp(torch.cat((tail, tail_free), 2))

            elif module_name == "protein_train":
                tail_part, head_part, seq_batch = sample
                tail = self.encode_proteins(seq_batch).unsqueeze(1)
                tail_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.protein_mlp(torch.cat((tail, tail_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(  # 16 * 97238 * 1000
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(  # 16 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)


        elif mode == "tail-batch":
            if module_name == "relation":
                head_part, tail_part = sample  # 16 * 3, 16 * 97238

                head = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                head_part, tail_part, smiles_batch = sample
                head = self.encode_compounds(smiles_batch).unsqueeze(1)
                head_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.molecule_mlp(torch.cat((head, head_free), 2))

            elif module_name == "protein_train":
                head_part, tail_part, seq_batch = sample
                head = self.encode_proteins(seq_batch).unsqueeze(1)
                head_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.protein_mlp(torch.cat((head, head_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == "single-head-batch":  # for single mode in self train. entities with self feat are tails.
            sample, self_feat = sample
            head = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            if module_name == "molecule_train":  # TODO
                tail = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                tail = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


        elif mode == "single-tail-batch":
            sample, self_feat = sample
            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            if module_name == "molecule_train":
                head = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                head = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

        else:
            raise ValueError('mode %s not supported' % mode)


        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
            # tail batch
            # 1024 * 1 * 1000
            # 1024 * 1 * 1000
            # 1024 * 256 * 1000
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # gamma - ||h+r-t||
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args):
        if args.negative_adversarial_sampling:  # true. will amplify good negative sample
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (nnfun.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * nnfun.logsigmoid(-negative_score)).sum(dim=1)  # the neg loss is neged
        else:
            negative_score = nnfun.logsigmoid(-negative_score).mean(dim=1)  # not exactly same with the transe paper

        if args.uni_weight:  # false
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:  # false
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        # loss =
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss


    @staticmethod
    def train_relation_step(model, optimizer, train_iterator, args):  # TILL HERE
        """
        A single train step. Apply back-propation and return the loss
        """

        model.train()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator) # it is test that it can iterate as a loop

        positive_sample = positive_sample.to(args.device)
        negative_sample = negative_sample.to(args.device)
        subsampling_weight = subsampling_weight.to(args.device)

        negative_score = model((positive_sample, negative_sample), module_name="relation", mode=mode)

        positive_score = model(positive_sample, module_name="relation", mode="single")  # 1024 * 1

        positive_score = nnfun.logsigmoid(positive_score).squeeze(dim=1)


        return model.kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args)


    @staticmethod
    def evaluate(args, model, test_dataloader, labels, if_metric=True):
        """
        Evaluate the model on test or valid datasets
        """
        def correct_count(labels, scores):
            return torch.isclose(scores, labels, atol=0.5, rtol=0).sum().item()



        model.eval()
        preds = list()
        with torch.no_grad():
            for i, sample in enumerate(test_dataloader):
                sample = sample.to(args.device)

                scores = torch.sigmoid(model(sample, module_name="relation", mode="single"))  # sigmoid(gamma - ||h+r-t||)  # TILL HERE
                scores = scores.cpu().squeeze()
                preds += scores.tolist()
                if i % 10000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = correct_count(labels, torch.tensor(preds)) / len(test_dataloader.dataset)
            auc = roc_auc_score(labels.numpy(), preds)
            aupr = average_precision_score(labels.numpy(), preds)
            return preds, acc, auc, aupr
        else:
            return preds


class SelfModel(nn.Module):
    def __init__(self, hidden_dim, esm_model):
        super().__init__()
        self.kwargs = {"hidden_dim": hidden_dim}
        self.entity_dim = hidden_dim
        self.layer_filters_proteins = [1280, 96, 128, self.entity_dim]
        self.cpi_hidden_dim = [78, self.entity_dim, self.entity_dim]

        self.drug_gcn = nn.ModuleList(   #  CHGH
            [GraphConv(in_feats=self.cpi_hidden_dim[i], out_feats=self.cpi_hidden_dim[i + 1]) for i in
             range(len(self.cpi_hidden_dim) - 1)])
        self.drug_output_layer = MLPNodeReadout(self.entity_dim, self.entity_dim, self.entity_dim,
                                                activation=nn.ReLU(), mode='max')  # 2-layer mlp and readout

        self.target_cnn = nn.ModuleList(
            [nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i + 1],
                       kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins) - 1)])

        self.fc = nn.Sequential(nn.Linear(self.entity_dim*2, self.entity_dim), nn.ReLU(), nn.Linear(self.entity_dim, 2))
        self.esm_model = esm_model
        for para in self.esm_model.parameters():
            para.requires_grad = False




    def encode_compounds(self, smiles_batch):
        """
        First embed entities with feature
        """
        compound_graphs, compound_vectors = smiles_batch
        # if len(compound_vectors) > 0:  # up and keep
        #
        #     for l in self.drug_gcn:  # l.weight: float32  The dtype error does not occur every time
        #         compound_vectors = nnfun.relu(l(compound_graphs,
        #                                         compound_vectors))  # compound_graphs has node data x with dtype float64, compound_vector is also float64
        #
        #     compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors))
        # else:
        #     compound_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        for l in self.drug_gcn:  # l.weight: float32  The dtype error does not occur every time
            compound_vectors = nnfun.relu(l(compound_graphs,
                                            compound_vectors))  # compound_graphs has node data x with dtype float64, compound_vector is also float64

        compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors.squeeze()))  # CHGH

        return compound_vectors



    def encode_proteins(self, protein_vectors):
        # if len(protein_seqs) > 0:  # self.entity_dim, down and up
        protein_vectors = self.esm_model(protein_vectors, repr_layers=[33], return_contacts=False)
        protein_vectors = protein_vectors["representations"][33]
        protein_vectors = protein_vectors.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vectors = nnfun.relu(l(protein_vectors))
        protein_vectors = nnfun.adaptive_max_pool1d(protein_vectors, output_size=1)
        protein_vectors = protein_vectors.view(protein_vectors.size(0), -1)
        # else:
        #     protein_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        return protein_vectors



    def forward(self, smiles_embeds, seq_embeds):
        smiles_embeds = self.encode_compounds(smiles_embeds)
        seq_embeds = self.encode_proteins(seq_embeds)
        res = self.fc(torch.cat((smiles_embeds, seq_embeds), 1))
        return res

    @staticmethod
    def _move_self_inputs(smiles_feats, seq_feats, device):
        def move(value):
            if isinstance(value, (tuple, list)):
                return tuple(move(item) for item in value)
            return value.to(device)
        return move(smiles_feats), move(seq_feats)


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                smiles_feats, seq_feats, labels, _, _, sample_weights = batch
            else:
                smiles_feats, seq_feats, labels, _, _ = batch
                sample_weights = None
            smiles_feats, seq_feats = self._move_self_inputs(smiles_feats, seq_feats, args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            if sample_weights is not None:
                sample_weights = sample_weights.to(args.device)
                losses = nnfun.cross_entropy(outs, labels, reduction="none")
                loss = (losses * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
            else:
                loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                smiles_feats, seq_feats, labels = batch[0], batch[1], batch[2]
                smiles_feats, seq_feats = self._move_self_inputs(smiles_feats, seq_feats, args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs



class SelfEmbedModel(nn.Module):
    def __init__(self, hidden_dim, drug_embed_dim, protein_embed_dim, dropout=0.2):
        super().__init__()
        self.kwargs = {
            "hidden_dim": hidden_dim,
            "self_encoder": "pretrained_embed",
            "drug_embed_dim": drug_embed_dim,
            "protein_embed_dim": protein_embed_dim,
            "self_embed_dropout": dropout,
        }
        self.entity_dim = hidden_dim
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(2, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(2, hidden_dim // 2), 2),
        )

    def forward(self, drug_embeds, protein_embeds):
        drug_embeds = self.drug_proj(drug_embeds.float())
        protein_embeds = self.protein_proj(protein_embeds.float())
        pair_features = torch.cat((
            drug_embeds,
            protein_embeds,
            drug_embeds * protein_embeds,
            torch.abs(drug_embeds - protein_embeds),
        ), dim=1)
        return self.fc(pair_features)

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                drug_feats, protein_feats, labels, _, _, sample_weights = batch
            else:
                drug_feats, protein_feats, labels, _, _ = batch
                sample_weights = None
            drug_feats = drug_feats.to(args.device)
            protein_feats = protein_feats.to(args.device)
            labels = labels.to(args.device)
            outs = self(drug_feats, protein_feats)
            if sample_weights is not None:
                sample_weights = sample_weights.to(args.device)
                losses = nnfun.cross_entropy(outs, labels, reduction="none")
                loss = (losses * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
            else:
                loss = loss_fn(outs, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        all_probs, all_classes, all_labels = list(), list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                drug_feats, protein_feats, labels = batch[0], batch[1], batch[2]
                drug_feats = drug_feats.to(args.device)
                protein_feats = protein_feats.to(args.device)
                out = self(drug_feats, protein_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")
        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        return all_probs


class HybridSelfModel(nn.Module):
    def __init__(self, hidden_dim, esm_model, drug_embed_dim, protein_embed_dim, dropout=0.2):
        super().__init__()
        self.kwargs = {
            "hidden_dim": hidden_dim,
            "self_encoder": "hybrid_embed",
            "drug_embed_dim": drug_embed_dim,
            "protein_embed_dim": protein_embed_dim,
            "self_embed_dropout": dropout,
        }
        self.entity_dim = hidden_dim
        self.graph_branch = SelfModel(hidden_dim, esm_model)
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(2, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(2, hidden_dim // 2), 2),
        )

    @staticmethod
    def _move_inputs(value, device):
        if isinstance(value, (tuple, list)):
            return tuple(HybridSelfModel._move_inputs(item, device) for item in value)
        return value.to(device)

    def forward(self, smiles_inputs, seq_inputs):
        graph_inputs, drug_embeds = smiles_inputs
        seq_tokens, protein_embeds = seq_inputs
        graph_drug = self.graph_branch.encode_compounds(graph_inputs)
        graph_protein = self.graph_branch.encode_proteins(seq_tokens)
        embed_drug = self.drug_proj(drug_embeds.float())
        embed_protein = self.protein_proj(protein_embeds.float())
        pair_features = torch.cat((
            graph_drug,
            graph_protein,
            embed_drug,
            embed_protein,
            embed_drug * embed_protein,
            torch.abs(embed_drug - embed_protein),
        ), dim=1)
        return self.fc(pair_features)

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                smiles_feats, seq_feats, labels, _, _, sample_weights = batch
            else:
                smiles_feats, seq_feats, labels, _, _ = batch
                sample_weights = None
            smiles_feats = self._move_inputs(smiles_feats, args.device)
            seq_feats = self._move_inputs(seq_feats, args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            if sample_weights is not None:
                sample_weights = sample_weights.to(args.device)
                losses = nnfun.cross_entropy(outs, labels, reduction="none")
                loss = (losses * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
            else:
                loss = loss_fn(outs, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        all_probs, all_classes, all_labels = list(), list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                smiles_feats, seq_feats, labels = batch[0], batch[1], batch[2]
                smiles_feats = self._move_inputs(smiles_feats, args.device)
                seq_feats = self._move_inputs(seq_feats, args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")
        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        return all_probs




class RelationModel(nn.Module):
    def __init__(self, entity_dim, entity_embeddings):  # suspect: embedding re-saved cannot be added to statedict
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(entity_dim * 2, entity_dim), nn.ReLU(),
                                nn.Linear(entity_dim, 2))
        self.entity_embeddings = nn.Parameter(entity_embeddings)
        self.kwargs = {
            "entity_dim": entity_dim,
        }

    def forward(self, heads, tails):
        heads = self.entity_embeddings[heads]
        tails = self.entity_embeddings[tails]
        res = self.fc(torch.cat((heads, tails), 1))
        return res

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                _, _, labels, heads, tails, sample_weights = batch
            else:
                _, _, labels, heads, tails = batch
                sample_weights = None
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            outs = self(heads, tails)
            if sample_weights is not None:
                sample_weights = sample_weights.to(args.device)
                losses = nnfun.cross_entropy(outs, labels, reduction="none")
                loss = (losses * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
            else:
                loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def train_distill_epoch(self, data_loader, optimizer, args, distill_weight):
        self.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            _, _, labels, heads, tails, sample_weights = batch
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            sample_weights = sample_weights.to(args.device).float()

            outs = self(heads, tails)
            logits = outs[:, 1] - outs[:, 0]
            probs = nnfun.softmax(outs, dim=1)[:, 1]

            hard_mask = labels >= 0
            loss_terms = list()
            if hard_mask.any():
                hard_losses = nnfun.cross_entropy(outs[hard_mask], labels[hard_mask], reduction="none")
                hard_weights = sample_weights[hard_mask]
                loss_terms.append((hard_losses * hard_weights).sum() / hard_weights.sum().clamp(min=1e-8))

            soft_mask = labels < 0
            if soft_mask.any() and distill_weight > 0:
                soft_targets = sample_weights[soft_mask].clamp(0.0, 1.0)
                soft_loss = nnfun.mse_loss(probs[soft_mask], soft_targets)
                loss_terms.append(distill_weight * soft_loss)

                pos_logits = logits[soft_mask & (sample_weights >= args.s2r_distill_pos_threshold)]
                neg_logits = logits[soft_mask & (sample_weights <= args.s2r_distill_neg_threshold)]
                if pos_logits.numel() > 0 and neg_logits.numel() > 0 and args.s2r_ranking_weight > 0:
                    pair_count = min(pos_logits.numel(), neg_logits.numel(), args.s2r_ranking_pairs)
                    if pair_count > 0:
                        pos_logits = pos_logits[:pair_count]
                        neg_logits = neg_logits[:pair_count]
                        target = torch.ones(pair_count, device=args.device)
                        ranking_loss = nnfun.margin_ranking_loss(pos_logits, neg_logits, target,
                                                                  margin=args.s2r_ranking_margin)
                        loss_terms.append(distill_weight * args.s2r_ranking_weight * ranking_loss)

            if len(loss_terms) == 0:
                continue
            loss = sum(loss_terms)
            epoch_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss


    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                labels, heads, tails = batch[2], batch[3], batch[4]
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs = self(heads, tails)
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs


    def freeze_embeddings(self):
        self.entity_embeddings.requires_grad = False

    def unfreeze_embeddings(self):
        self.entity_embeddings.requires_grad = True



class GateModel(nn.Module):
    def __init__(self, relation_model, self_model, if_double_layer):
        super().__init__()
        self.relation_model = relation_model
        self.self_model = self_model
        self.kwargs = {"if_double_layer": if_double_layer, "gate_type": "safe_free"}
        feature_dim = 12
        hidden_dim = 16 if if_double_layer else 8
        self.free_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.complement_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.anchor_mode = "balanced"
        self.anchor_prior_is_self = 1.0
        self.adaptive_non_prior_budget = None
        self.adaptive_budget_stats = None
        self.adaptive_regularization_regime = None
        self.complement_reliability = 1.0
        self.complement_reliability_stats = None
        self.local_competence_bank = None
        self.local_competence_weak_win = None
        self.local_competence_anchor_win = None
        self.local_competence_stats = None
        nn.init.zeros_(self.free_head[-1].weight)
        nn.init.zeros_(self.free_head[-1].bias)
        nn.init.zeros_(self.complement_head[-1].weight)
        nn.init.zeros_(self.complement_head[-1].bias)
        self.relation_model.freeze_embeddings()

    def freeze(self):
        for para in self.relation_model.parameters():
            para.requires_grad = False
        for para in self.self_model.parameters():
            para.requires_grad = False

    def set_anchor_mode(self, anchor_mode, anchor_prior_is_self=None):
        self.anchor_mode = anchor_mode
        if anchor_prior_is_self is not None:
            self.anchor_prior_is_self = float(anchor_prior_is_self)

    def set_adaptive_budget(self, budget, stats=None, regime=None):
        self.adaptive_non_prior_budget = float(budget)
        self.adaptive_budget_stats = stats
        self.adaptive_regularization_regime = regime

    def set_complement_reliability(self, reliability, stats=None):
        self.complement_reliability = float(reliability)
        self.complement_reliability_stats = stats

    def set_local_competence_bank(self, features, weak_win, anchor_win, stats=None):
        if features is None or weak_win is None or anchor_win is None or features.numel() == 0:
            self.local_competence_bank = None
            self.local_competence_weak_win = None
            self.local_competence_anchor_win = None
            self.local_competence_stats = stats
            return
        self.local_competence_bank = features.detach().float().cpu()
        self.local_competence_weak_win = weak_win.detach().float().cpu()
        self.local_competence_anchor_win = anchor_win.detach().float().cpu()
        self.local_competence_stats = stats

    def _local_competence(self, base_features, args):
        if (not getattr(args, "gate_local_competence", False) or
                self.local_competence_bank is None or
                self.local_competence_weak_win is None):
            zeros = base_features.new_zeros(base_features.size(0))
            return zeros, zeros, zeros
        bank = self.local_competence_bank.to(base_features.device)
        weak_win = self.local_competence_weak_win.to(base_features.device)
        anchor_win = self.local_competence_anchor_win.to(base_features.device)
        if bank.size(0) == 0:
            zeros = base_features.new_zeros(base_features.size(0))
            return zeros, zeros, zeros
        k = int(max(1, min(getattr(args, "gate_local_competence_k", 5), bank.size(0))))
        tau = max(getattr(args, "gate_local_competence_tau", 1.0), 1e-8)
        distances = torch.cdist(base_features.detach(), bank)
        raw_distances = distances
        exclude_eps = getattr(args, "gate_local_competence_exclude_self_eps", 1e-7)
        if exclude_eps is not None and exclude_eps >= 0 and bank.size(0) > k:
            distances = distances.masked_fill(distances <= exclude_eps, float("inf"))
            empty_rows = torch.isinf(distances).all(dim=1)
            if empty_rows.any():
                distances = torch.where(empty_rows.unsqueeze(1), raw_distances, distances)
        knn_dist, knn_idx = torch.topk(distances, k=k, dim=1, largest=False)
        weights = torch.softmax(-knn_dist / tau, dim=1)
        local_weak_win = (weak_win[knn_idx] * weights).sum(dim=1)
        local_anchor_win = (anchor_win[knn_idx] * weights).sum(dim=1)
        local_gap = local_weak_win - local_anchor_win
        return local_weak_win, local_anchor_win, local_gap

    def _non_prior_budget(self, args):
        if getattr(args, "gate_adaptive_budget", False) and self.adaptive_non_prior_budget is not None:
            return self.adaptive_non_prior_budget
        return getattr(args, "gate_non_prior_budget", 0.2)

    def _bounded_logits(self, logits, args=None):
        logit_clip = getattr(args, "gate_logit_clip", 0.0) if args is not None else 0.0
        if logit_clip is None or logit_clip <= 0:
            return torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.nan_to_num(logits, nan=0.0, posinf=logit_clip, neginf=-logit_clip).clamp(-logit_clip, logit_clip)

    def _expert_logits(self, heads, tails, smiles_embeds, seq_embeds):
        with torch.no_grad():
            res_self = self.self_model(smiles_embeds, seq_embeds)
            res_relation = self.relation_model(heads, tails)
        return res_self, res_relation

    @staticmethod
    def _move_self_inputs(smiles_feats, seq_feats, device):
        def move(value):
            if isinstance(value, (tuple, list)):
                return tuple(move(item) for item in value)
            return value.to(device)
        return move(smiles_feats), move(seq_feats)

    def _forward_with_aux(self, heads, tails, smiles_embeds, seq_embeds, args=None):
        raw_self, raw_relation = self._expert_logits(heads, tails, smiles_embeds, seq_embeds)
        res_self = self._bounded_logits(raw_self, args)
        res_relation = self._bounded_logits(raw_relation, args)

        self_prob = nnfun.softmax(res_self.detach(), dim=1)[:, 1]
        rel_prob = nnfun.softmax(res_relation.detach(), dim=1)[:, 1]
        self_conf = torch.abs(self_prob - 0.5) * 2.0
        rel_conf = torch.abs(rel_prob - 0.5) * 2.0
        disagreement = torch.abs(self_prob - rel_prob)
        self_margin = (res_self[:, 1] - res_self[:, 0]).detach()
        rel_margin = (res_relation[:, 1] - res_relation[:, 0]).detach()
        eps = 1e-8
        self_entropy = -(self_prob * torch.log(self_prob + eps) +
                         (1.0 - self_prob) * torch.log(1.0 - self_prob + eps))
        rel_entropy = -(rel_prob * torch.log(rel_prob + eps) +
                       (1.0 - rel_prob) * torch.log(1.0 - rel_prob + eps))
        with torch.no_grad():
            head_emb = self.relation_model.entity_embeddings[heads]
            tail_emb = self.relation_model.entity_embeddings[tails]
            emb_cos = nnfun.cosine_similarity(head_emb, tail_emb, dim=1)
            head_norm = torch.tanh(torch.linalg.norm(head_emb, dim=1) / 10.0)
            tail_norm = torch.tanh(torch.linalg.norm(tail_emb, dim=1) / 10.0)
            emb_norm_gap = torch.abs(head_norm - tail_norm)
        base_features = torch.stack((
            self_prob,
            rel_prob,
            self_conf,
            rel_conf,
            disagreement,
            torch.tanh(self_margin),
            torch.tanh(rel_margin),
            torch.tanh(self_margin - rel_margin),
            self_entropy,
            rel_entropy,
            emb_cos,
            emb_norm_gap,
        ), dim=1)
        prior = getattr(args, "gate_anchor_prior_logit", 1.5) if args is not None else 1.5
        rel_prior = -prior if self.anchor_prior_is_self >= 0.5 else prior
        rel_weight = torch.sigmoid(self.free_head(base_features).squeeze(1) + rel_prior)
        self_weight = 1.0 - rel_weight
        raw_self_weight = self_weight
        raw_rel_weight = rel_weight
        strong_is_self = torch.full_like(self_weight, float(self.anchor_prior_is_self))
        anchor_conf = strong_is_self * self_conf + (1.0 - strong_is_self) * rel_conf
        weak_conf = strong_is_self * rel_conf + (1.0 - strong_is_self) * self_conf
        anchor_margin = strong_is_self * self_margin + (1.0 - strong_is_self) * rel_margin
        weak_margin = strong_is_self * rel_margin + (1.0 - strong_is_self) * self_margin
        if getattr(args, "gate_selective_complement", False):
            comp_temp = max(getattr(args, "gate_complement_temperature", 0.2), 1e-6)
            comp_delta = getattr(args, "gate_complement_delta", 0.0)
            comp_min = getattr(args, "gate_complement_min_select", 0.0)
            comp_max_weight = getattr(args, "gate_complement_max_weak_weight", 1.0)
            if getattr(args, "gate_supervised_complement", False):
                comp_score = self.complement_head(base_features).squeeze(1)
                complement_select_raw = torch.sigmoid((comp_score - comp_delta) / comp_temp)
            else:
                comp_score = (
                    getattr(args, "gate_complement_conf_scale", 1.0) * (weak_conf - anchor_conf) +
                    getattr(args, "gate_complement_margin_scale", 1.0) * torch.tanh(weak_margin - anchor_margin) +
                    getattr(args, "gate_complement_disagreement_scale", 0.5) * disagreement
                )
                complement_select_raw = torch.sigmoid((comp_score - comp_delta) / comp_temp)
            local_weak_win, local_anchor_win, local_gap = self._local_competence(base_features, args)
            if getattr(args, "gate_local_competence", False):
                local_scale = getattr(args, "gate_local_competence_scale", 4.0)
                local_bias = getattr(args, "gate_local_competence_bias", 0.0)
                local_floor = getattr(args, "gate_local_competence_floor", 0.0)
                local_gate = torch.sigmoid(local_scale * (local_gap - local_bias))
                if local_floor > 0:
                    local_gate = local_floor + (1.0 - local_floor) * local_gate
                complement_select_raw = complement_select_raw * local_gate
            complement_select = complement_select_raw
            if comp_min > 0:
                complement_select = comp_min + (1.0 - comp_min) * complement_select
            reliability = torch.as_tensor(self.complement_reliability, dtype=rel_weight.dtype, device=rel_weight.device)
            complement_select = complement_select * reliability
            if self.anchor_prior_is_self >= 0.5:
                weak_weight = (rel_weight * complement_select).clamp(max=comp_max_weight)
                rel_weight = weak_weight
                self_weight = 1.0 - weak_weight
            else:
                weak_weight = (self_weight * complement_select).clamp(max=comp_max_weight)
                self_weight = weak_weight
                rel_weight = 1.0 - weak_weight
        else:
            complement_select = torch.ones_like(rel_weight)
            complement_select_raw = complement_select
            comp_score = torch.zeros_like(rel_weight)
            local_weak_win = torch.zeros_like(rel_weight)
            local_anchor_win = torch.zeros_like(rel_weight)
            local_gap = torch.zeros_like(rel_weight)
        res = self_weight.unsqueeze(1) * res_self + rel_weight.unsqueeze(1) * res_relation
        anchor_logits = strong_is_self.unsqueeze(1) * res_self + (1.0 - strong_is_self).unsqueeze(1) * res_relation
        weak_logits = strong_is_self.unsqueeze(1) * res_relation + (1.0 - strong_is_self).unsqueeze(1) * res_self
        non_prior_weight = rel_weight if self.anchor_prior_is_self >= 0.5 else self_weight
        return res, {
            "weight": torch.stack((self_weight, rel_weight), dim=1),
            "raw_weight": torch.stack((raw_self_weight, raw_rel_weight), dim=1),
            "complement_select": complement_select,
            "complement_select_raw": complement_select_raw,
            "complement_score": comp_score,
            "local_weak_win": local_weak_win,
            "local_anchor_win": local_anchor_win,
            "local_gap": local_gap,
            "res_self": res_self,
            "res_relation": res_relation,
            "self_prob": self_prob,
            "rel_prob": rel_prob,
            "anchor_logits": anchor_logits,
            "weak_logits": weak_logits,
            "anchor_is_self": self_weight,
            "anchor_conf": anchor_conf,
            "weak_conf": weak_conf,
            "anchor_margin": anchor_margin,
            "weak_margin": weak_margin,
            "uncertainty": disagreement,
            "defer_prob": non_prior_weight,
            "alpha": torch.ones_like(non_prior_weight),
            "correction": non_prior_weight,
            "base_features": base_features,
            "gate_type": "safe_free",
        }

    def forward(self, heads, tails, smiles_embeds, seq_embeds):
        res, _ = self._forward_with_aux(heads, tails, smiles_embeds, seq_embeds)
        return res

    def _pairwise_rank_safe_loss(self, final_logits, anchor_logits, labels, real_mask, args):
        if getattr(args, "gate_rank_loss_weight", 1.0) <= 0:
            return final_logits.sum() * 0.0
        pos_mask = (labels == 1) & real_mask.bool()
        neg_mask = (labels == 0) & real_mask.bool()
        if not pos_mask.any() or not neg_mask.any():
            return final_logits.sum() * 0.0
        final_score = final_logits[:, 1] - final_logits[:, 0]
        anchor_score = anchor_logits[:, 1] - anchor_logits[:, 0]
        final_margin = final_score[pos_mask].unsqueeze(1) - final_score[neg_mask].unsqueeze(0)
        anchor_margin = anchor_score[pos_mask].detach().unsqueeze(1) - anchor_score[neg_mask].detach().unsqueeze(0)
        protect_mask = anchor_margin > 0
        if not protect_mask.any():
            return final_logits.sum() * 0.0
        margin = getattr(args, "gate_rank_margin", 0.0)
        terms = nnfun.relu(anchor_margin - final_margin + margin)
        return terms[protect_mask].mean()

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        self.freeze()
        self.self_model.eval()
        self.relation_model.eval()
        epoch_loss = 0
        sums = {
            "total": 0.0, "final": 0.0, "real_final": 0.0, "pseudo_final": 0.0,
            "safe": 0.0, "rank": 0.0, "anchor": 0.0, "defer": 0.0, "sparse": 0.0,
            "budget": 0.0, "cap": 0.0, "cap_mean": 0.0, "switch": 0.0,
            "contrast": 0.0, "protect": 0.0, "comp_select": 0.0,
            "comp_raw": 0.0, "comp_loss": 0.0, "comp_pos": 0.0, "comp_used": 0.0,
            "local_weak_win": 0.0, "local_anchor_win": 0.0, "local_gap": 0.0,
            "self_w": 0.0, "rel_w": 0.0, "defer_p": 0.0, "alpha": 0.0,
            "anchor_self": 0.0, "anchor_target_self": 0.0, "target": 0.0,
            "real_n": 0.0, "pseudo_n": 0.0, "n": 0.0,
        }
        pseudo_weight_cap = getattr(args, "gate_pseudo_weight_cap", 0.2)
        safe_weight = getattr(args, "gate_safe_loss_weight", 2.0)
        rank_weight = getattr(args, "gate_rank_loss_weight", 1.0)
        anchor_weight = getattr(args, "gate_anchor_loss_weight", 1.0)
        sparse_weight = getattr(args, "gate_sparse_loss_weight", 0.05)
        safe_margin = getattr(args, "gate_safe_margin", 0.0)
        defer_delta = getattr(args, "gate_defer_delta", 0.02)
        selector_temp = getattr(args, "gate_selector_temperature", 0.2)
        selector_pos_weight = getattr(args, "gate_selector_pos_weight", 3.0)
        non_prior_budget = self._non_prior_budget(args)
        budget_weight = getattr(args, "gate_budget_loss_weight", 1.0)
        cap_weight = getattr(args, "gate_adaptive_cap_loss_weight", 0.0)
        cap_extra = getattr(args, "gate_adaptive_cap_extra", 0.15)
        cap_min = getattr(args, "gate_adaptive_cap_min", 0.05)
        cap_max = getattr(args, "gate_adaptive_cap_max", 0.40)
        conf_scale = getattr(args, "gate_adaptive_conf_scale", 2.0)
        margin_scale = getattr(args, "gate_adaptive_margin_scale", 1.0)
        disagreement_scale = getattr(args, "gate_adaptive_disagreement_scale", 1.0)
        contrast_weight = getattr(args, "gate_contrast_loss_weight", 0.05)
        contrast_margin = getattr(args, "gate_contrast_margin", 0.05)
        protect_weight = getattr(args, "gate_self_protect_loss_weight", 0.1)
        comp_sup_weight = getattr(args, "gate_supervised_complement_loss_weight", 0.0)
        comp_sup_margin = getattr(args, "gate_supervised_complement_margin", 0.0)
        comp_sup_pos_weight = getattr(args, "gate_supervised_complement_pos_weight", 1.0)
        reg_regime = self.adaptive_regularization_regime or "none"
        if getattr(args, "gate_adaptive_regularization", False):
            if reg_regime == "low_gap":
                rank_weight *= getattr(args, "gate_low_gap_rank_scale", 0.3)
                anchor_weight *= getattr(args, "gate_low_gap_anchor_scale", 0.5)
                sparse_weight *= getattr(args, "gate_low_gap_sparse_scale", 0.2)
                protect_weight *= getattr(args, "gate_low_gap_protect_scale", 0.2)
            elif reg_regime == "mid_gap":
                rank_weight *= getattr(args, "gate_mid_gap_rank_scale", 1.0)
                anchor_weight *= getattr(args, "gate_mid_gap_anchor_scale", 1.0)
                sparse_weight *= getattr(args, "gate_mid_gap_sparse_scale", 1.0)
                protect_weight *= getattr(args, "gate_mid_gap_protect_scale", 2.0)
            elif reg_regime == "high_gap":
                rank_weight *= getattr(args, "gate_high_gap_rank_scale", 0.7)
                anchor_weight *= getattr(args, "gate_high_gap_anchor_scale", 0.8)
                sparse_weight *= getattr(args, "gate_high_gap_sparse_scale", 0.5)
                protect_weight *= getattr(args, "gate_high_gap_protect_scale", 1.0)
            print(f"[anchor-gate] adaptive_regularization regime={reg_regime}, "
                  f"anchor_w={anchor_weight:.4f}, rank_w={rank_weight:.4f}, "
                  f"sparse_w={sparse_weight:.4f}, protect_w={protect_weight:.4f}")
        grad_clip = getattr(args, "gate_grad_clip", 5.0)

        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                smiles_feats, seq_feats, labels, heads, tails, sample_weights = batch
            else:
                smiles_feats, seq_feats, labels, heads, tails = batch
                sample_weights = None
            smiles_feats, seq_feats = self._move_self_inputs(smiles_feats, seq_feats, args.device)
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            if sample_weights is not None:
                sample_weights = sample_weights.to(args.device).float()
                real_mask = (sample_weights >= 0.999).float()
                sample_weights = torch.where(
                    real_mask.bool(),
                    torch.ones_like(sample_weights),
                    sample_weights.clamp(max=pseudo_weight_cap)
                )
            else:
                sample_weights = torch.ones_like(labels, dtype=torch.float32, device=args.device)
                real_mask = torch.ones_like(sample_weights)

            outs, aux = self._forward_with_aux(heads, tails, smiles_feats, seq_feats, args)
            final_losses = nnfun.cross_entropy(outs, labels, reduction="none")
            pseudo_mask = 1.0 - real_mask
            real_count = real_mask.sum().clamp(min=1e-8)
            pseudo_count = pseudo_mask.sum().clamp(min=1e-8)
            real_final_loss = (final_losses * real_mask).sum() / real_count
            pseudo_final_loss = (final_losses * sample_weights * pseudo_mask).sum() / pseudo_count
            final_loss = real_final_loss + pseudo_final_loss

            anchor_losses = nnfun.cross_entropy(aux["anchor_logits"].detach(), labels, reduction="none")
            self_losses = nnfun.cross_entropy(aux["res_self"].detach(), labels, reduction="none")
            rel_losses = nnfun.cross_entropy(aux["res_relation"].detach(), labels, reduction="none")
            if self.anchor_prior_is_self >= 0.5:
                strong_losses, weak_losses = self_losses, rel_losses
                non_prior_weight = aux["weight"][:, 1]
                non_prior_targets = torch.sigmoid((strong_losses - weak_losses) / max(selector_temp, 1e-6)).detach()
                anchor_targets = 1.0 - non_prior_targets
            else:
                strong_losses, weak_losses = rel_losses, self_losses
                non_prior_weight = aux["weight"][:, 0]
                non_prior_targets = torch.sigmoid((strong_losses - weak_losses) / max(selector_temp, 1e-6)).detach()
                anchor_targets = non_prior_targets
            hard_weak_targets = (weak_losses + defer_delta < strong_losses).float()
            weak_clear = weak_losses + comp_sup_margin < strong_losses
            anchor_clear = strong_losses + comp_sup_margin < weak_losses
            comp_supervision_mask = (weak_clear | anchor_clear).float() * real_mask
            comp_targets = weak_clear.float()
            if getattr(args, "gate_local_competence_supervision", False):
                local_margin = getattr(args, "gate_local_competence_label_margin", 0.05)
                local_weak_clear = aux["local_gap"] > local_margin
                local_anchor_clear = aux["local_gap"] < -local_margin
                local_mask = (local_weak_clear | local_anchor_clear).float() * real_mask
                if getattr(args, "gate_local_competence_override_loss_labels", False):
                    comp_supervision_mask = local_mask
                    comp_targets = local_weak_clear.float()
                else:
                    agree_mask = ((local_weak_clear & weak_clear) | (local_anchor_clear & anchor_clear)).float()
                    comp_supervision_mask = comp_supervision_mask * agree_mask * real_mask
            if getattr(args, "gate_selective_complement", False) and getattr(args, "gate_supervised_complement", False):
                comp_pred = aux["complement_select_raw"].clamp(min=1e-6, max=1.0 - 1e-6)
                comp_bce = nnfun.binary_cross_entropy(comp_pred, comp_targets, reduction="none")
                comp_weights = 1.0 + (comp_sup_pos_weight - 1.0) * comp_targets
                comp_loss = (comp_bce * comp_weights * comp_supervision_mask).sum() / comp_supervision_mask.sum().clamp(min=1e-8)
            else:
                comp_loss = outs.sum() * 0.0
            anchor_terms = nnfun.binary_cross_entropy(non_prior_weight, non_prior_targets, reduction="none")
            selector_weights = 1.0 + (selector_pos_weight - 1.0) * hard_weak_targets
            selector_weights = selector_weights * real_mask
            anchor_loss = (anchor_terms * selector_weights).sum() / selector_weights.sum().clamp(min=1e-8)
            safe_terms = nnfun.relu(final_losses - anchor_losses + safe_margin)
            safe_loss = (safe_terms * real_mask).sum() / real_count
            rank_loss = self._pairwise_rank_safe_loss(outs, aux["anchor_logits"], labels, real_mask, args)
            defer_targets = hard_weak_targets
            defer_loss = outs.sum() * 0.0
            sparse_loss = non_prior_weight.mean()
            budget_loss = nnfun.relu(sparse_loss - non_prior_budget)
            switch_score = torch.sigmoid(
                conf_scale * (aux["weak_conf"] - aux["anchor_conf"]) +
                margin_scale * torch.tanh(aux["weak_margin"] - aux["anchor_margin"]) +
                disagreement_scale * aux["uncertainty"]
            ).detach()
            adaptive_cap = torch.clamp(non_prior_budget + cap_extra * switch_score, min=cap_min, max=cap_max)
            cap_terms = nnfun.relu(non_prior_weight - adaptive_cap)
            cap_loss = (cap_terms * real_mask).sum() / real_count
            strong_better = (strong_losses + defer_delta < weak_losses) & real_mask.bool()
            weak_better = hard_weak_targets.bool() & real_mask.bool()
            if strong_better.any():
                protect_loss = non_prior_weight[strong_better].mean()
            else:
                protect_loss = outs.sum() * 0.0
            if strong_better.any() and weak_better.any():
                contrast_loss = nnfun.relu(
                    contrast_margin - non_prior_weight[weak_better].mean() + non_prior_weight[strong_better].mean()
                )
            else:
                contrast_loss = outs.sum() * 0.0
            loss = (final_loss + safe_weight * safe_loss + rank_weight * rank_loss +
                    anchor_weight * anchor_loss + sparse_weight * sparse_loss +
                    budget_weight * budget_loss + cap_weight * cap_loss +
                    contrast_weight * contrast_loss +
                    protect_weight * protect_loss +
                    comp_sup_weight * comp_loss)
            epoch_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            n = labels.numel()
            sums["total"] += float(loss.detach().cpu()) * n
            sums["final"] += float(final_loss.detach().cpu()) * n
            sums["real_final"] += float(real_final_loss.detach().cpu()) * n
            sums["pseudo_final"] += float(pseudo_final_loss.detach().cpu()) * n
            sums["safe"] += float(safe_loss.detach().cpu()) * n
            sums["rank"] += float(rank_loss.detach().cpu()) * n
            sums["anchor"] += float(anchor_loss.detach().cpu()) * n
            sums["defer"] += float(defer_loss.detach().cpu()) * n
            sums["sparse"] += float(sparse_loss.detach().cpu()) * n
            sums["budget"] += float(budget_loss.detach().cpu()) * n
            sums["cap"] += float(cap_loss.detach().cpu()) * n
            sums["cap_mean"] += float(adaptive_cap.detach().sum().cpu())
            sums["switch"] += float(switch_score.detach().sum().cpu())
            sums["contrast"] += float(contrast_loss.detach().cpu()) * n
            sums["protect"] += float(protect_loss.detach().cpu()) * n
            sums["comp_select"] += float(aux["complement_select"].detach().sum().cpu())
            sums["comp_raw"] += float(aux["complement_select_raw"].detach().sum().cpu())
            sums["comp_loss"] += float(comp_loss.detach().cpu()) * n
            sums["comp_pos"] += float((comp_targets.detach() * comp_supervision_mask).sum().cpu())
            sums["comp_used"] += float(comp_supervision_mask.detach().sum().cpu())
            sums["local_weak_win"] += float(aux["local_weak_win"].detach().sum().cpu())
            sums["local_anchor_win"] += float(aux["local_anchor_win"].detach().sum().cpu())
            sums["local_gap"] += float(aux["local_gap"].detach().sum().cpu())
            sums["self_w"] += float(aux["weight"][:, 0].detach().sum().cpu())
            sums["rel_w"] += float(aux["weight"][:, 1].detach().sum().cpu())
            sums["defer_p"] += float(aux["defer_prob"].detach().sum().cpu())
            sums["alpha"] += float(aux["alpha"].detach().sum().cpu())
            sums["anchor_self"] += float(aux["anchor_is_self"].detach().sum().cpu())
            sums["anchor_target_self"] += float((anchor_targets.detach() * real_mask).sum().cpu())
            sums["target"] += float((defer_targets.detach() * real_mask).sum().cpu())
            sums["real_n"] += float(real_mask.detach().sum().cpu())
            sums["pseudo_n"] += float(pseudo_mask.detach().sum().cpu())
            sums["n"] += n

        count = max(1.0, sums["n"])
        real_count_total = max(1e-8, sums["real_n"])
        print("[anchor-gate] "
              f"total={sums['total']/count:.4f}, final={sums['final']/count:.4f}, "
              f"real_final={sums['real_final']/count:.4f}, pseudo_final={sums['pseudo_final']/count:.4f}, "
              f"safe={sums['safe']/count:.4f}, rank={sums['rank']/count:.4f}, "
              f"anchor={sums['anchor']/count:.4f}, "
              f"defer={sums['defer']/count:.4f}, sparse={sums['sparse']/count:.4f}, "
              f"budget={sums['budget']/count:.4f}, cap={sums['cap']/count:.4f}, "
              f"cap_mean={sums['cap_mean']/count:.4f}, switch={sums['switch']/count:.4f}, "
              f"contrast={sums['contrast']/count:.4f}, "
              f"protect={sums['protect']/count:.4f}, comp_loss={sums['comp_loss']/count:.4f}, "
              f"comp_select={sums['comp_select']/count:.4f}, comp_raw={sums['comp_raw']/count:.4f}, "
              f"local_weak_win={sums['local_weak_win']/count:.4f}, "
              f"local_anchor_win={sums['local_anchor_win']/count:.4f}, "
              f"local_gap={sums['local_gap']/count:.4f}, "
              f"comp_pos_rate={sums['comp_pos']/max(1e-8, sums['comp_used']):.4f}, "
              f"comp_used={sums['comp_used']:.0f}, "
              f"self_w={sums['self_w']/count:.4f}, rel_w={sums['rel_w']/count:.4f}, "
              f"defer_p={sums['defer_p']/count:.4f}, alpha={sums['alpha']/count:.4f}, "
              f"anchor_self={sums['anchor_self']/count:.4f}, "
              f"anchor_target_self={sums['anchor_target_self']/real_count_total:.4f}, "
              f"defer_target={sums['target']/real_count_total:.4f}, "
              f"real_n={sums['real_n']:.0f}, pseudo_n={sums['pseudo_n']:.0f}, "
              f"pseudo_ratio={sums['pseudo_n']/(sums['real_n'] + sums['pseudo_n'] + 1e-8):.4f}")
        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True, if_probs=False):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        self_probs, rel_probs, self_classes, rel_classes = list(), list(), list(), list()
        self_weights, rel_weights = list(), list()
        comp_selects, comp_raws = list(), list()
        local_weak_wins, local_anchor_wins, local_gaps = list(), list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                smiles_feats, seq_feats, labels, heads, tails = batch[0], batch[1], batch[2], batch[3], batch[4]
                smiles_feats, seq_feats = self._move_self_inputs(smiles_feats, seq_feats, args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs, aux = self._forward_with_aux(heads, tails, smiles_feats, seq_feats, args)
                if if_probs:
                    print(f"batch {i}:", end=" ")
                    print(nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist())
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                self_probs += nnfun.softmax(aux["res_self"], dim=1)[:, 1].cpu().tolist()
                rel_probs += nnfun.softmax(aux["res_relation"], dim=1)[:, 1].cpu().tolist()
                self_classes += aux["res_self"].argmax(1).cpu().tolist()
                rel_classes += aux["res_relation"].argmax(1).cpu().tolist()
                self_weights += aux["weight"][:, 0].cpu().tolist()
                rel_weights += aux["weight"][:, 1].cpu().tolist()
                comp_selects += aux["complement_select"].cpu().tolist()
                comp_raws += aux["complement_select_raw"].cpu().tolist()
                local_weak_wins += aux["local_weak_win"].cpu().tolist()
                local_anchor_wins += aux["local_anchor_win"].cpu().tolist()
                local_gaps += aux["local_gap"].cpu().tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            if getattr(args, "gate_eval_diagnostics", False):
                self_acc = accuracy_score(all_labels, self_classes)
                rel_acc = accuracy_score(all_labels, rel_classes)
                self_auc = roc_auc_score(all_labels, self_probs)
                rel_auc = roc_auc_score(all_labels, rel_probs)
                self_aupr = average_precision_score(all_labels, self_probs)
                rel_aupr = average_precision_score(all_labels, rel_probs)
                self_correct = [pred == y for pred, y in zip(self_classes, all_labels)]
                rel_correct = [pred == y for pred, y in zip(rel_classes, all_labels)]
                gate_correct = [pred == y for pred, y in zip(all_classes, all_labels)]
                fp_mask = [pred == 1 and y == 0 for pred, y in zip(all_classes, all_labels)]
                fn_mask = [pred == 0 and y == 1 for pred, y in zip(all_classes, all_labels)]
                tp = sum(pred == 1 and y == 1 for pred, y in zip(all_classes, all_labels))
                tn = sum(pred == 0 and y == 0 for pred, y in zip(all_classes, all_labels))
                fp = sum(fp_mask)
                fn = sum(fn_mask)

                def masked_mean(values, mask):
                    picked = [v for v, m in zip(values, mask) if m]
                    return sum(picked) / max(1, len(picked))

                self_only = [s and not r for s, r in zip(self_correct, rel_correct)]
                rel_only = [r and not s for s, r in zip(self_correct, rel_correct)]
                both_correct = [s and r for s, r in zip(self_correct, rel_correct)]
                both_wrong = [not s and not r for s, r in zip(self_correct, rel_correct)]
                print("[gate-eval] "
                      f"n={len(all_labels)}, acc={acc:.4f}, auc={auc:.4f}, aupr={aupr:.4f}, "
                      f"self_acc={self_acc:.4f}, self_auc={self_auc:.4f}, self_aupr={self_aupr:.4f}, "
                      f"rel_acc={rel_acc:.4f}, rel_auc={rel_auc:.4f}, rel_aupr={rel_aupr:.4f}, "
                      f"expert_gap_aupr={abs(self_aupr - rel_aupr):.4f}, "
                      f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
                      f"self_w={sum(self_weights)/max(1, len(self_weights)):.4f}, "
                      f"rel_w={sum(rel_weights)/max(1, len(rel_weights)):.4f}, "
                      f"comp_select={sum(comp_selects)/max(1, len(comp_selects)):.4f}, "
                      f"comp_raw={sum(comp_raws)/max(1, len(comp_raws)):.4f}, "
                      f"local_weak_win={sum(local_weak_wins)/max(1, len(local_weak_wins)):.4f}, "
                      f"local_anchor_win={sum(local_anchor_wins)/max(1, len(local_anchor_wins)):.4f}, "
                      f"local_gap={sum(local_gaps)/max(1, len(local_gaps)):.4f}, "
                      f"FP_self_w={masked_mean(self_weights, fp_mask):.4f}, FP_rel_w={masked_mean(rel_weights, fp_mask):.4f}, "
                      f"FN_self_w={masked_mean(self_weights, fn_mask):.4f}, FN_rel_w={masked_mean(rel_weights, fn_mask):.4f}, "
                      f"both_correct={sum(both_correct)}, self_only={sum(self_only)}, "
                      f"rel_only={sum(rel_only)}, both_wrong={sum(both_wrong)}, "
                      f"self_only_rel_w={masked_mean(rel_weights, self_only):.4f}, "
                      f"rel_only_rel_w={masked_mean(rel_weights, rel_only):.4f}, "
                      f"both_wrong_rel_w={masked_mean(rel_weights, both_wrong):.4f}, "
                      f"gate_hit_self_only={sum(g and m for g, m in zip(gate_correct, self_only))}, "
                      f"gate_hit_rel_only={sum(g and m for g, m in zip(gate_correct, rel_only))}, "
                      f"gate_hit_both_wrong={sum(g and m for g, m in zip(gate_correct, both_wrong))}")
            return acc, auc, aupr
        else:
            return all_probs

    def evaluate_experts(self, data_loader, args, if_metric=True):
        self.eval()
        self_all_probs, self_all_classes, all_labels = list(), list(), list()
        rel_all_probs, rel_all_classes = list(), list()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                smiles_feats, seq_feats, labels, heads, tails = batch[0], batch[1], batch[2], batch[3], batch[4]
                smiles_feats, seq_feats = self._move_self_inputs(smiles_feats, seq_feats, args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                self_outs = self.self_model(smiles_feats, seq_feats)
                rel_outs = self.relation_model(heads, tails)
                self_all_probs += nnfun.softmax(self_outs, dim=1)[:, 1].cpu().tolist()
                rel_all_probs += nnfun.softmax(rel_outs, dim=1)[:, 1].cpu().tolist()
                self_all_classes += self_outs.argmax(1).cpu().tolist()
                rel_all_classes += rel_outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished")

        if if_metric:
            self_acc = accuracy_score(all_labels, self_all_classes)
            rel_acc = accuracy_score(all_labels, rel_all_classes)
            self_auc = roc_auc_score(all_labels, self_all_probs)
            rel_auc = roc_auc_score(all_labels, rel_all_probs)
            self_aupr = average_precision_score(all_labels, self_all_probs)
            rel_aupr = average_precision_score(all_labels, rel_all_probs)
            return self_acc, self_auc, self_aupr, rel_acc, rel_auc, rel_aupr
        else:
            return self_all_probs, rel_all_probs
