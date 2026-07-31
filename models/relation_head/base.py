import json

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaModel(nn.Module):
    def __init__(self, input_dim, num_relations, rel_freq=None, tau=1.0):
        super(VanillaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)

        self.span_head = nn.Linear(input_dim // 4, num_relations)
        self.pred_head = nn.Linear(input_dim // 4, num_relations)

        # prepare the log_prior values (i.e., the relative frequencies of the relations)
        self.tau = tau

        if rel_freq is not None:
            # 1. Align dictionary values with class indices [0 .. num_relations-1]
            raw_counts = [rel_freq.get(str(i), rel_freq.get(i, 0)) for i in range(num_relations)]
            class_counts = torch.tensor(raw_counts, dtype=torch.float32)
            
            # 2. Compute probabilities (pi)
            freq_sum = class_counts.sum()
            priors = class_counts / freq_sum
            
            # 3. Compute log-priors safely
            log_priors = torch.log(priors + 1e-12)

            # 4. Register buffer so it moves to GPU automatically
            self.register_buffer("log_priors", log_priors, persistent=False)
        else:
            self.log_priors = None  # Assign None directly instead of register_buffer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        span_pred = self.span_head(x)

        relation_pred = self.pred_head(x)

        # apply the logit adjustment based on the relative frequencies of the relations
        if self.log_priors is not None:
            relation_pred = relation_pred - (self.tau * self.log_priors)

        relation_pred = torch.max(relation_pred, dim=1).values

        return span_pred, relation_pred


class ObjectEncoder(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 hidden_dim=512,
                 num_heads=8,
                 num_layers=2):
        super(ObjectEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class PairProposalNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(PairProposalNetwork, self).__init__()
        self.pair_ffn = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim),
                                      nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, encoded_subjects, encoded_objects):
        sub_tokens = encoded_subjects.max(dim=1).values
        obj_tokens = encoded_objects.max(dim=1).values
        num_objects = obj_tokens.size(0)
        pair_matrix = torch.zeros(num_objects, num_objects)

        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    combined_features = torch.cat(
                        [sub_tokens[i], obj_tokens[j]], dim=-1)
                    pair_matrix[i, j] = self.pair_ffn(combined_features)

        return pair_matrix
