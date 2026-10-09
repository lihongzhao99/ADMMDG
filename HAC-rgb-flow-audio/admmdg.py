"""Adaptive dual-objective feature learning for multimodal DG.

This implementation follows the public ADMMDG method description: modality-
domain invariant learning (MDIL), modality-specific domain-invariant learning
(MSDIL), adversarially learned pair weights, and feature separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def gradient_reverse(value, scale=1.0):
    """Reverse gradients for the adaptive weights, not for encoder features."""
    return _GradientReverse.apply(value, scale)


class ProjectionHead(nn.Module):
    """Projection head matching the official SimMMDG implementation."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return F.normalize(self.layers(features), dim=1)


def _adaptive_supervised_contrastive_loss(
        features, positive_mask, pair_logits, temperature,
        base_temperature=0.07):
    """SimMMDG/SupCon loss extended with normalized learnable pair weights."""
    sample_count = features.size(0)
    if sample_count < 2:
        return features.sum() * 0.0

    eye = torch.eye(sample_count, dtype=torch.bool, device=features.device)
    positive_mask = positive_mask.bool() & ~eye
    valid_anchors = positive_mask.any(dim=1)
    if not valid_anchors.any():
        return features.sum() * 0.0

    similarities = torch.matmul(features, features.T) / temperature
    logits_max = similarities.max(dim=1, keepdim=True).values
    logits = similarities - logits_max.detach()
    logits_mask = ~eye
    exp_logits = torch.exp(logits) * logits_mask
    log_probabilities = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    # Softmax constrains each anchor's weights to sum to one. Gradient
    # reversal makes the learned weights emphasize hard positive pairs while
    # the encoder still minimizes their contrastive loss.
    masked_pair_logits = pair_logits.masked_fill(~positive_mask, float('-inf'))
    positive_weights = torch.softmax(
        masked_pair_logits[valid_anchors], dim=1)
    positive_log_probabilities = log_probabilities[valid_anchors].masked_fill(
        ~positive_mask[valid_anchors], 0.0)
    mean_log_probability = (
        positive_weights * positive_log_probabilities).sum(dim=1)
    return -(temperature / base_temperature) * mean_log_probability.mean()


class AdaptiveDualObjectiveFeatureLearning(nn.Module):
    """ADMMDG auxiliary objectives for a dictionary of modality embeddings."""

    def __init__(self, embedding_dims, num_domains, projection_dim=128,
                 projection_hidden_dim=2048, temperature=0.1,
                 weight_grl_scale=1.0):
        super().__init__()
        if num_domains < 1:
            raise ValueError('num_domains must be at least one.')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')

        self.modality_names = tuple(embedding_dims)
        self.modality_to_idx = {
            name: index for index, name in enumerate(self.modality_names)
        }
        self.num_domains = num_domains
        self.single_source = num_domains == 1
        self.temperature = temperature
        self.weight_grl_scale = weight_grl_scale

        odd_dimensions = {
            name: dim for name, dim in embedding_dims.items() if dim % 2
        }
        if odd_dimensions:
            raise ValueError(
                'SimMMDG feature splitting requires even embedding dimensions: '
                f'{odd_dimensions}')

        self.mdil_heads = nn.ModuleDict({
            name: ProjectionHead(dim // 2, projection_hidden_dim, projection_dim)
            for name, dim in embedding_dims.items()
        })
        self.msdil_heads = nn.ModuleDict({
            name: ProjectionHead(dim // 2, projection_hidden_dim, projection_dim)
            for name, dim in embedding_dims.items()
        })

        group_count = len(self.modality_names) * num_domains
        self.mdil_pair_logits = nn.Parameter(torch.zeros(group_count, group_count))
        if self.single_source:
            # With one source domain, MSDIL has no cross-domain pair to
            # reweight. It therefore reduces to fixed-weight, within-modality
            # supervised contrastive learning.
            self.register_parameter('msdil_pair_logits', None)
        else:
            self.msdil_pair_logits = nn.Parameter(torch.zeros(
                len(self.modality_names), num_domains, num_domains))

    @staticmethod
    def _symmetric(logits):
        return 0.5 * (logits + logits.transpose(-1, -2))

    def forward(self, embeddings, labels, domain_labels):
        active_names = [
            name for name in self.modality_names if name in embeddings
        ]
        if len(active_names) < 2:
            raise ValueError('ADMMDG requires at least two active modalities.')

        labels = labels.long()
        domain_labels = domain_labels.to(labels.device).long()
        if domain_labels.min() < 0 or domain_labels.max() >= self.num_domains:
            raise ValueError('domain label is outside the configured range.')

        mdil_features = []
        msdil_features = []
        repeated_labels = []
        repeated_domains = []
        modality_indices = []
        separation_terms = []

        for name in active_names:
            split_dim = embeddings[name].size(1) // 2
            shared_raw = embeddings[name][:, :split_dim]
            specific_raw = embeddings[name][:, split_dim:]
            shared = self.mdil_heads[name](shared_raw)
            specific = self.msdil_heads[name](specific_raw)
            mdil_features.append(shared)
            msdil_features.append(specific)
            repeated_labels.append(labels)
            repeated_domains.append(domain_labels)
            modality_indices.append(torch.full_like(
                labels, self.modality_to_idx[name]))
            # SimMMDG maximizes the MSE between the two raw feature halves.
            separation_terms.append(-F.mse_loss(shared_raw, specific_raw))

        mdil_features = torch.cat(mdil_features, dim=0)
        msdil_features = torch.cat(msdil_features, dim=0)
        all_labels = torch.cat(repeated_labels, dim=0)
        all_domains = torch.cat(repeated_domains, dim=0)
        all_modalities = torch.cat(modality_indices, dim=0)

        same_label = all_labels[:, None].eq(all_labels[None, :])
        same_modality = all_modalities[:, None].eq(all_modalities[None, :])

        md_groups = all_modalities * self.num_domains + all_domains
        mdil_logits = self._symmetric(self.mdil_pair_logits)
        mdil_logits = gradient_reverse(mdil_logits, self.weight_grl_scale)
        mdil_pair_logits = mdil_logits[md_groups[:, None], md_groups[None, :]]
        mdil_loss = _adaptive_supervised_contrastive_loss(
            mdil_features, same_label, mdil_pair_logits, self.temperature)

        if self.single_source:
            msdil_pair_logits = msdil_features.new_zeros(
                (msdil_features.size(0), msdil_features.size(0)))
        else:
            msdil_logits = self._symmetric(self.msdil_pair_logits)
            msdil_logits = gradient_reverse(msdil_logits, self.weight_grl_scale)
            msdil_pair_logits = msdil_logits[
                all_modalities[:, None], all_domains[:, None],
                all_domains[None, :]]
        msdil_loss = _adaptive_supervised_contrastive_loss(
            msdil_features, same_label & same_modality,
            msdil_pair_logits, self.temperature)

        return {
            'mdil': mdil_loss,
            'msdil': msdil_loss,
            'separation': torch.stack(separation_terms).mean(),
        }
