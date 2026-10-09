"""Projected Conflicting Gradient (PCGrad) optimizer wrapper."""

import random

import torch


class PCGrad:
    """Apply PCGrad to a standard PyTorch optimizer.

    Conflicting gradients are projected independently for every objective.
    Gradients shared by all objectives are averaged; task-specific gradients
    are summed, matching the reference PCGrad implementation.
    """

    def __init__(self, optimizer, reduction='mean'):
        if reduction not in ('mean', 'sum'):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.optimizer = optimizer
        self.reduction = reduction

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optimizer.step(*args, **kwargs)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def pc_backward(self, objectives):
        objectives = list(objectives)
        if not objectives:
            raise ValueError('PCGrad requires at least one objective.')

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group['params']
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError('The wrapped optimizer has no trainable parameters.')

        packed_gradients = []
        gradient_masks = []
        shapes = [parameter.shape for parameter in parameters]

        for objective_index, objective in enumerate(objectives):
            gradients = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=objective_index < len(objectives) - 1,
                allow_unused=True,
            )
            flattened = []
            mask = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    flattened.append(torch.zeros_like(parameter).flatten())
                    mask.append(torch.zeros_like(parameter).flatten())
                else:
                    flattened.append(gradient.detach().clone().flatten())
                    mask.append(torch.ones_like(parameter).flatten())
            packed_gradients.append(torch.cat(flattened))
            gradient_masks.append(torch.cat(mask))

        projected_gradients = self._project_conflicts(packed_gradients)
        shared = torch.stack(gradient_masks).prod(dim=0).bool()
        merged = torch.zeros_like(projected_gradients[0])
        stacked = torch.stack(projected_gradients)
        if self.reduction == 'mean':
            merged[shared] = stacked[:, shared].mean(dim=0)
        else:
            merged[shared] = stacked[:, shared].sum(dim=0)
        merged[~shared] = stacked[:, ~shared].sum(dim=0)

        offset = 0
        for parameter, shape in zip(parameters, shapes):
            element_count = parameter.numel()
            parameter.grad = merged[offset:offset + element_count].view(shape)
            offset += element_count

    @staticmethod
    def _project_conflicts(gradients):
        projected = [gradient.clone() for gradient in gradients]
        for gradient in projected:
            other_gradients = list(gradients)
            random.shuffle(other_gradients)
            for other in other_gradients:
                inner_product = torch.dot(gradient, other)
                if inner_product < 0:
                    gradient.add_(
                        other,
                        alpha=(-inner_product / other.square().sum().clamp_min(1e-12)).item(),
                    )
        return projected

