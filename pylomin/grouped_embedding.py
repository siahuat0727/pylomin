import torch
import torch.nn as nn

from .utils import maybe_print_memory_trace, rgetattr, rsetattr


class GroupedEmbedding(nn.Module):
    def __init__(self, embedding, len_per_group=1000, device=None, dtype=None):
        super().__init__()

        self._device = device if device is not None else embedding.weight.device
        self._dtype = dtype if dtype is not None else embedding.weight.dtype
        self.len_per_group = len_per_group

        self.embedding_dim = embedding.embedding_dim
        self.num_embeddings = embedding.num_embeddings
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

        def get_embedding_part(start_i):
            _weight = embedding.weight[start_i:start_i +
                                       len_per_group].detach().clone()
            num_embeddings = _weight.size(0)
            return nn.Embedding(
                num_embeddings,
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
                _weight=_weight,
            )

        self.grouped_embeddings = nn.ModuleList([
            get_embedding_part(i)
            for i in range(0, self.num_embeddings, len_per_group)
        ])

    def forward(self, input_):
        output = torch.empty((*input_.size(), self.embedding_dim),
                             device=self._device, dtype=self._dtype)
        group = input_ // self.len_per_group

        for group_i in group.unique():
            mask_i = (group == group_i).nonzero(as_tuple=True)
            group_input = input_[mask_i] - group_i * self.len_per_group
            output[mask_i] = self.grouped_embeddings[group_i](group_input)

        return output


@maybe_print_memory_trace
def grouped_embedding(model, target_module_name, len_per_group=8000, verbose=False):
    old_embedding = rgetattr(model, target_module_name)
    assert isinstance(old_embedding, nn.Embedding), (
        f'{target_module_name} is not an nn.Embedding'
    )
    new_embedding = GroupedEmbedding(old_embedding,
                                     len_per_group=len_per_group)
    rsetattr(model, target_module_name, new_embedding)
    return model


if __name__ == '__main__':
    embedding = nn.Embedding(42, 10)
    indices = torch.arange(42).view(2, 21)
    ground_truth = embedding(indices)

    group_embedding = GroupedEmbedding(embedding, len_per_group=5)
    output = group_embedding(indices)

    assert output.equal(ground_truth)
    print('Passed!')
