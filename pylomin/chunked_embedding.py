import torch
import torch.nn as nn

from .utils import maybe_print_gpu_memory_trace, rgetattr, rsetattr


class ChunkedEmbedding(nn.Module):
    def __init__(self, embedding, chunk_size=1000, device=None, dtype=None):
        super().__init__()

        self._dtype = dtype if dtype is not None else embedding.weight.dtype
        self.chunk_size = chunk_size

        self.embedding_dim = embedding.embedding_dim
        self.num_embeddings = embedding.num_embeddings
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

        def get_embedding_part(start_i):
            _weight = embedding.weight[start_i:start_i +
                                       chunk_size].detach().clone()
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

        self.embeding_chunks = nn.ModuleList([
            get_embedding_part(i)
            for i in range(0, self.num_embeddings, chunk_size)
        ])

    def forward(self, input_):
        output = torch.empty((*input_.size(), self.embedding_dim),
                             device=input_.device, dtype=self._dtype)
        group = input_.div(self.chunk_size, rounding_mode='floor')

        for group_i in group.unique():
            group_indice = (group == group_i).nonzero(as_tuple=True)
            group_input = input_[group_indice] - group_i * self.chunk_size
            output[group_indice] = self.embeding_chunks[group_i](group_input)

        return output


@maybe_print_gpu_memory_trace
def chunked_embedding(model, target_module_name, chunk_size=8000, verbose=False):
    r""" Attempts to split an `torch.nn.Embedding` layer into multiple chunks of `torch.nn.Embedding` with smaller `num_embeddings`.
    The `num_embeddings` of all chunks will be equal to `chunk_size`, except the last one.
    """
    old_embedding = rgetattr(model, target_module_name)
    assert isinstance(old_embedding, nn.Embedding), (
        f'{target_module_name} is not an nn.Embedding'
    )
    new_embedding = ChunkedEmbedding(old_embedding,
                                     chunk_size=chunk_size)
    rsetattr(model, target_module_name, new_embedding)
    return model


if __name__ == '__main__':
    embedding = nn.Embedding(42, 10)
    indices = torch.arange(42).view(2, 21)
    ground_truth = embedding(indices)

    chunked_emedding = ChunkedEmbedding(embedding, chunk_size=5)
    output = chunked_emedding(indices)

    assert output.equal(ground_truth)
    print('Passed!')
