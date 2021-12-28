from typing import List

import torch
import torch.nn as nn

from .utils import rgetattr, rsetattr


class ChunkedEmbedding(nn.Module):
    def __init__(self, embedding, chunk_size=1000, dtype=None):
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

        def get_embedding_chunk(start_i):
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

        self.embedding_chunks = nn.ModuleList([
            get_embedding_chunk(i)
            for i in range(0, self.num_embeddings, chunk_size)
        ])
        self.n_chunk = len(self.embedding_chunks)

    @torch.jit.ignore
    def get_size(self, input_) -> List[int]:
        size = [*input_.size(), self.embedding_dim]
        return size

    @staticmethod
    @torch.jit.ignore
    def update(output, group_indice: List[torch.Tensor],
               group_output) -> None:
        output[group_indice] = group_output

    @staticmethod
    @torch.jit.ignore
    def get_group_indice(group, group_i: int) -> List[torch.Tensor]:
        group_indice = (group == group_i).nonzero(as_tuple=True)
        return group_indice

    @torch.jit.ignore
    def get_group_input(self, input_, group_indice: List[torch.Tensor],
                        group_i: int):
        group_input = input_[group_indice] - group_i * self.chunk_size
        return group_input

    def forward(self, input_):
        # Don't know why torch.jit.ignore the whole method still raises error
        # (even define a new method and call by this forward) torch==1.10.1
        output = torch.empty(self.get_size(input_),
                             device=input_.device, dtype=self._dtype)
        group = input_.div(self.chunk_size, rounding_mode='floor')

        # TODO use set instead of list
        # (Doesn't seem to be supported by TorchScript? torch==1.10.1)
        group_unique: List[int] = torch.unique(group).tolist()

        for group_i, embedding_chunk in enumerate(self.embedding_chunks):
            if group_i in group_unique:
                group_indice = self.get_group_indice(group, group_i)
                group_input = self.get_group_input(
                    input_, group_indice, group_i)
                group_output = embedding_chunk(group_input)
                self.update(output, group_indice, group_output)

        return output

    # Clean implementation
    # def forward(self, input_):
    #     output = torch.empty((*input_.size(), self.embedding_dim),
    #                          device=input_.device, dtype=self._dtype)
    #     group = input_.div(self.chunk_size, rounding_mode='floor')
    #     for group_i in group.unique():
    #         group_indice = (group == group_i).nonzero(as_tuple=True)
    #         group_input = input_[group_indice] - group_i * self.chunk_size
    #         output[group_indice] = self.embedding_chunks[group_i](group_input)
    #     return


def chunked_embedding(model, target_module_name, chunk_size=4096):
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


def test_chunked_embedding():
    embedding = nn.Embedding(42, 10)
    indices = torch.arange(42).view(2, 21)
    ground_truth = embedding(indices)

    chunked_emedding = ChunkedEmbedding(embedding, chunk_size=5)
    result = chunked_emedding(indices)

    assert result.equal(ground_truth)
