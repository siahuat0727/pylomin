import torch
import torch.nn as nn
from pylomin.chunked_embedding import ChunkedEmbedding


def test_chunked_embedding():
    embedding = nn.Embedding(42, 10)
    indices = torch.arange(42).view(2, 21)
    ground_truth = embedding(indices)

    chunked_embedding = ChunkedEmbedding(embedding, chunk_size=5)
    result = chunked_embedding(indices)
    assert result.equal(ground_truth)
