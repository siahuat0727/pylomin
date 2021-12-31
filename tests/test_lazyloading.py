import pylomin
import pytest
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def get_model_and_input():
    model = Net().eval()
    x = torch.rand(1, 1)
    return model, x


@pytest.mark.parametrize(
    "device, storage, is_jit",
    [
        # ('cuda', 'cpu', False),
        # ('cuda', 'disk', False),
        ('cpu', 'disk', False),
        # ('cuda', 'cpu', True),
        # TODO ('cuda', 'disk', True),
        ('cpu', 'disk', True),
    ]
)
def test_lazyloading(device, storage, is_jit):
    model, x = get_model_and_input()
    with torch.no_grad():
        ground_truth = model(x)

    x = x.to(device)
    ground_truth = ground_truth.to(device)

    model = pylomin.lazy_loading(model,
                                 device=device,
                                 storage=storage,
                                 jit=is_jit)
    with torch.no_grad():
        if is_jit:
            model = torch.jit.trace(model, x)
        assert model(x).allclose(ground_truth)
