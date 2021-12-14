import os


def get_module2name(model):
    return {
        module: name
        for name, module in model.named_modules()
    }


class DataPorter:
    def __init__(self, model, **kwargs):
        self.model = model
        self.module2name = get_module2name(model)
        self.computing_device = kwargs.get('computing_device', 'cpu')
        self.weight_dir = kwargs.get('weight_dir', 'weights')

    def get_save_path(self, module):
        return os.path.join(self.weight_dir,
                            f'{self.module2name[module]}.pt')
