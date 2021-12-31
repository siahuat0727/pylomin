class DataPorter:
    def __init__(self, model, **kwargs):
        self.model = model
        self.computing_device = kwargs.get('computing_device', 'cpu')

    @staticmethod
    def get_direct_parameters(module):
        # TODO: Don't access protected attribute (check if name contains dot?)
        return module._parameters.items()  # pylint: disable=protected-access

    @staticmethod
    def get_direct_buffers(module):
        # TODO: Don't access protected attribute (check if name contains dot?)
        return module._buffers.items()  # pylint: disable=protected-access
