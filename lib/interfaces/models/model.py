from ...tools.call_graph import cg


class Model:
    optimizer = None
    loss_fn = None
    device = None
    dtype = None
    model = None

    def __init__(self):
        cg.add("Model.__init__")
        raise NotImplementedError("Implement Me!")

    def set_optimizer(self, optimizer):
        cg.add("Model.set_optimizer")
        self.optimizer = optimizer
        return self

    def set_loss_fn(self, loss_fn):
        cg.add("Model.set_loss_fn")
        self.loss_fn = loss_fn
        return self

    def set_device(self, device):
        cg.add("Model.set_device")
        self.device = device
        return self

    def set_dtype(self, dtype):
        cg.add("Model.set_dtype")
        self.dtype = dtype
        return self

    def train_batch(self, *args, **kwargs):
        cg.add("Model.train_batch")
        raise NotImplementedError("Implement Me!")

    def eval_batch(self, *args, **kwargs):
        cg.add("Model.eval_batch")
        raise NotImplementedError("Implement Me!")

    def forward(self, x):
        cg.add("Model.forward")
        assert self.model is not None
        return self.model.forward(x)

    def to(self, device=None, dtype=None):
        cg.add("Model.to")
        raise NotImplementedError("Implement Me!")

    def load(self, checkpoint_path):
        cg.add("Model.load")
        raise NotImplementedError("Implement Me!")

    def save(self, checkpoint_path):
        cg.add("Model.save")
        raise NotImplementedError("Implement Me!")
