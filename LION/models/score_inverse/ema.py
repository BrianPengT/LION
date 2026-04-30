"""
This module implements the Exponential Moving Average (EMA) class for smoother training.

Author: Tianzhen Peng

References
----------
.. [HeyAmit2024] Hey Amit. "Exponential Moving Average (EMA) in PyTorch." Medium, 2024. https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb
"""

class EMA:
    def __init__(self, model, decay=0.9999):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = {name: tensor.clone() for name, tensor in state_dict["shadow"].items()}

    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model. Use for validation or inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]