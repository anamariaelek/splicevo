import torch

class SoftmaxSumPool(torch.nn.Module):
    def __init__(self, dim=-1, mult_factor = 1.0, mult_factor_learnable = False, direction = 'forward', **kwargs):
        super(SoftmaxSumPool, self).__init__()
        self.mult_factor_learnable = mult_factor_learnable
        self.min_mult_factor = mult_factor  # Store the minimum allowed value
        # Initialize learnable parameter with the given value
         # TODO: enable learnable tensor that provides different parameter for each channel 
        if mult_factor_learnable:
            self.mult_factor = torch.nn.Parameter(torch.tensor(self.min_mult_factor))
        else:
            self.mult_factor = torch.tensor(self.min_mult_factor)
        self.dim = dim
        self.direction = direction

    def forward(self, x):
        if self.mult_factor_learnable:
            # Ensure the learnable parameter does not go below the minimum value
            # I AM NOT SURE IF THIS IS THE RIGHT WAY TO DO IT to constrain a learnable parameter
            self.mult_factor.data = torch.clamp(self.mult_factor.data, min=self.min_mult_factor)
        weights = torch.exp(self.mult_factor * x)
        if self.direction == 'backward':
            weights = torch.flip(weights, [self.dim])
            x = torch.flip(x, [self.dim])
        cum_weights = torch.cumsum(weights, dim=self.dim)
        x = x * weights
        x = torch.cumsum(x, dim=self.dim)
        if self.direction == 'backward':
            x = torch.flip(x, [self.dim])
            cum_weights = torch.flip(cum_weights, [self.dim])
        return x/cum_weights

if __name__ == "__main__":
    torch.manual_seed(42)
    mu_factor = 1.
    layer_for = SoftmaxSumPool(mult_factor = mu_factor, direction='forward')
    layer_back = SoftmaxSumPool(mult_factor = mu_factor, direction='backward')
    x = torch.randn(1, 1, 20)
    print("Input:\n", x)
    y = layer_for(x)
    print("Output (forward):\n", y)
    y = layer_back(x)
    print("Output (backward):\n", y)
    # Compare to logcumsumexp
    # forward
    y = torch.logcumsumexp(mu_factor * x, dim=-1) / mu_factor
    print("Output (logcumsumexp):\n", y)
    # backward
    y = torch.flip(torch.logcumsumexp(mu_factor * torch.flip(x, [-1]), dim=-1), [-1]) / mu_factor
    print("Output (logcumsumexp backward):\n", y)

    mu_factor = 5.
    layer_for = SoftmaxSumPool(mult_factor = mu_factor, direction='forward')
    layer_back = SoftmaxSumPool(mult_factor = mu_factor, direction='backward')
    x = torch.randn(1, 1, 20)
    print("\nMultiplication factor", mu_factor)
    print("Input:\n", x)
    y = layer_for(x)
    print("Output (forward):\n", y)
    y = layer_back(x)
    print("Output (backward):\n", y)
    # Compare to logcumsumexp
    # forward
    y = torch.logcumsumexp(mu_factor * x, dim=-1) / mu_factor
    print("Output (logcumsumexp):\n", y)
    # backward
    y = torch.flip(torch.logcumsumexp(mu_factor * torch.flip(x, [-1]), dim=-1), [-1]) / mu_factor
    print("Output (logcumsumexp backward):\n", y)
