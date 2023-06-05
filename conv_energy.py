import torch


class SimpleConvEnergy2D(torch.nn.Module):

    def __init__(self, input_embedding_dim, n_conv, n_out_channels, kernel_size, f_nonlin=torch.nn.ReLU):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.nonlin = f_nonlin()
        self.convs = [torch.nn.Conv2d(input_embedding_dim, n_out_channels, kernel_size)]
        for j in range(1, n_conv):
            self.convs.append(torch.nn.Conv2d(n_out_channels, n_out_channels, kernel_size))
        self.readout = torch.nn.LazyLinear(1, bias=False)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        for conv in self.convs:
            x = conv(x)
            x = self.nonlin(x)
        x = torch.flatten(x, start_dim=-3, end_dim=-1)
        return self.readout(x)
