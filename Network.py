import torch

class Network(torch.nn.Module):
    def __init__(self, d_in, h_tab, d_out, f=torch.nn.Sigmoid):
        super(Network, self).__init__()
        self.fun = f
        if len(h_tab) == 0:
            linear = [torch.nn.Linear(d_in, d_out, bias=False)]
        else:
            linear = list()
            linear.append(torch.nn.Linear(d_in, h_tab[0], bias=True))
            linear.append(self.fun())
            for h in range(len(h_tab) - 1):
                next_h = h + 1
                linear.append(torch.nn.Linear(h_tab[h], h_tab[next_h],
                                              bias=True))
                linear.append(self.fun())
            linear.append(torch.nn.Linear(h_tab[-1], d_out, bias=True))
            linear.append(torch.nn.Softmax(dim=1))
        self.param = torch.nn.ModuleList(linear)

    def forward(self, x_v):
        for f in self.param:
            x_v = f(x_v)
        return x_v
