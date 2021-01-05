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
        self.param = torch.nn.ModuleList(linear)

    def forward(self, x_v):
        for f in self.param:
            x_v = f(x_v)
        return x_v

class CNN(torch.nn.Module):
    def __init__(self, reso, d_out):
        super(CNN, self).__init__()
        self.c1 = torch.nn.Conv2d(4, 16, kernel_size=7, stride=3)
        self.n1 = torch.nn.BatchNorm2d(16)
        self.c2 = torch.nn.Conv2d(16, 64, kernel_size=5, stride=2)
        self.n2 = torch.nn.BatchNorm2d(64)
        self.c3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.n3 = torch.nn.BatchNorm2d(64)

        output_width = self.calc_output_size(reso[0], 7, 3)
        output_width = self.calc_output_size(output_width, 5, 2)
        output_width = self.calc_output_size(output_width, 3, 1)

        output_height = self.calc_output_size(reso[1], 7, 3)
        output_height = self.calc_output_size(output_height, 5, 2)
        output_height = self.calc_output_size(output_height, 3, 1)

        print(int(output_height) * int(output_width) * 64)
        self.l1 = torch.nn.Linear(int((int(output_height) * int(output_width) * 64)), d_out)

    def forward(self, x_v):
        x_v = torch.nn.functional.relu(self.n1(self.c1(x_v)))
        x_v = torch.nn.functional.relu(self.n2(self.c2(x_v)))
        x_v = torch.nn.functional.relu(self.n3(self.c3(x_v)))
        #x_v = torch.nn.functional.relu(self.c1(x_v))
        #x_v = torch.nn.functional.relu(self.c2(x_v))
        #x_v = torch.nn.functional.relu(self.c3(x_v))
        return self.l1(x_v.view(x_v.size(0), -1))

    def calc_output_size(self, size, k, s):
        return (size - (k - 1) - 1) // s + 1