import torch

class OurWeight(torch.nn.Module):
    def __init__(self, feat_length):
        super(OurWeight, self).__init__()
        self.W = torch.nn.Sequential(
            torch.nn.Linear(2*feat_length, 1),
            torch.nn.Sigmoid()
        )




    def forward(self, x1, x2):

        y = self.W(torch.cat((x1, x2), dim=1))
        return y



