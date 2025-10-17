import torch 
import torchvision 


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.convnext_tiny(pretrained=False)
        self.model.classifier[-1] = torch.nn.Linear(768, 1)

    
    def forward(self, x):
        return self.model(x)


if __name__  == "__main__":
    model = Model()
    x = torch.randn(1, 3, 64, 64)
