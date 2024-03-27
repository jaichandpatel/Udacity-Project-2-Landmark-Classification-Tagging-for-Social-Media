import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:

        super().__init__()
    
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv_bn1 = nn.BatchNorm2d(224)
        
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.conv_bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.conv_bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(64,128,3,1,1)
        self.conv_bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv_bn5 = nn.BatchNorm2d(128)
        
        self.conv_bn6 = nn.BatchNorm2d(256)
     
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*7*7, 4096)
        self.fc2 = nn.Linear(4096, 50)
     
        
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
     
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_bn2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_bn3(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv_bn4(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv_bn5(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv_bn6(x)

        # Flatten image into vector, pass to FC layers
        # print(x.shape)# [32, 64, 28, 28]
        x = x.view(-1, 256*7*7)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)
   
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=50, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 50]
    ), f"Expected an output tensor of size (2, 50), got {out.shape}"
