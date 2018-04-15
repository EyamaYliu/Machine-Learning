import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable

##TO-DO: Import data here:
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

data_train = datasets.MNIST(root = "./data/",transform=transform,train = True,download = True)

data_test = datasets.MNIST(root="./data/",transform = transform,train = False)


data_loader_train = torch.utils.data.DataLoader(dataset=data_train,batch_size = 64,shuffle = True,num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,batch_size = 64,shuffle = True,num_workers=2)



##


##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ##Define layers making use of torch.nn functions:
    
    def forward(self, x):

        ##Define how forward pass / inference is done:

        
        #return out #return output

my_net = Net()


##TO-DO: Train your model:

#torch.save(my net.state dict(), ’model.pkl’)
