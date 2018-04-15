import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms,datasets


Iteration = 1
LR = 0.001

##TO-DO: Import data here:
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

test_data = datasets.MNIST(root="./mnist/",transform = transform,train = False)

test_data_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size = 64,shuffle = True)



##


##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16,kernel_size = 5,stride = 1,padding = 2),
            nn.ReLU(), #Activation use ReLU
            nn.MaxPool2d(kernel_size = 2), #sample in 2x2 space
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32,kernel_size = 5,stride = 1,padding = 2),
            nn.ReLU(), #Activation use ReLU
            nn.MaxPool2d(kernel_size = 2), #sample in 2x2 space
        )
        self.out = nn.Linear(32*7*7,10) #output 10 classes
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output
        
        #return out #return output

my_net = Net()



optimizer = torch.optim.Adam(my_net.parameters(),lr = LR)
loss_function = nn.CrossEntropyLoss()
        
my_net.load_state_dict(torch.load('model.pkl'))

correct = 0

for step,(x,y) in enumerate(test_data_loader):
        batch_x = Variable(x)
        batch_y = Variable(y) 
  
        res = my_net(batch_x) #Output
        prediction = torch.max(res.data,1)[1]
        correct += torch.sum(prediction == batch_y.data)

print('Accuracy:',float(100*correct/len(test_data)),'%')



 


