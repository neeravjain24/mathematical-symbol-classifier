import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

#from google.colab import drive
#drive.mount('/content/drive')


"""
Here is where we load the data, if you have a banlance dataset, we can run the stratify=True, if not, choose
shuffle=True
"""
data = np.load("Images.npy")
label = np.load("Labels.npy")
#X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,stratify=label)
X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,shuffle=True)


"""
Here is where we define the functions
The Resnet arhcitecuture code is learned from the Tutorial channel "Aladdin Persson"
We tried multiple DNN algorithms, the resnet is the best one, but we make some optimization 
to fit our spicific dataset
"""
class GetLoader(torch.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        #data = self.transform(data)
        return data, labels
    def __len__(self):
        return len(self.data)



class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )    
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
            
            """
            we tried to add attention map technique, which cause the model extrmely unrobust
            #gap = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
            #gap = gap.unsqueeze(2)
            #gap = gap.unsqueeze(3)     
        #print(x.size())
        #print(gap.size())
        
        
        """
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc1 = nn.Linear(512 * 4, 4096)
        self.fc1 = nn.Linear(512 *4, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512*4, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        #x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

        #return torch.nn.functional.softmax(x,dim=1)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)

def ResNet50(img_channel=3, num_classes=7 * 7 * (2 + 2 * 5)):
    return ResNet(block, [1, 1, 1, 1], img_channel, num_classes)
    #return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)




    
"""
All the transform technques have been tried for training, but none of them can increase the accuracy, so we decided to 
not use any of those

    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        #transforms.Normalize(0.5, 0.5), 
        #transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomChoice([
                                #transforms.RandomHorizontalFlip(p=0.5),
                                #transforms.ToTensor(),
                                #transforms.Normalize(0.5, 0.5), 
                                #transforms.RandomVerticalFlip(p=0.5),
                                transforms.GaussianBlur(5,(0.5,3)),
                                transforms.RandomAffine(degrees=(-15,15)),
        ]),
       
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.3, interpolation=2),

    ])

"""
    
    
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    acc = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == y).sum().item()
        tmp_acc=correct/BATCH_SIZE
        acc.append(tmp_acc)
        #print("the acc is: ",tmp_acc)


        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    print(f"Mean acc was {sum(acc)/len(acc)}")

    
"""
Here is where we preprocess the dataset, and define the model object
"""
X_train = np.expand_dims(X_train,1)
train_data_tmp = X_train/255
data = torch.tensor(train_data_tmp).float()
labels = torch.tensor(y_train.astype(int)).long()
#labels = one_hot(labels,25)
loader = GetLoader(data,labels)


X_test = np.expand_dims(X_test,1)
test_data_tmp = X_test/255.0
test_data = torch.tensor(test_data_tmp).float()
test_labels = torch.tensor(y_test.astype(int)).long()
test_loader = GetLoader(test_data,test_labels)




seed = 2021
torch.manual_seed(seed)
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0001
EPOCHS = 30
NUM_WORKERS = 2



"""
Here is where we start training the model, we applied a decay by each 5 epochs
to achieve the best performance
"""
model=ResNet50(1,25)
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()   
train_loader = DataLoader(loader, batch_size = BATCH_SIZE, drop_last=False,shuffle=True, num_workers=0)


for epoch in range(EPOCHS):
    print(epoch)
    if epoch >= 8 and epoch < 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2e-4
    if epoch >= 15 and epoch<20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2e-5
    if epoch >= 20 and epoch < 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2e-6
    if epoch >= 25 and epoch < 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 2e-7



    train_fn(train_loader, model, optimizer, loss_fn)

