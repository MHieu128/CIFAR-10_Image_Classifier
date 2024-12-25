# Introduction

In this project, you will build a neural network of your own design to evaluate the CIFAR-10 dataset.
Our target accuracy is 70%, but any accuracy over 50% is a great start.
Some of the benchmark results on CIFAR-10 include:

78.9% Accuracy | [Deep Belief Networks; Krizhevsky, 2010](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf)

90.6% Accuracy | [Maxout Networks; Goodfellow et al., 2013](https://arxiv.org/pdf/1302.4389.pdf)

96.0% Accuracy | [Wide Residual Networks; Zagoruyko et al., 2016](https://arxiv.org/pdf/1605.07146.pdf)

99.0% Accuracy | [GPipe; Huang et al., 2018](https://arxiv.org/pdf/1811.06965.pdf)

98.5% Accuracy | [Rethinking Recurrent Neural Networks and other Improvements for ImageClassification; Nguyen et al., 2020](https://arxiv.org/pdf/2007.15161.pdf)

Research with this dataset is ongoing. Notably, many of these networks are quite large and quite expensive to train. 

## Imports


```python
## This cell contains the essential imports you will need – DO NOT CHANGE THE CONTENTS! ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```

## Load the Dataset

Specify your transforms as a list first.
The transforms module is already loaded as `transforms`.

CIFAR-10 is fortunately included in the torchvision module.
Then, you can create your dataset using the `CIFAR10` object from `torchvision.datasets` ([the documentation is available here](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)).
Make sure to specify `download=True`! 

Once your dataset is created, you'll also need to define a `DataLoader` from the `torch.utils.data` module for both the train and the test set.


```python
# Define transforms
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create training set and define training dataloader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                      shuffle=True, num_workers=2)

# Create test set and define test dataloader
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                     shuffle=False, num_workers=2)

# The 10 classes in the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

    Files already downloaded and verified
    Files already downloaded and verified


## Explore the Dataset
Using matplotlib, numpy, and torch, explore the dimensions of your data.

You can view images using the `show5` function defined below – it takes a data loader as an argument.
Remember that normalized images will look really weird to you! You may want to try changing your transforms to view images.
Typically using no transforms other than `toTensor()` works well for viewing – but not as well for training your network.
If `show5` doesn't work, go back and check your code for creating your data loaders and your training/test sets.


```python
def show5(img_loader):
    dataiter = iter(img_loader)
    
    batch = next(dataiter)
    labels = batch[1][0:5]
    images = batch[0][0:5]
    for i in range(5):
        print(classes[labels[i]])
    
        image = images[i].numpy()
        plt.imshow(image.T)
        plt.show()
```


```python
# Explore data
show5(trainloader)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.94509804..1.0].


    truck



    
![png](output_6_2.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.96862745..0.99215686].


    cat



    
![png](output_6_5.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.9529412..1.0].


    truck



    
![png](output_6_8.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.0..0.9372549].


    bird



    
![png](output_6_11.png)
    


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.9764706..0.67058825].


    horse



    
![png](output_6_14.png)
    


## Build your Neural Network
Using the layers in `torch.nn` (which has been imported as `nn`) and the `torch.nn.functional` module (imported as `F`), construct a neural network based on the parameters of the dataset. 
Feel free to construct a model of any architecture – feedforward, convolutional, or even something more advanced!


```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.fc1 = nn.Linear(128 * 4 * 4, 512)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    x = x.view(-1, 128 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()
```

Specify a loss function and an optimizer, and instantiate the model.

If you use a less common loss function, please note why you chose that loss function in a comment.


```python
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss is suitable for multi-class classification tasks
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001
```

## Running your Neural Network
Use whatever method you like to train your neural network, and ensure you record the average loss at each epoch. 
Don't forget to use `torch.device()` and the `.to()` method for both your model and your data if you are using GPU!

If you want to print your loss during each epoch, you can use the `enumerate` function and print the loss after a set number of batches. 250 batches works well for most people!


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

num_epochs = 10
for epoch in range(num_epochs):  # loop over the dataset multiple times
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 250 == 249:    # print every 250 mini-batches
      print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 250:.3f}")
      running_loss = 0.0

print('Finished Training')
```

    [Epoch 1, Batch 250] loss: 2.230
    [Epoch 1, Batch 500] loss: 2.055
    [Epoch 1, Batch 750] loss: 1.993
    [Epoch 1, Batch 1000] loss: 1.910
    [Epoch 1, Batch 1250] loss: 1.914
    [Epoch 1, Batch 1500] loss: 1.852
    [Epoch 1, Batch 1750] loss: 1.830
    [Epoch 1, Batch 2000] loss: 1.792
    [Epoch 1, Batch 2250] loss: 1.756
    [Epoch 1, Batch 2500] loss: 1.794
    [Epoch 1, Batch 2750] loss: 1.746
    [Epoch 1, Batch 3000] loss: 1.657
    [Epoch 1, Batch 3250] loss: 1.667
    [Epoch 1, Batch 3500] loss: 1.660
    [Epoch 1, Batch 3750] loss: 1.669
    [Epoch 1, Batch 4000] loss: 1.627
    [Epoch 1, Batch 4250] loss: 1.587
    [Epoch 1, Batch 4500] loss: 1.592
    [Epoch 1, Batch 4750] loss: 1.575
    [Epoch 1, Batch 5000] loss: 1.588
    [Epoch 1, Batch 5250] loss: 1.547
    [Epoch 1, Batch 5500] loss: 1.565
    [Epoch 1, Batch 5750] loss: 1.507
    [Epoch 1, Batch 6000] loss: 1.485
    [Epoch 1, Batch 6250] loss: 1.488
    [Epoch 1, Batch 6500] loss: 1.458
    [Epoch 1, Batch 6750] loss: 1.467
    [Epoch 1, Batch 7000] loss: 1.503
    [Epoch 1, Batch 7250] loss: 1.459
    [Epoch 1, Batch 7500] loss: 1.431
    [Epoch 1, Batch 7750] loss: 1.377
    [Epoch 1, Batch 8000] loss: 1.467
    [Epoch 1, Batch 8250] loss: 1.451
    [Epoch 1, Batch 8500] loss: 1.384
    [Epoch 1, Batch 8750] loss: 1.426
    [Epoch 1, Batch 9000] loss: 1.335
    [Epoch 1, Batch 9250] loss: 1.354
    [Epoch 1, Batch 9500] loss: 1.280
    [Epoch 1, Batch 9750] loss: 1.379
    [Epoch 1, Batch 10000] loss: 1.323
    [Epoch 2, Batch 250] loss: 1.297
    [Epoch 2, Batch 500] loss: 1.318
    [Epoch 2, Batch 750] loss: 1.296
    [Epoch 2, Batch 1000] loss: 1.304
    [Epoch 2, Batch 1250] loss: 1.303
    [Epoch 2, Batch 1500] loss: 1.280
    [Epoch 2, Batch 1750] loss: 1.299
    [Epoch 2, Batch 2000] loss: 1.247
    [Epoch 2, Batch 2250] loss: 1.262
    [Epoch 2, Batch 2500] loss: 1.208
    [Epoch 2, Batch 2750] loss: 1.227
    [Epoch 2, Batch 3000] loss: 1.201
    [Epoch 2, Batch 3250] loss: 1.241
    [Epoch 2, Batch 3500] loss: 1.228
    [Epoch 2, Batch 3750] loss: 1.174
    [Epoch 2, Batch 4000] loss: 1.239
    [Epoch 2, Batch 4250] loss: 1.238
    [Epoch 2, Batch 4500] loss: 1.223
    [Epoch 2, Batch 4750] loss: 1.209
    [Epoch 2, Batch 5000] loss: 1.185
    [Epoch 2, Batch 5250] loss: 1.205
    [Epoch 2, Batch 5500] loss: 1.195
    [Epoch 2, Batch 5750] loss: 1.154
    [Epoch 2, Batch 6000] loss: 1.215
    [Epoch 2, Batch 6250] loss: 1.206
    [Epoch 2, Batch 6500] loss: 1.190
    [Epoch 2, Batch 6750] loss: 1.168
    [Epoch 2, Batch 7000] loss: 1.260
    [Epoch 2, Batch 7250] loss: 1.135
    [Epoch 2, Batch 7500] loss: 1.182
    [Epoch 2, Batch 7750] loss: 1.154
    [Epoch 2, Batch 8000] loss: 1.155
    [Epoch 2, Batch 8250] loss: 1.124
    [Epoch 2, Batch 8500] loss: 1.188
    [Epoch 2, Batch 8750] loss: 1.175
    [Epoch 2, Batch 9000] loss: 1.128
    [Epoch 2, Batch 9250] loss: 1.142
    [Epoch 2, Batch 9500] loss: 1.194
    [Epoch 2, Batch 9750] loss: 1.157
    [Epoch 2, Batch 10000] loss: 1.095
    [Epoch 3, Batch 250] loss: 1.104
    [Epoch 3, Batch 500] loss: 1.042
    [Epoch 3, Batch 750] loss: 1.001
    [Epoch 3, Batch 1000] loss: 1.124
    [Epoch 3, Batch 1250] loss: 1.083
    [Epoch 3, Batch 1500] loss: 1.112
    [Epoch 3, Batch 1750] loss: 1.083
    [Epoch 3, Batch 2000] loss: 1.087
    [Epoch 3, Batch 2250] loss: 1.129
    [Epoch 3, Batch 2500] loss: 1.059
    [Epoch 3, Batch 2750] loss: 1.063
    [Epoch 3, Batch 3000] loss: 1.060
    [Epoch 3, Batch 3250] loss: 1.059
    [Epoch 3, Batch 3500] loss: 1.086
    [Epoch 3, Batch 3750] loss: 1.113
    [Epoch 3, Batch 4000] loss: 1.089
    [Epoch 3, Batch 4250] loss: 1.060
    [Epoch 3, Batch 4500] loss: 1.054
    [Epoch 3, Batch 4750] loss: 1.105
    [Epoch 3, Batch 5000] loss: 1.051
    [Epoch 3, Batch 5250] loss: 1.035
    [Epoch 3, Batch 5500] loss: 1.003
    [Epoch 3, Batch 5750] loss: 1.041
    [Epoch 3, Batch 6000] loss: 1.048
    [Epoch 3, Batch 6250] loss: 1.081
    [Epoch 3, Batch 6500] loss: 0.990
    [Epoch 3, Batch 6750] loss: 1.057
    [Epoch 3, Batch 7000] loss: 1.049
    [Epoch 3, Batch 7250] loss: 0.985
    [Epoch 3, Batch 7500] loss: 0.996
    [Epoch 3, Batch 7750] loss: 1.075
    [Epoch 3, Batch 8000] loss: 1.002
    [Epoch 3, Batch 8250] loss: 1.022
    [Epoch 3, Batch 8500] loss: 1.084
    [Epoch 3, Batch 8750] loss: 1.023
    [Epoch 3, Batch 9000] loss: 1.016
    [Epoch 3, Batch 9250] loss: 1.016
    [Epoch 3, Batch 9500] loss: 1.022
    [Epoch 3, Batch 9750] loss: 1.057
    [Epoch 3, Batch 10000] loss: 0.999
    [Epoch 4, Batch 250] loss: 0.919
    [Epoch 4, Batch 500] loss: 0.913
    [Epoch 4, Batch 750] loss: 0.936
    [Epoch 4, Batch 1000] loss: 0.953
    [Epoch 4, Batch 1250] loss: 1.001
    [Epoch 4, Batch 1500] loss: 1.028
    [Epoch 4, Batch 1750] loss: 0.948
    [Epoch 4, Batch 2000] loss: 0.890
    [Epoch 4, Batch 2250] loss: 0.959
    [Epoch 4, Batch 2500] loss: 0.971
    [Epoch 4, Batch 2750] loss: 0.965
    [Epoch 4, Batch 3000] loss: 0.955
    [Epoch 4, Batch 3250] loss: 0.995
    [Epoch 4, Batch 3500] loss: 0.948
    [Epoch 4, Batch 3750] loss: 0.896
    [Epoch 4, Batch 4000] loss: 0.971
    [Epoch 4, Batch 4250] loss: 0.973
    [Epoch 4, Batch 4500] loss: 0.980
    [Epoch 4, Batch 4750] loss: 0.985
    [Epoch 4, Batch 5000] loss: 0.980
    [Epoch 4, Batch 5250] loss: 0.969
    [Epoch 4, Batch 5500] loss: 0.940
    [Epoch 4, Batch 5750] loss: 0.953
    [Epoch 4, Batch 6000] loss: 0.892
    [Epoch 4, Batch 6250] loss: 0.974
    [Epoch 4, Batch 6500] loss: 0.917
    [Epoch 4, Batch 6750] loss: 0.945
    [Epoch 4, Batch 7000] loss: 0.927
    [Epoch 4, Batch 7250] loss: 0.897
    [Epoch 4, Batch 7500] loss: 0.924
    [Epoch 4, Batch 7750] loss: 0.947
    [Epoch 4, Batch 8000] loss: 0.957
    [Epoch 4, Batch 8250] loss: 0.916
    [Epoch 4, Batch 8500] loss: 0.954
    [Epoch 4, Batch 8750] loss: 0.919
    [Epoch 4, Batch 9000] loss: 0.884
    [Epoch 4, Batch 9250] loss: 0.950
    [Epoch 4, Batch 9500] loss: 0.976
    [Epoch 4, Batch 9750] loss: 0.984
    [Epoch 4, Batch 10000] loss: 0.957
    [Epoch 5, Batch 250] loss: 0.821
    [Epoch 5, Batch 500] loss: 0.911
    [Epoch 5, Batch 750] loss: 0.916
    [Epoch 5, Batch 1000] loss: 0.835
    [Epoch 5, Batch 1250] loss: 0.890
    [Epoch 5, Batch 1500] loss: 0.843
    [Epoch 5, Batch 1750] loss: 0.902
    [Epoch 5, Batch 2000] loss: 0.921
    [Epoch 5, Batch 2250] loss: 0.906
    [Epoch 5, Batch 2500] loss: 0.924
    [Epoch 5, Batch 2750] loss: 0.908
    [Epoch 5, Batch 3000] loss: 0.862
    [Epoch 5, Batch 3250] loss: 0.895
    [Epoch 5, Batch 3500] loss: 0.857
    [Epoch 5, Batch 3750] loss: 0.901
    [Epoch 5, Batch 4000] loss: 0.831
    [Epoch 5, Batch 4250] loss: 0.833
    [Epoch 5, Batch 4500] loss: 0.860
    [Epoch 5, Batch 4750] loss: 0.889
    [Epoch 5, Batch 5000] loss: 0.872
    [Epoch 5, Batch 5250] loss: 0.876
    [Epoch 5, Batch 5500] loss: 0.832
    [Epoch 5, Batch 5750] loss: 0.846
    [Epoch 5, Batch 6000] loss: 0.920
    [Epoch 5, Batch 6250] loss: 0.890
    [Epoch 5, Batch 6500] loss: 0.853
    [Epoch 5, Batch 6750] loss: 0.874
    [Epoch 5, Batch 7000] loss: 0.854
    [Epoch 5, Batch 7250] loss: 0.901
    [Epoch 5, Batch 7500] loss: 0.894
    [Epoch 5, Batch 7750] loss: 0.828
    [Epoch 5, Batch 8000] loss: 0.882
    [Epoch 5, Batch 8250] loss: 0.853
    [Epoch 5, Batch 8500] loss: 0.866
    [Epoch 5, Batch 8750] loss: 0.855
    [Epoch 5, Batch 9000] loss: 0.852
    [Epoch 5, Batch 9250] loss: 0.852
    [Epoch 5, Batch 9500] loss: 0.885
    [Epoch 5, Batch 9750] loss: 0.854
    [Epoch 5, Batch 10000] loss: 0.912
    [Epoch 6, Batch 250] loss: 0.783
    [Epoch 6, Batch 500] loss: 0.756
    [Epoch 6, Batch 750] loss: 0.783
    [Epoch 6, Batch 1000] loss: 0.775
    [Epoch 6, Batch 1250] loss: 0.803
    [Epoch 6, Batch 1500] loss: 0.749
    [Epoch 6, Batch 1750] loss: 0.839
    [Epoch 6, Batch 2000] loss: 0.792
    [Epoch 6, Batch 2250] loss: 0.786
    [Epoch 6, Batch 2500] loss: 0.781
    [Epoch 6, Batch 2750] loss: 0.801
    [Epoch 6, Batch 3000] loss: 0.840
    [Epoch 6, Batch 3250] loss: 0.781
    [Epoch 6, Batch 3500] loss: 0.821
    [Epoch 6, Batch 3750] loss: 0.814
    [Epoch 6, Batch 4000] loss: 0.819
    [Epoch 6, Batch 4250] loss: 0.767
    [Epoch 6, Batch 4500] loss: 0.845
    [Epoch 6, Batch 4750] loss: 0.815
    [Epoch 6, Batch 5000] loss: 0.810
    [Epoch 6, Batch 5250] loss: 0.773
    [Epoch 6, Batch 5500] loss: 0.782
    [Epoch 6, Batch 5750] loss: 0.793
    [Epoch 6, Batch 6000] loss: 0.795
    [Epoch 6, Batch 6250] loss: 0.800
    [Epoch 6, Batch 6500] loss: 0.806
    [Epoch 6, Batch 6750] loss: 0.907
    [Epoch 6, Batch 7000] loss: 0.786
    [Epoch 6, Batch 7250] loss: 0.858
    [Epoch 6, Batch 7500] loss: 0.820
    [Epoch 6, Batch 7750] loss: 0.831
    [Epoch 6, Batch 8000] loss: 0.833
    [Epoch 6, Batch 8250] loss: 0.842
    [Epoch 6, Batch 8500] loss: 0.821
    [Epoch 6, Batch 8750] loss: 0.830
    [Epoch 6, Batch 9000] loss: 0.802
    [Epoch 6, Batch 9250] loss: 0.830
    [Epoch 6, Batch 9500] loss: 0.872
    [Epoch 6, Batch 9750] loss: 0.845
    [Epoch 6, Batch 10000] loss: 0.816
    [Epoch 7, Batch 250] loss: 0.732
    [Epoch 7, Batch 500] loss: 0.758
    [Epoch 7, Batch 750] loss: 0.754
    [Epoch 7, Batch 1000] loss: 0.724
    [Epoch 7, Batch 1250] loss: 0.789
    [Epoch 7, Batch 1500] loss: 0.690
    [Epoch 7, Batch 1750] loss: 0.753
    [Epoch 7, Batch 2000] loss: 0.709
    [Epoch 7, Batch 2250] loss: 0.779
    [Epoch 7, Batch 2500] loss: 0.760
    [Epoch 7, Batch 2750] loss: 0.755
    [Epoch 7, Batch 3000] loss: 0.718
    [Epoch 7, Batch 3250] loss: 0.831
    [Epoch 7, Batch 3500] loss: 0.715
    [Epoch 7, Batch 3750] loss: 0.737
    [Epoch 7, Batch 4000] loss: 0.741
    [Epoch 7, Batch 4250] loss: 0.776
    [Epoch 7, Batch 4500] loss: 0.693
    [Epoch 7, Batch 4750] loss: 0.699
    [Epoch 7, Batch 5000] loss: 0.767
    [Epoch 7, Batch 5250] loss: 0.744
    [Epoch 7, Batch 5500] loss: 0.805
    [Epoch 7, Batch 5750] loss: 0.717
    [Epoch 7, Batch 6000] loss: 0.782
    [Epoch 7, Batch 6250] loss: 0.728
    [Epoch 7, Batch 6500] loss: 0.688
    [Epoch 7, Batch 6750] loss: 0.804
    [Epoch 7, Batch 7000] loss: 0.748
    [Epoch 7, Batch 7250] loss: 0.814
    [Epoch 7, Batch 7500] loss: 0.833
    [Epoch 7, Batch 7750] loss: 0.848
    [Epoch 7, Batch 8000] loss: 0.748
    [Epoch 7, Batch 8250] loss: 0.839
    [Epoch 7, Batch 8500] loss: 0.808
    [Epoch 7, Batch 8750] loss: 0.822
    [Epoch 7, Batch 9000] loss: 0.744
    [Epoch 7, Batch 9250] loss: 0.755
    [Epoch 7, Batch 9500] loss: 0.714
    [Epoch 7, Batch 9750] loss: 0.797
    [Epoch 7, Batch 10000] loss: 0.796
    [Epoch 8, Batch 250] loss: 0.668
    [Epoch 8, Batch 500] loss: 0.685
    [Epoch 8, Batch 750] loss: 0.654
    [Epoch 8, Batch 1000] loss: 0.736
    [Epoch 8, Batch 1250] loss: 0.716
    [Epoch 8, Batch 1500] loss: 0.690
    [Epoch 8, Batch 1750] loss: 0.676
    [Epoch 8, Batch 2000] loss: 0.714
    [Epoch 8, Batch 2250] loss: 0.667
    [Epoch 8, Batch 2500] loss: 0.683
    [Epoch 8, Batch 2750] loss: 0.706
    [Epoch 8, Batch 3000] loss: 0.727
    [Epoch 8, Batch 3250] loss: 0.700
    [Epoch 8, Batch 3500] loss: 0.740
    [Epoch 8, Batch 3750] loss: 0.745
    [Epoch 8, Batch 4000] loss: 0.711
    [Epoch 8, Batch 4250] loss: 0.710
    [Epoch 8, Batch 4500] loss: 0.697
    [Epoch 8, Batch 4750] loss: 0.723
    [Epoch 8, Batch 5000] loss: 0.740
    [Epoch 8, Batch 5250] loss: 0.728
    [Epoch 8, Batch 5500] loss: 0.735
    [Epoch 8, Batch 5750] loss: 0.723
    [Epoch 8, Batch 6000] loss: 0.749
    [Epoch 8, Batch 6250] loss: 0.762
    [Epoch 8, Batch 6500] loss: 0.743
    [Epoch 8, Batch 6750] loss: 0.736
    [Epoch 8, Batch 7000] loss: 0.708
    [Epoch 8, Batch 7250] loss: 0.685
    [Epoch 8, Batch 7500] loss: 0.746
    [Epoch 8, Batch 7750] loss: 0.687
    [Epoch 8, Batch 8000] loss: 0.701
    [Epoch 8, Batch 8250] loss: 0.755
    [Epoch 8, Batch 8500] loss: 0.777
    [Epoch 8, Batch 8750] loss: 0.699
    [Epoch 8, Batch 9000] loss: 0.774
    [Epoch 8, Batch 9250] loss: 0.723
    [Epoch 8, Batch 9500] loss: 0.704
    [Epoch 8, Batch 9750] loss: 0.757
    [Epoch 8, Batch 10000] loss: 0.731
    [Epoch 9, Batch 250] loss: 0.651
    [Epoch 9, Batch 500] loss: 0.644
    [Epoch 9, Batch 750] loss: 0.638
    [Epoch 9, Batch 1000] loss: 0.658
    [Epoch 9, Batch 1250] loss: 0.592
    [Epoch 9, Batch 1500] loss: 0.627
    [Epoch 9, Batch 1750] loss: 0.667
    [Epoch 9, Batch 2000] loss: 0.717
    [Epoch 9, Batch 2250] loss: 0.672
    [Epoch 9, Batch 2500] loss: 0.723
    [Epoch 9, Batch 2750] loss: 0.692
    [Epoch 9, Batch 3000] loss: 0.720
    [Epoch 9, Batch 3250] loss: 0.714
    [Epoch 9, Batch 3500] loss: 0.685
    [Epoch 9, Batch 3750] loss: 0.668
    [Epoch 9, Batch 4000] loss: 0.659
    [Epoch 9, Batch 4250] loss: 0.684
    [Epoch 9, Batch 4500] loss: 0.659
    [Epoch 9, Batch 4750] loss: 0.653
    [Epoch 9, Batch 5000] loss: 0.681
    [Epoch 9, Batch 5250] loss: 0.662
    [Epoch 9, Batch 5500] loss: 0.656
    [Epoch 9, Batch 5750] loss: 0.675
    [Epoch 9, Batch 6000] loss: 0.719
    [Epoch 9, Batch 6250] loss: 0.649
    [Epoch 9, Batch 6500] loss: 0.596
    [Epoch 9, Batch 6750] loss: 0.701
    [Epoch 9, Batch 7000] loss: 0.757
    [Epoch 9, Batch 7250] loss: 0.661
    [Epoch 9, Batch 7500] loss: 0.658
    [Epoch 9, Batch 7750] loss: 0.737
    [Epoch 9, Batch 8000] loss: 0.725
    [Epoch 9, Batch 8250] loss: 0.668
    [Epoch 9, Batch 8500] loss: 0.693
    [Epoch 9, Batch 8750] loss: 0.665
    [Epoch 9, Batch 9000] loss: 0.662
    [Epoch 9, Batch 9250] loss: 0.646
    [Epoch 9, Batch 9500] loss: 0.725
    [Epoch 9, Batch 9750] loss: 0.709
    [Epoch 9, Batch 10000] loss: 0.713
    [Epoch 10, Batch 250] loss: 0.597
    [Epoch 10, Batch 500] loss: 0.658
    [Epoch 10, Batch 750] loss: 0.645
    [Epoch 10, Batch 1000] loss: 0.675
    [Epoch 10, Batch 1250] loss: 0.673
    [Epoch 10, Batch 1500] loss: 0.666
    [Epoch 10, Batch 1750] loss: 0.571
    [Epoch 10, Batch 2000] loss: 0.647
    [Epoch 10, Batch 2250] loss: 0.649
    [Epoch 10, Batch 2500] loss: 0.653
    [Epoch 10, Batch 2750] loss: 0.562
    [Epoch 10, Batch 3000] loss: 0.645
    [Epoch 10, Batch 3250] loss: 0.664
    [Epoch 10, Batch 3500] loss: 0.666
    [Epoch 10, Batch 3750] loss: 0.673
    [Epoch 10, Batch 4000] loss: 0.661
    [Epoch 10, Batch 4250] loss: 0.663
    [Epoch 10, Batch 4500] loss: 0.646
    [Epoch 10, Batch 4750] loss: 0.701
    [Epoch 10, Batch 5000] loss: 0.638
    [Epoch 10, Batch 5250] loss: 0.617
    [Epoch 10, Batch 5500] loss: 0.643
    [Epoch 10, Batch 5750] loss: 0.633
    [Epoch 10, Batch 6000] loss: 0.690
    [Epoch 10, Batch 6250] loss: 0.646
    [Epoch 10, Batch 6500] loss: 0.617
    [Epoch 10, Batch 6750] loss: 0.594
    [Epoch 10, Batch 7000] loss: 0.658
    [Epoch 10, Batch 7250] loss: 0.638
    [Epoch 10, Batch 7500] loss: 0.695
    [Epoch 10, Batch 7750] loss: 0.623
    [Epoch 10, Batch 8000] loss: 0.636
    [Epoch 10, Batch 8250] loss: 0.612
    [Epoch 10, Batch 8500] loss: 0.646
    [Epoch 10, Batch 8750] loss: 0.692
    [Epoch 10, Batch 9000] loss: 0.650
    [Epoch 10, Batch 9250] loss: 0.655
    [Epoch 10, Batch 9500] loss: 0.646
    [Epoch 10, Batch 9750] loss: 0.700
    [Epoch 10, Batch 10000] loss: 0.670
    Finished Training


Plot the training loss (and validation loss/accuracy, if recorded).


```python
# Assuming the training loss was recorded during the initial training process Assuming
epochs = range(1, num_epochs + 1)
train_losses = []

# Re-run the training loop to collect loss data
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  train_losses.append(running_loss / len(trainloader))

plt.plot(epochs, train_losses, label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
```


    
![png](output_14_0.png)
    


## Testing your model
Using the previously created `DataLoader` for the test set, compute the percentage of correct predictions using the highest probability prediction. 

If your accuracy is over 70%, great work! 
This is a hard task to exceed 70% on.

If your accuracy is under 45%, you'll need to make improvements.
Go back and check your model architecture, loss function, and optimizer to make sure they're appropriate for an image classification task.


```python
correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```

    Accuracy of the network on the 10000 test images: 72.78%



```python
# Get a random test image
dataiter = iter(testloader)
images, labels = next(dataiter)

# Display the image
img = images[1] / 2 + 0.5  # unnormalize
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

# Print the true label
print(f'True label: {classes[labels[1]]}')

# Predict the label
images = images.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

# Print the predicted label
print(f'Predicted label: {classes[predicted[1]]}')
```


    
![png](output_17_0.png)
    


    True label: ship
    Predicted label: ship


## Saving your model
Using `torch.save`, save your model for future loading.


```python
torch.save(net.state_dict(), 'cifar_net.pth')
```

## Make a Recommendation

Based on your evaluation, what is your recommendation on whether to build or buy? Explain your reasoning below.



**Double click this cell to modify it**


