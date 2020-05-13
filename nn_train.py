import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import timeit

from sklearn.metrics import classification_report, confusion_matrix

import helpers

#Set random seed
np.random.seed(0)
torch.manual_seed(0)

print("torch version", torch.__version__)
print("torchvision version", torchvision.__version__)

#Define paths and set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

root_dir = "./images/"
print("The data lies here =>", root_dir)


show = 1

p_val = 0.2 # %of train samples used to validate
DIM = [28,28]
EPOCHS = 30
LEARNING_RATE = 0.05

BATCH_SIZE = {
	"train": 100,
	"val": 2,
	"test": 2}

accuracy_stats = {
	'train': [],
	"val": []}

loss_stats = {
	'train': [],
	"val": []}



#Define transforms
image_transforms = {
	"train": transforms.Compose([
	transforms.Grayscale(num_output_channels=1),
	transforms.Resize((DIM[0],DIM[1])),
	transforms.ToTensor(),
	transforms.Normalize(mean=0.8717,std=0.2662)
	]),
	"test": transforms.Compose([
	transforms.Grayscale(num_output_channels=1),
	transforms.Resize((DIM[0],DIM[1])),
	transforms.ToTensor(),
	transforms.Normalize(mean=0.8717,std=0.2662)
	])
}

#### Initialize TRAIN/VAL/TEST Datasets ####
train_dataset = datasets.ImageFolder(root = root_dir + "train",transform = image_transforms["train"])
test_dataset = datasets.ImageFolder(root = root_dir + "test", transform = image_transforms["test"])


#Get Train and Validation Samples
train_dataset_size = int(len(train_dataset))
train_dataset_indices = list(range(train_dataset_size)) #from 0 to length of dataset

np.random.shuffle(train_dataset_indices) #shuffle
val_split_index = int(np.floor(p_val * train_dataset_size)) #find the index to split the list

train_idx, val_idx = train_dataset_indices[val_split_index:],train_dataset_indices[:val_split_index]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)


#### DATA LOADERS ####
train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE["train"], sampler=train_sampler, num_workers=4)
val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE["val"], sampler=val_sampler)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE["test"])

print(train_dataset)


#### INPUT DATA PLOTS ####
if not show:

	idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}

	print("---PRINT DATASET BREAKDOWN")
	helpers.plot_from_dict2(train_dataset,test_dataset,idx2class)
	
	print("TRAINING DATA")
	inputs, labels = next(iter(train_loader))
	helpers.plot_image(DIM,inputs[0],idx2class[labels[0].item()])
	helpers.plot_image_grid(DIM,BATCH_SIZE["train"], inputs, [idx2class[x.item()] for x in labels])

	print("VALIDATION DATA")
	inputs, labels = next(iter(val_loader))
	helpers.plot_image(DIM,inputs[0],idx2class[labels[0].item()])
	helpers.plot_image_grid(DIM,BATCH_SIZE["val"], inputs, [idx2class[x.item()] for x in labels])

	print("TEST DATA")
	inputs, labels = next(iter(test_loader))
	helpers.plot_image(DIM,inputs[0],idx2class[labels[0].item()])
	helpers.plot_image_grid(DIM,BATCH_SIZE["test"], inputs, [idx2class[x.item()] for x in labels])


#### NETWORK ARCHITECTURE ####
class Net(nn.Module):
	def __init__(self,dim):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# compute the size of the the input for the first fully connected layer
		# You can track what happens to a n-by-n images when passes through the previous layers
		# you will endup with 64 channels each of size x-by-x therefore 
		self.size_linear = 64*(dim)**2
		self.fc1 = nn.Linear(self.size_linear, 512)
		self.fc2 = nn.Linear(512, 2)
   
	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x))) 
		x = self.pool2(F.relu(self.conv2(x)))
		x = x.view(-1, self.size_linear) 
		x = self.fc1(x)
		x = self.fc2(x) 
		return x


#### TRAINING CLASS ####
class Trainer():
	def __init__(self,net=None, optim=None, loss_function=None, acc_function=None):
		self.net = net
		self.optim = optim
		self.loss_function = loss_function
		self.acc_function = acc_function

	def train(self,epochs):
		losses = []
		for epoch in range(epochs):
			epoch_loss = 0.0
			epoch_acc = 0.0
			epoch_steps = 0

			for data in train_loader:
				# Moving this batch to GPU
				# Note that X has shape (batch_size, number of channels, height, width)
				# which is equal to (256,1,28,28) since our default batch_size = 256 and 
				# the image has only 1 channel
				#print("hi")
				X = data[0].to(device)
				y = data[1].to(device)
				
				# Zero the gradient in the optimizer i.e. self.optim
				self.optim.zero_grad() 

				# Getting the output of the Network
				output = self.net(X)

				# Computing loss using loss function i.e. self.loss_function
				loss = self.loss_function(output,y)
				acc = self.acc_function(output, y)

				# compute gradients of parameteres (backpropagation)
				loss.backward() 

				# Call the optimizer i.e. self.optim
				self.optim.step()

				epoch_loss += loss.item()
				epoch_acc += acc.item()

				epoch_steps += 1

			with torch.no_grad():	
				net.eval()
				val_epoch_loss = 0.0
				val_epoch_acc = 0.0
				val_epoch_steps = 0


				for val_data in val_loader:

					val_X = val_data[0].to(device)
					val_y = val_data[1].to(device)

					val_output = self.net(val_X)

					val_loss = self.loss_function(val_output,val_y)
					val_acc = self.acc_function(val_output, val_y)

					val_epoch_loss += val_loss.item()
					val_epoch_acc += val_acc.item()

					val_epoch_steps += 1
			
			loss_stats['train'].append(epoch_loss/epoch_steps)
			loss_stats['val'].append(val_epoch_loss/val_epoch_steps)

			accuracy_stats['train'].append(epoch_acc/epoch_steps)	
			accuracy_stats['val'].append(val_epoch_acc/val_epoch_steps)	

			# average loss of epoch
			print("epoch [%d]: train loss %.3f, val loss %.3f: train acc %.3f, val acc %.3f"
				%(epoch+1,loss_stats['train'][-1],loss_stats['val'][-1], accuracy_stats['train'][-1], accuracy_stats['val'][-1]))

		return loss_stats, accuracy_stats


#### RUN TRAINING ####
dim_after_pool = int(DIM[0]/2/2)

net = Net(dim_after_pool)
a = net.to(device)

opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

def binary_acc(y_pred, y_test):
	y_pred_tag = torch.log_softmax(y_pred, dim = 1)

	_, y_pred_tags = torch.max(y_pred_tag, dim = 1)
	correct_results_sum = (y_pred_tags == y_test).sum().float()

	acc = correct_results_sum/y_test.shape[0]
	acc = torch.round(acc * 100)

	return acc

trainer  = Trainer(net=net, optim=opt, loss_function=loss_function, acc_function = binary_acc)


starttime = timeit.default_timer()
print("Begin Training...")
losses,accuracy = trainer.train(EPOCHS)
print("Training elapse time:", timeit.default_timer() - starttime)



#### OUTPUT DATA PLOTS ####
if show:
	helpers.plot_loss_accuracy_epoch(losses,accuracy)




#### SAVE MODEL ####
torch.save(net.state_dict(), './saved_model')

#for param in net.parameters():
#	print(param)



##### LOAD MODEL ####

print("LOAD MODEL")
model = Net(dim_after_pool)
model.load_state_dict(torch.load('./saved_model'))
model.eval()


print("EVALUATE MODEL")
y_pred_list = []
y_true_list = []
with torch.no_grad():
	for test_data in test_loader:

		X = test_data[0].to(device)
		y = test_data[1].to(device)

		output = model(X)
		output = torch.log_softmax(output, dim=1)
		_, y_pred = torch.max(output, dim = 1)

		y_pred_list.append(y_pred.cpu().numpy())
		y_true_list.append(y.cpu().numpy())

print(y_pred_list)
print(y_true_list)

y_pred_list = np.concatenate(y_pred_list).ravel()
y_true_list = np.concatenate(y_true_list).ravel()

print(y_pred_list)
print(y_true_list)

print(classification_report(y_true_list, y_pred_list))
print(confusion_matrix(y_true_list, y_pred_list))


print("done!!!!")





