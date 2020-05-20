import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#### WITH DROPOUT ####
class Model1(nn.Module):
	def __init__(self,prms):
		super(Model1, self).__init__()
		print("\nCREATING NEURAL NET MODEL")

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# compute the size of the the input for the first fully connected layer
		# You can track what happens to a n-by-n images when passes through the previous layers
		# you will endup with 64 channels each of size x-by-x therefore 
		self.size_linear = self.calc_size(64,prms.DIM)

		self.fc1 = nn.Linear(self.size_linear, 512)
		self.fc2 = nn.Linear(512, 2)

		#Dropout
		self.dropout = nn.Dropout(p=prms.DROP)

		print(self)
	
	def calc_size(self, out_channels, start_dims):
		d = start_dims[0]/(2*2)
		return int(out_channels*d*d)

	def forward(self, x):
		x = self.dropout(self.pool1(F.relu(self.conv1(x))))
		x = self.dropout(self.pool2(F.relu(self.conv2(x))))
		x = x.view(-1, self.size_linear) 
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x) 
		return x

#### WITHOUT DROPOUT ####
class Model2(nn.Module):
	def __init__(self,prms):
		super(Model2, self).__init__()
		print("\nCREATING NEURAL NET MODEL")

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# compute the size of the the input for the first fully connected layer
		# You can track what happens to a n-by-n images when passes through the previous layers
		# you will endup with 64 channels each of size x-by-x therefore 
		self.size_linear = self.calc_size(64,prms.DIM)

		self.fc1 = nn.Linear(self.size_linear, 512)
		self.fc2 = nn.Linear(512, 2)

		print(self)
	
	def calc_size(self, out_channels, start_dims):
		d = start_dims[0]/(2*2)
		return int(out_channels*d*d)

	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = x.view(-1, self.size_linear) 
		x = F.relu(self.fc1(x))
		x = self.fc2(x) 
		return x