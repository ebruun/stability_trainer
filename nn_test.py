import torch
import torchvision
import torch.nn as nn


DIM = [4,4]

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

#### LOAD MODEL ####
dim_after_pool = int(DIM[0]/2/2)

model = Net(dim_after_pool)

model.load_state_dict(torch.load('./saved_model'))
model.eval()

#for param in model.parameters():
	#print(param.data)

#print(model)

y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)        y_test_pred = model(x_batch)        y_test_pred = torch.log_softmax(y_test_pred, dim=1)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())