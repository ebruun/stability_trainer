import os

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import helpers

#### TRAINING CLASS ####
class Trainer():
	def __init__(self,model, prms):
		print("\nCREATE THE TRAINER")

		self.p = prms

		self.model = model
		self.optim = optim.SGD(model.parameters(), lr=self.p.LEARNING_RATE, momentum=self.p.MOM)
		self.loss_function = nn.CrossEntropyLoss()

		self.accuracy_stats = {
			'train': [],
			'val': []}

		self.loss_stats = {
			'train': [],
			'val': []}

	def binary_acc(self,y_pred, y_test):
		y_pred_tag = torch.log_softmax(y_pred, dim = 1)

		_, y_pred_tags = torch.max(y_pred_tag, dim = 1)
		correct_results_sum = (y_pred_tags == y_test).sum().float()

		acc = correct_results_sum/y_test.shape[0]
		acc = torch.round(acc * 100)

		return acc

	#### SAVE MODEL ####
	def save_model(self, name):
		torch.save(self.model.state_dict(), os.path.join(self.p.OUT_PATH,name))


	#### TRAIN MODEL ####	
	def train(self, train_loader, val_loader):
		losses = []
		for epoch in range(self.p.EPOCHS):
			epoch_loss = 0.0
			epoch_acc = 0.0

			self.model.train()
			for ii, data in enumerate(train_loader):
				# Note that X has shape (batch_size, number of channels, height, width)
				# the image has only 1 channel
				X = data[0].to(self.p.device)
				y = data[1].to(self.p.device)
				
				# Zero the gradient in the optimizer i.e. self.optim
				self.optim.zero_grad() 

				# Getting the output of the Network
				output = self.model(X)

				# Computing loss using loss function i.e. self.loss_function
				loss = self.loss_function(output,y)
				acc = self.binary_acc(output, y)

				# compute gradients of parameteres (backpropagation)
				loss.backward() 

				# Call the optimizer i.e. self.optim
				self.optim.step()

				epoch_loss += loss.item()
				epoch_acc += acc.item()

			with torch.no_grad():	
				self.model.eval()

				val_epoch_loss = 0.0
				val_epoch_acc = 0.0

				for val_data in val_loader:
					val_X = val_data[0].to(self.p.device)
					val_y = val_data[1].to(self.p.device)

					val_output = self.model(val_X)

					val_loss = self.loss_function(val_output,val_y)
					val_acc = self.binary_acc(val_output, val_y)

					val_epoch_loss += val_loss.item()
					val_epoch_acc += val_acc.item()


			self.loss_stats['train'].append(epoch_loss/len(train_loader))
			self.loss_stats['val'].append(val_epoch_loss/len(val_loader))

			self.accuracy_stats['train'].append(epoch_acc/len(train_loader))	
			self.accuracy_stats['val'].append(val_epoch_acc/len(val_loader))	

			# average loss of epoch
			print("epoch [%d]: train loss %.3f, val loss %.3f: train acc %.3f, val acc %.3f"
				%(epoch+1,self.loss_stats['train'][-1],self.loss_stats['val'][-1], self.accuracy_stats['train'][-1], self.accuracy_stats['val'][-1]))

		return self.loss_stats, self.accuracy_stats

	def plot_training(self, losses,accuracy):
		helpers.plot_loss_accuracy_epoch(losses,accuracy)



