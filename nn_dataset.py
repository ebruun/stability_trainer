import numpy as np

import torch
from torchvision import datasets
from torch.utils.data import  DataLoader, SubsetRandomSampler

import helpers


#### TRAINING CLASS ####
class Data():
	def __init__(self, prms, find_normalize = False, plot_dist = False, plot_image =False):
		print("\nREADING DATA IN")

		self.p = prms
		
		if find_normalize:
			print("(find normalize run)")
			t = self.p.image_transforms_nonorm
			self.set_train, self.set_test = self.data_in(t)
			self.loader_train, self.loader_val, self.loader_test = self.create_loaders()

			print("Training: MEAN/STD", self.online_mean_and_sd(self.loader_train))
			print("Test: MEAN/STD", self.online_mean_and_sd(self.loader_test))
			print("Update the input normalization values...")
			exit()
		else:
			print("(regular run)")
			t = self.p.image_transforms
			self.set_train, self.set_test = self.data_in(t)
			self.loader_train, self.loader_val, self.loader_test = self.create_loaders()

		idx2class = {v: k for k, v in self.set_train.class_to_idx.items()}

		if plot_image:
			print("--plotting images")
			self.plot_data(self.loader_train, idx2class, "train")
			self.plot_data(self.loader_val, idx2class, "val")
			self.plot_data(self.loader_test, idx2class, "test")
		else:
			print("--no plotting images")


		if plot_dist:
			print("--plotting distribution")
			a = []
			self.plot_dist(self.set_train, self.set_test, idx2class)
		else:
			print("--no plotting distribution")


		print("\n",self.set_train)
		print("\n",self.set_test)






	def data_in(self, t):
		print("--Reading Training Data In")
		train_dataset = datasets.ImageFolder(root = self.p.root_dir + "train",transform = t["train"])

		print("--Reading Test Data In")
		test_dataset = datasets.ImageFolder(root = self.p.root_dir + "test", transform = t["test"])

		return train_dataset, test_dataset



	def create_samplers(self):
		#Get Train and Validation Samples
		train_dataset_size = int(len(self.set_train))
		train_dataset_indices = list(range(train_dataset_size)) #from 0 to length of dataset

		np.random.shuffle(train_dataset_indices) #shuffle
		val_split_index = int(np.floor(self.p.VAL_SPLIT * train_dataset_size)) #find the index to split the list

		train_idx, val_idx = train_dataset_indices[val_split_index:],train_dataset_indices[:val_split_index]

		train_sampler = SubsetRandomSampler(train_idx)
		val_sampler = SubsetRandomSampler(val_idx)

		return train_sampler, val_sampler


	def create_loaders(self):
		train_sampler, val_sampler = self.create_samplers()

		#subsetsampler already shuffles the data each epoch
		train_loader = DataLoader(dataset=self.set_train, shuffle=False, batch_size=self.p.BATCH_SIZE["train"], sampler=train_sampler, num_workers=4)

		val_loader = DataLoader(dataset=self.set_train, shuffle=False, batch_size=self.p.BATCH_SIZE["val"], sampler=val_sampler, num_workers=4)

		test_loader = DataLoader(dataset=self.set_test, shuffle=False, batch_size=self.p.BATCH_SIZE["test"], num_workers=4)
		
		return train_loader, val_loader, test_loader


	def online_mean_and_sd(self,loader):
		"""Compute the mean and sd in an online fashion

			Var[x] = E[X^2] - E^2[X]
		"""
		cnt = 0
		fst_moment = torch.empty(3)
		snd_moment = torch.empty(3)

		for images, _ in loader:

			b, c, h, w = images.shape
			nb_pixels = b * h * w
			sum_ = torch.sum(images, dim=[0, 2, 3])
			sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
			fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
			snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

			cnt += nb_pixels

		return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


	def plot_data(self, loader, idx2class, name):
		print("-- --", name)
		inputs, labels = next(iter(loader))
		helpers.plot_image(self.p.DIM,inputs[0],idx2class[labels[0].item()])
		helpers.plot_image_grid(self.p.DIM,self.p.BATCH_SIZE[name], inputs, [idx2class[x.item()] for x in labels])

	def plot_dist(self, train_dataset, test_dataset, idx2class):
		helpers.plot_from_dict2(train_dataset,test_dataset,idx2class)
