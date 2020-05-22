import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import numpy as np

import torch


def get_class_distribution(dataset_obj,idx2class):
	count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()} #initialize count

	for _, label_id in dataset_obj:
		label = idx2class[label_id]
		count_dict[label] += 1
		
	return count_dict


def plot_from_dict2(train_dataset,test_dataset,idx2class,  **kwargs):
	plt.figure(figsize=(10,15))

	plt.subplot(2, 1, 1)
	dict_obj = get_class_distribution(train_dataset,idx2class)
	sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", **kwargs).set_title('Entire Dataset (TRAINING)')

	plt.subplot(2, 1, 2)
	dict_obj = get_class_distribution(test_dataset,idx2class)
	sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", **kwargs).set_title('Entire Dataset (TEST)')

	plt.show()


def plot_image(d,image, label):
	image = image.numpy()
	image = image.reshape(d[0],d[1])

	plt.figure()
	plt.imshow(image,cmap='gray')
	plt.xlabel(label)

	plt.colorbar()
	plt.grid(False)
	plt.show()
	
def plot_image_grid(d,num_images, images, labels):
	images = images.numpy()
	images = images.reshape(num_images,d[0],d[1])

	x_num = int(np.trunc(math.sqrt(num_images)))
	r = num_images - x_num**2
	add = math.ceil(r/x_num)

	plt.figure(figsize=(10,10))
	for i in range(num_images):
		plt.subplot(x_num+add, x_num, i+1)
		plt.xticks([])
		plt.yticks([])

		plt.imshow(images[i], cmap='gray')
		plt.xlabel(labels[i])

		plt.grid(False)
	plt.show()



def plot_loss_accuracy_epoch(loss_stats,accuracy_stats):
	train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
	train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})


	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))

	sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
	sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

	plt.show()



