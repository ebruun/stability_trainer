import numpy as np

from torchvision import transforms
import torch

class Input():

	def __init__(self):
		print("\nSETTING VARIABLES")

		#Define paths and set GPU
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("--We're using =>", self.device)

		self.root_dir = "./images/"
		print("--The data lies here =>", self.root_dir)

		self.OUT_PATH = "./trained_models/"

		np.random.seed(0)
		torch.manual_seed(0)

		self.VAL_SPLIT = 0.1 # %of train samples used to validate
		self.DROP = 0.4 # %to dropout
		self.DIM = [32,32]
		self.EPOCHS = 25
		self.LEARNING_RATE = 0.010 *(128/256)*2
		self.MOM = 0.9

		self.BATCH_SIZE = {
			"train": 128,
			"val": 128,
			"test": 128}


		#Define transforms
		self.image_transforms = {
			"train": transforms.Compose([
			transforms.Grayscale(num_output_channels=1),
			transforms.Resize((self.DIM[0],self.DIM[1])),
			transforms.ToTensor(),
			transforms.Normalize(mean=0.8919,std=0.2316)
			]),
			"test": transforms.Compose([
			transforms.Grayscale(num_output_channels=1),
			transforms.Resize((self.DIM[0],self.DIM[1])),
			transforms.ToTensor(),
			transforms.Normalize(mean=0.9016,std=0.2083)
			])
		}

		#Define transforms
		self.image_transforms_nonorm = {
			"train": transforms.Compose([
			transforms.Grayscale(num_output_channels=1),
			transforms.Resize((self.DIM[0],self.DIM[1])),
			transforms.ToTensor()
			]),
			"test": transforms.Compose([
			transforms.Grayscale(num_output_channels=1),
			transforms.Resize((self.DIM[0],self.DIM[1])),
			transforms.ToTensor()
			])
		}

