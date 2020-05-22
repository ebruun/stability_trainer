import os
import numpy as np

import torch

import helpers

from sklearn.metrics import classification_report, confusion_matrix

class Evaluation():
	def __init__(self, output_model, prms):
		print("\nOUTPUT")
		self.p = prms

		self.model = output_model

	def load_model(self,name):
		print("--loading saved model")
		self.model.load_state_dict(torch.load(os.path.join(self.p.OUT_PATH,name)))

	def evaluate_model(self,test_loader):
		print("--evaluate on test data")

		y_pred_list = []
		y_true_list = []

		with torch.no_grad():
			self.model.eval()

			for test_data in test_loader:

				X = test_data[0].to(self.p.device)
				y = test_data[1].to(self.p.device)

				output = self.model(X)
				output = torch.log_softmax(output, dim=1)
				_, y_pred = torch.max(output, dim = 1)

				self.plot_incorrect(X,y,y_pred,output)

				y_pred_list.append(y_pred.cpu().numpy())
				y_true_list.append(y.cpu().numpy())


		y_pred_list = np.concatenate(y_pred_list).ravel()
		y_true_list = np.concatenate(y_true_list).ravel()

		print(classification_report(y_true_list, y_pred_list))
		print(confusion_matrix(y_true_list, y_pred_list))


	def plot_incorrect(self,X,y,y_pred,output):
		idx_wrong = (y != y_pred).nonzero()

		output = torch.exp(output)
		name = []
		for idx in idx_wrong:
			if y[idx] == 0:
				num = output[idx][0][1]
				name.append('IS stable --> BUT {:.0f}% unstable'.format(100*num))
			else:
				num = output[idx][0][0]
				name.append('IS unstable --> BUT {:.0f}% stable'.format(100*num))

		#helpers.plot_image(self.p.DIM,X[idx_wrong[0]],name[0])
		helpers.plot_image_grid(self.p.DIM,len(idx_wrong), X[idx_wrong], name)

