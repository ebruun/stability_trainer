import os
import numpy as np

import torch

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

				y_pred_list.append(y_pred.cpu().numpy())
				y_true_list.append(y.cpu().numpy())


		y_pred_list = np.concatenate(y_pred_list).ravel()
		y_true_list = np.concatenate(y_true_list).ravel()

		print(classification_report(y_true_list, y_pred_list))
		print(confusion_matrix(y_true_list, y_pred_list))
