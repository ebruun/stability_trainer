import timeit

import pandas as pd

import helpers
import nn_model
import nn_trainer
import nn_input
import nn_dataset
import nn_evaluation


def training(prms,data, name):
	model = nn_model.Model1(prms).to(prms.device)

	trainer = nn_trainer.Trainer(model, prms)

	print("Begin Training...")
	starttime = timeit.default_timer()
	losses, accuracy = trainer.train(data.loader_train, data.loader_val)
	print("Training elapse time:", timeit.default_timer() - starttime)

	#trainer.plot_training(losses, accuracy)
	trainer.save_model(name)

	return model, losses, accuracy

def evaluation(prms,data, name = False, model = False):
	if model:
		evaluation = nn_evaluation.Evaluation(model,prms)
	else:
		output_model = nn_model.Model1(prms).to(prms.device)
		evaluation = nn_evaluation.Evaluation(output_model,prms)
		evaluation.load_model(name)

	evaluation.evaluate_model(data.loader_test)


if __name__ == "__main__":

	prms = nn_input.Input()
	data = nn_dataset.Data(prms, plot_dist=True)

	#data = nn_dataset.Data(prms)
	#data = nn_dataset.Data(prms, plot_image = True)
	#data = nn_dataset.Data(prms, find_normalize=True)

	runs = [[64,64]]

	for run in runs:

		name = "model_dim_" + str(run)

		#trained_model, l, a = training(prms,data, name)

		#evaluation(prms,data, model = trained_model)
		evaluation(prms,data, name = name)

		#df = pd.DataFrame(data={"train_loss": l['train'], "train_acc": a['train'], "val_loss": l['val'], "val_acc": a['val']})
		#df.to_csv("./saved_data/"+name+".csv", sep=',',index=False)

	

