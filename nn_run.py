import timeit

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

	trainer.plot_training(losses, accuracy)
	trainer.save_model(name)

	return model

def evaluation(prms,data, name, model = False):
	if model:
		evaluation = nn_evaluation.Evaluation(model,prms)
	else:
		output_model = nn_model.Model1(prms).to(prms.device)
		evaluation = nn_evaluation.Evaluation(output_model,prms)
		evaluation.load_model(name)

	evaluation.evaluate_model(data.loader_test)


if __name__ == "__main__":

	prms = nn_input.Input()
	#data = nn_dataset.Data(prms)
	#data = nn_dataset.Data(prms, plot_image = True)
	#data = nn_dataset.Data(prms, find_normalize=True)

	exit()
	name = "model3"
	trained_model = training(prms,data, name)

	evaluation(prms,data, name, model = trained_model)
	#evaluation(prms,data, name)

	

