Folders:
-data: contains data used for model development and validation
	-q_curve_28_419_cell_1.txt to q_curve_28_419_cell_8.txt: the input data.

-models: contains trained models
	example_trained_dnn.hdf5: A pretrained model as an example for method evaluation.


Python files:
-oxford_train.py: the python file for model development. It is used to train models.
-oxford_evaluation.py: the python file for model evaluation. It is used to make predictions and compuate capacity and energy errors. Note similar evaluation can be easily implemented for other datasets.

