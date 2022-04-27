# Important information

The experiment was run with Python 3.8.10, on Ubuntu 20.04.

# Required libraries

numpy==1.19.5,
pandas==1.1.4,
fire==0.4.0,
matplotlib==3.1.3,
tensorboard==2.5.0,
tensorflow==2.3.0.

# How to run the experiment

The following command will run the experiment :
```bash
python3 -m main.experiment
```
We first load all the training and validation data.
We then runs a base model against the training data for a specified 
number of epochs.
We further train a model for each individual starting from the base model,
by fine-tuning it against the individual's specific data.
Finally, the results are generated in the "result" directory.

# Model training plots

Run tensorboard at the logs directory and connect to localhost:6006 to
observe the loss evolution over the training iterations:
```bash
tensorboard --logdir logs
```


# Further information

The config folder provides config files for each component used for the
experiment:

## models/
model configurations

## main.ini

### [Predictor]
**model** ==> which model configuration to use

**dense_activation** ==> which activation function to use for dense 
layers (relu...)

**ckpt_path** ==> checkpoint path

### [Experiment]

**log_freq** ==> frequency at which we log training stats

**batch_size** ==> Number of trajectories considered for each training step

**trajectory_length** ==> length of the trajectory

**n_epoch** ==> number of epochs

**n_step_predict** ==> How much in the future we learn to predict (n x 5 mins)

**learning_rate** ==> learning rate used to train the model (with the Adam optimizer)

**result_path** ==> Path where we save the results

**gpu** ==> Which GPU to use. -1 if no GPU.

### [DataBag]

**data_path** ==> Path for the dataset
