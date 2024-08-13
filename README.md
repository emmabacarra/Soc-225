## Important Notes

functions.py is a Python script developed by me as a method of utilizing convenience functions to streamline the process of model training and evaluation. The main class used during training sessions is the experiment class, indicated by `from functions import experiment` at the top of each training file. `SignalDataset` is a second convenience function used to transform the provided dataset from h5 file format to the Dataloader format compatible with the Pytorch modules.

In a similar fashion, the model architecture is designed in its own Python script, and imported to each training file with `from model import SignalClassifier` at the top.

There are four variations of training files:
1. `1-cnn.ipynb`
2. `3.2-cnn.ipynb`
3. `5-cnn.ipynb`
4. `9-cnn.ipynb`

These models consist of various settings, notably either the selected optimizing algorithm and/or the learning rate at which the model trains with this optimizer. Other models made during this process but are not utilized in the presentation are filed into the Discarded Models folder.

### <b>Please consider model 9 as the final iteration of training sessions.</b>