# Voting ensemble with sklearn
This program uses 6 classifiers of sklearn library for training and voting.
We uses these classifiers:
- SGD
- Random Forest
- Decision Tree
- Bernoli NB
- SVM
- Extra Tree

This program calculates the confusion matrix for your test dataset. For this you should change the elements of the list labels=['pos','neg'] into your labels. It calculates also accuracy and precision and total time for creation of your model. The accuracy for our test dataset is 74%.
## Train and test datasets
To test this classifier we put to your disposal train and test datasets.
These files are some simples of files available [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/). We do not any preprocessing of train or test dataset.

## Installing and requirements
You need Python >=3.3 
You need NumPy (>= 1.8.2)
You need SciPy (>= 0.13.3)

You need to install sklearn library
```
 pip install -U scikit-learn
```
## How to use

Usage : voting_ensemble_sklearn.py
You should only fill the paths. You can put one tweet par line in your TRAIN_TWEETS_PATH and TEST_TWEETS_PATH.
```
TRAIN_DATASET_PATH=''
TRAIN_LABELS_PATH=''

TEST_DATASET_PATH=''
TEST_LABELS_PATH=''

PRED_LABELS_PATH=''
MODEL_PATH=''
```
You can put the labes of these datasets in a seperated files. Each line one label. Each line of TRAIN_LABELS_PATH corresponds to each line of TRAIN_DATASET_PATH and the same for test datstet.
in PRED_LABELS_PATH you put the path where you want to save the predictions. In the MODEL_PATH you can save your model.
