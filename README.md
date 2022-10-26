# Higgs Boson Challenge 2022 (EPFL)
In the repository you can find the code we used for [Higgs Boson Challenge 2022](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), proposed as project 1 in ML course at EPFL [(CS-433)](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/). 

## Team:
Our team (named `FGRxML`) is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)  
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)   

With an accuracy of 0.836 we obtained the n-th place out of more than 200 teams.

# Project pipeline

## Environment:
We worked with `python3.8.5`. The library we used to compute our ML model is `numpy` and the library we used for visualization is `matplotlib`.

## Description of notebooks:
Here you can find what each of the file in the repo does. The order in which we describe them follows the pipeline we used to obtain our results.
- `helpers.py`: we implemented all the "support" functions we used in others .py files.
- `gradients.py`: we implemented all the gradients used in the implementations of the 6 methods.
- `costs.py`: we implemented all the cost functions used in the implementations of the 6 methods.
- `implementations.py`: we implemented all the 6 methods.
- `preprocessing.py`: we implemented some functions we used to process the data. The majority of them were used in `features_engineering.ipynb`
- `features_engineering.ipynb`: In this notebook we made feature engineering (feature selection, feature transformation,...).
- `crossvalidation.py`: we implemented the functions used to train the best hyperparameters for our model (ridge regression).
- `choosing_hyperparameters.ipynb`: we trained the best hyperparameters for our model (ridge regression).
- `dataset_splitting.py`: we implemented some functions useful to have a local computation of accuracy of our model before the submission.
- `main.py`: it returns our predictions after using the selected model to predict test data.


