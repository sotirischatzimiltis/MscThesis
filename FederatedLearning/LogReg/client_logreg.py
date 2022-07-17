import warnings
import flwr as fl
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from pathlib import Path

import utils

if __name__ == "__main__":
    train_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/Binary_Classification/StandardScaler/All_features/train_binary.csv"
    df=pd.read_csv(train_Dataset)
    X_train=df.iloc[:,:-1].values
    y_train=df.iloc[:,-1].values
    
    test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/Binary_Classification/StandardScaler/All_features/test_binary.csv"
    df=pd.read_csv(test_Dataset)
    X_test=df.iloc[:,:-1].values
    y_test=df.iloc[:,-1].values
   
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return list(utils.get_model_parameters(model)), len(X_train),{}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=FlowerClient(), root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),)