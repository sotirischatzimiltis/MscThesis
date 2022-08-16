import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
from sklearn.metrics import confusion_matrix 
from pathlib import Path

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""
    test_Dataset = "C:/Users/sotir/Desktop/Masters/Dissertation/DATA/Binary_Classification/StandardScaler/All_features/test_binary.csv"
    df=pd.read_csv(test_Dataset)
    X_test=df.iloc[:,:-1].values
    y_test=df.iloc[:,-1].values

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test)) # calculate loss 
        y_pred = model.predict(X_test) # make prediction 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # get confusion matrix 
        precision = 100 * (tp / (tp + fp))
        recall = 100 * (tp / (tp + fn)) 
        far = 100 * (fp / (fp + tn)) # false alarm rate %
        dr = 100 * (tp / (tp + fn)) # detection rate % 
        f1_score = 100 * ((2 * precision * recall) / (precision + recall)) # f1 score %
        fpr = 100 * (fp / (fp + tp)) # false positive rate
        fnr = 100 * (fn / (fn + tn)) # false negative rate
        accuracy = model.score(X_test, y_test)
        correct = (y_test == y_pred).sum()
        total = len(y_test)
        acc = correct/total
        return loss, {"accuracy": accuracy, "precision":precision,"recall":recall,"f1_score":f1_score,"detection_rate":dr,
        "false_alarm_rate":far,"false_positive_rate":fpr,"false_negative_rate":fnr,"true_negatives":tn,"false_positives":fp,
        "false_negatives":fn,"true_positives":tp,"acc":acc}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,fraction_fit=1,min_fit_clients=3,min_available_clients=3,min_eval_clients=3,)
    fl.server.start_server(erver_address="localhost:8080", config={"num_rounds": 15}, strategy=strategy)
#     fl.server.start_server(
#     server_address="localhost:8080",
#     config={"num_rounds": 15},
#     strategy=strategy,
#     certificates=(
#         Path(".cache/certificates/ca.crt").read_bytes(),
#         Path(".cache/certificates/server.pem").read_bytes(),
#         Path(".cache/certificates/server.key").read_bytes(),
#     )
# )
    
