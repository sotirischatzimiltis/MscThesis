#!pip install scikit-learn==0.23.2

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import seaborn as sn

"""# Dataset"""
###### PATHS TO THE CSV FILES ####################
train_Dataset = 'C:/Users/sotir/Desktop/Masters/Dissertation/DATA/MulticlassClassification/StandarScaler/PCA/train_pca_multiclass.csv'
test_Dataset_full= 'C:/Users/sotir/Desktop/Masters/Dissertation/DATA/MulticlassClassification/StandarScaler/PCA/test_pca_multiclass.csv'

"""# Prepare Data"""
##### READ CSV FILES ##########################
traindata = pd.read_csv(train_Dataset)
testdata = pd.read_csv(test_Dataset_full)

#### SPLIT DATA INTO FEATURES AND LABELS #######
y_train = traindata['label']
x_train = traindata.drop('label',axis=1)

y_test = testdata['label']
x_test = testdata.drop('label',axis=1)


"""
# Metrics function
FUNCTION THAT COMPUTES AND OUTPUTS THE PERFORMANCE BASED ON A SPECIFIC DATASET
"""
def performance(y_pred,y_test):
    correct = (y_pred==y_test).sum()
    total = len(y_pred)
    classes = ('Normal','DoS','Probe','R2L','U2R')
    print('\nConfusion Matrix\n')
    cf_matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                     columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    print(cf_matrix)
    print(f"Total correctly classified instances: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=classes))

    print('\nConfusion Matrices for every class type\n')
    for i in range(0,len(classes)):
        print(classes[i])
        tp = cf_matrix[i,i]
        fp = cf_matrix[0,i] + cf_matrix[1,i] + cf_matrix[2,i] + cf_matrix[3,i] + cf_matrix[4,i] - cf_matrix[i,i]
        fn = cf_matrix[i,0] + cf_matrix[i,1] + cf_matrix[i,2] + cf_matrix[i,3] + cf_matrix[i,4] - cf_matrix[i,i]
        tn = np.sum(np.array(cf_matrix)) - tp - fp -fn
        cf_bin = np.array([tp,fp,fn,tn]).reshape((2,2))
        print(cf_bin)
        precision = (tp / (tp + fp))
        recall = (tp / (tp + fn))
        print(f"False alarm rate: {100 * (fp / (fp + tn)):.2f}%")
        print(f"Detection rate: {100 * (tp / (tp + fn)):.2f}%")
        print(f"Precision: {100 * precision:.2f}%")
        print(f"Recall: {100 * recall:.2f}%")
        print(f"F1-score: {100 * ((2 * precision * recall) / (precision + recall)):.2f}%")
        print(f"False positive rate: {100 * (fp / (fp + tp)):.2f}%")
        print(f"False negative rate: {100 * (fn / (fn + tn)):.2f}%")

"""# Decision Tree model"""

parameters = {'splitter': ['best','random'],
              'min_samples_leaf': [1,2],
              'max_features':["auto","sqrt","log2"]
              }

grid_DT = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid = parameters, cv = 2, n_jobs=-1,verbose=2)
grid_DT.fit(x_train,y_train)
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_DT.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_DT.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_DT.best_params_)

clf_dt = DecisionTreeClassifier(**grid_DT.best_params_) # create decision tree classifier
clf_dt = clf_dt.fit(x_train,y_train) # train DT classifier
y_pred = clf_dt.predict(x_test) # predict for whole test dataset
print("WHOLE DATASET")
performance(y_pred,y_test)


"""# Random Forest Model"""

parameters = {
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6, 8, 10],
    'n_estimators': [100, 120]
}

grid_RF = GridSearchCV(estimator=RandomForestClassifier(), param_grid = parameters, cv = 2, n_jobs=-1,verbose =2)
grid_RF.fit(x_train,y_train)
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_RF.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_RF.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_RF.best_params_)

clf_rf = RandomForestClassifier(**grid_RF.best_params_)
clf_rf = clf_rf.fit(x_train,y_train)
y_pred = clf_rf.predict(x_test)
print("WHOLE DATASET")
performance(y_pred,y_test)

# """# MLP Classifier Model"""
# parameters = {
#   'solver': ['sgd','adam'], #lbfgs
#   'hidden_layer_sizes':[(100,),(100,100,),(1000,),(1000,1000,)]
#  }
#
# grid_MLP = GridSearchCV(estimator=MLPClassifier(), param_grid = parameters, cv = 2, n_jobs=-1,verbose =5)
# grid_MLP.fit(x_train,y_train)
# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n",grid_MLP.best_estimator_)
# print("\n The best score across ALL searched params:\n",grid_MLP.best_score_)
# print("\n The best parameters across ALL searched params:\n",grid_MLP.best_params_)
#
print("Training")
clf_mlp = MLPClassifier((1000,1000,),solver='adam',verbose=1)
#clf_mlp = MLPClassifier(**grid_MLP.best_params_)
clf_mlp = clf_mlp.fit(x_train,y_train)
y_pred = clf_mlp.predict(x_test)
print("WHOLE DATASET")
performance(y_pred,y_test)

# """# Gradient Boosting Classifier"""

parameters = {'learning_rate': [0.05,0.01],
              'subsample'    : [0.5, 0.2],
              'n_estimators' : [200,500],
              'max_depth'    : [4,6]
              }

grid_GBC = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid = parameters, cv = 2, n_jobs=-1,verbose =5)
grid_GBC.fit(x_train,y_train)
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_GBC.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_GBC.best_params_)

clf_gbc = GradientBoostingClassifier()
clf_gbc = clf_gbc.fit(x_train,y_train)
y_pred = clf_gbc.predict(x_test)
print("WHOLE DATASET")
performance(y_pred,y_test)

# """# K-Nearest Neighbors"""

parameters = {
    'n_neighbors': [3,5],
    'weights': ['uniform','distance'],
    'p': [1,2]
}

grid_KNN = GridSearchCV(estimator= KNeighborsClassifier(), param_grid = parameters, cv = 2, n_jobs=-1,verbose =5)
grid_KNN.fit(x_train,y_train)
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_KNN.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_KNN.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_KNN.best_params_)

clf_knn = KNeighborsClassifier()
clf_knn = clf_knn.fit(x_train,y_train)
y_pred = clf_knn.predict(x_test)
print("WHOLE DATASET")
performance(y_pred,y_test)
