# Importing Packages

import numpy as np
import seaborn as sns
from imblearn.over_sampling import ADASYN
from matplotlib import pyplot as plt, pyplot
from numpy import mean
from numpy import std
from pandas import read_csv
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import lightgbm as lgb

# Set dataset location
full_path = 'oil-spill.csv'

# load csv file as data frame
df = read_csv(full_path, header=None)

# Dataset dimensions
print('Dimension du Dataset:', df.shape)


print()
print(df.head(5))
print()

# Create a histogram of each variable
ax = df.hist()

# Disable axis labels
for axis in ax.flatten():
	axis.set_title('')
	axis.set_xticklabels([])
	axis.set_yticklabels([])
pyplot.show()


# See the distribution of [0 and 1] classes
target = df.values[:, -1]
counter = Counter(target)
for k, v in counter.items():
    per = v / len(target) * 100
    print('Classe= %d, Count= %d, Pourcentage= %.3f%%' % (k, v, per))
# print(df.columns) -- Afficher les caractéristiques des images.
print()

# Data preprocessing
def charger_dataset(filename):
    # loads the dataset as a numpy array
    data = read_csv(filename, header=None)
    # delete unused columns
    data.drop(22, axis=1, inplace=True)
    data.drop(0, axis=1, inplace=True)
    # retrieve the numpy array
    data = data.values
    # Divided into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # encodes target variable labels to have classes 0 and 1 using the sklearn LabelEncoder class
    y = LabelEncoder().fit_transform(y)
    return X, y

# Define a model evaluation function
def evaluate_model(X, y, model):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model evaluation metrics
    metric = make_scorer(geometric_mean_score)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring= metric, cv=cv, n_jobs=-1)
    return scores


# ---- > MAIN PROGRAM

# View dataset imbalance in target 49 column.
plt.figure(figsize=(8, 6))
sns.countplot(x=df.values[:, -1], palette="rocket")
plt.xticks([0, 1], ['Non-Déversement', 'Déversement'], fontsize=15)
plt.show()

# df[49].value_counts().plot.pie()
# pyplot.show()

# Create the model (Data normalization and Algorithm)



estimateur = [('t1', MaxAbsScaler()),

            ('m', lgb.LGBMClassifier(n_estimators=362, num_leaves=1208, min_child_samples=8,
            learning_rate=0.02070742242160566, colsample_bytree=0.37915528071680865,
            reg_alpha=0.002982599447751338, reg_lambda= 1.136605174453919))]
model = Pipeline(estimateur)

# Test the model
X, y = charger_dataset(full_path)

# summarize the loaded data set
print('Dimension du dataset après preprocessing:', X.shape, 'Distribution des classes:', Counter(y))
print()

# Visualize class distribution after Overfitting

# transform the dataset using the ADASYN oversampling method.
oversample = ADASYN()
X_sam, y_sam = oversample.fit_resample(X, y)

print("Ensemble de données rééchantillonné après la forme ADASYN ")
# trace distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y_sam, palette="rocket")
plt.xticks([0, 1], ['Non-Déversement', 'Déversement'], fontsize=15)
plt.show()

# See the distribution of classes [0 and 1].
counter = Counter(y_sam)
for k, v in counter.items():
    per = v / len(y) * 100
    print('Classe= %d, Count= %d, Pourcentage= %.3f%%' % (k, v, per))


# Evaluate our final model.

# Divide the data set into Train Set and Test Set.
X_train, X_test, y_train, y_test = train_test_split(X_sam, y_sam, test_size=0.2, stratify=y_sam, random_state=1)

'''
from flaml import AutoML
automl = AutoML()
automl.fit(X_sam, y_sam, task="classification")
params = automl.fit(X_sam, y_sam, task="classification", estimator_list=["lgbm"])
print('Best hyperparmeter config:', automl.best_config)
'''

scores = evaluate_model(X_sam, y_sam, model)
print('Score final d’évaluation du modèle')
print('Moyenne de G-Mean des scores des 10 plis: %.3f - Ecart type: (%.3f)' % (mean(scores), std(scores)))
model.fit(X_sam, y_sam)

print()

# Evaluate the model over the entire 10% TestSet
test_scores = evaluate_model(X_test, y_test, model)

print('Mean G-Mean du TestSet: %.3f - Ecart type: (%.3f)' % (mean(test_scores), std(test_scores)))

print()

# Classification report
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print()
'''
# Draw the confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice de Confusion', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matrice de Confusion Normalisée")
    else:
        print('Matrice de confusion, Non Normalisée')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label reel')
    plt.xlabel('Label predit')


cnf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['- VE', '+ VE']
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Matrice de confusion Normalisée')
plt.show()
'''

# --------------------------------------------------- #

# Make predictions based on new data

# Load dataset
def charger_data(filename):
    # loads the dataset as a numpy array
    data = read_csv(filename, header=None)
    # Retrieve the numpy array
    data.drop(0, axis=1, inplace=True)
    data = data.values
    # Divided into input and output elements
    X, y = data[:, :-1], data[:, -1]
    y = LabelEncoder().fit_transform(y)
    return X, y

X, y = charger_data(full_path)
# Adapt the model
model.fit(X, y)
# evaluate some cases Non-spill (class 0 known)
print()
print('Cas sans déversement:')
img = [[329,1627.54,1409.43,51,822500,35,6.1,4610,0.17,178.4,0.2,0.24,0.39,0.12,0.27,138.32,34.81,2.02,0.14,0.19,75.26,0,0.47,351.67,0.18,9.24,0.38,2.57,-2.96,-0.28,1.93,0,1.93,34,1710,0,25.84,78,55,1460.31,710.63,451.78,150.85,3.23,0,4530.75,66.25,7.85],
    [3234,1091.56,1357.96,32,8085000,40.08,8.98,25450,0.22,317.7,0.18,0.2,0.49,0.09,0.41,114.69,41.87,2.31,0.15,0.18,75.26,0,0.53,351.67,0.18,9.24,0.24,3.56,-3.09,-0.31,2.17,0,2.17,281,14490,0,80.11,78,55,4287.77,3095.56,1937.42,773.69,2.21,0,4927.51,66.15,7.24],
    [2339,1537.68,1633.02,45,5847500,38.13,9.29,22110,0.24,264.5,0.21,0.26,0.79,0.08,0.71,89.49,32.23,2.2,0.17,0.22,75.26,0,0.51,351.67,0.18,9.24,0.27,4.21,-2.84,-0.29,2.16,0,2.16,228,12150,0,83.6,78,55,3959.8,2404.16,1530.38,659.67,2.59,0,4732.04,66.34,7.67]]

for ligne in img:
    # Make predictions
    yhat = model.predict([ligne])
    # get the label
    label = yhat[0]
    # Display
    print('-- > Prédite= %d (attendue 0)' % (label))
    # See the probability of belonging to a class [In this case 0 or 1].
    # print(model.predict_proba(ligne))

# evaluate certain spill cases (class 1 known)
print()
print('Cas de déversement:')

img = [[2971,1020.91,630.8,59,7427500,32.76,10.48,17380,0.32,427.4,0.22,0.29,0.5,0.08,0.42,149.87,50.99,1.89,0.14,0.18,75.26,0,0.44,351.67,0.18,9.24,2.5,10.63,-3.07,-0.28,2.18,0,2.18,164,8730,0,40.67,78,55,5650.88,1749.29,1245.07,348.7,4.54,0,25579.34,65.78,7.41],
    [3155,1118.08,469.39,11,7887500,30.41,7.99,15880,0.26,496.7,0.2,0.26,0.69,0.11,0.58,118.11,43.96,1.76,0.15,0.18,75.26,0,0.4,351.67,0.18,9.24,0.78,8.68,-3.19,-0.33,2.19,0,2.19,150,8100,0,31.97,78,55,3471.31,3059.41,2043.9,477.23,1.7,0,28172.07,65.72,7.58],
    [115,1449.85,608.43,88,287500,40.42,7.34,3340,0.18,86.1,0.21,0.32,0.5,0.17,0.34,71.2,16.73,1.82,0.19,0.29,87.65,0,0.46,132.78,-0.01,3.78,0.7,4.79,-3.36,-0.23,1.95,0,1.95,29,1530,0.01,38.8,89,69,1400,250,150,45.13,9.33,1,31692.84,65.81,7.84]]

for ligne in img:
    # Make predictions
    yhat = model.predict([ligne])
    # get the label
    label = yhat[0]

    # Display
    print('-- > Prédite= %d (attendue 1)' % (label))






