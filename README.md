# Oil-Spill-Detection-Using-Machine-Learning
Project for the Benefit of the National Hydrocarbons Company SONATRACH Algeria.

## Introduction

Only 10% of oil spills come from natural sources such as seabed leaks. Pollution caused intentionally by transport vessels is far more widespread. Radar images from Synthetic Aperture Radar (SAR) mounted on satellites, offer the possibility of monitoring coastal waters.

![SAR](https://github.com/A-Mohamed0/Oil-Spill-Detection-Using-Machine-Learning/assets/154687338/b7da4168-6a8c-47c6-baec-8adfeb9d78ce)

*Fragment of a SAR image - The oil sheath is the elongated dark region visible in the top right of the image but The dark areas in the middle of the image and bottom left are similarities -*

<span style="color: #26B260"> This project is a machine learning model dedicated to the detection of oil slicks from satellite images. </span>

## DATA SET
A standard imbalanced dataset was introduced in the [Research Article](https://link.springer.com/content/pdf/10.1023/A:1007452223027.pdf). The dataset consists of satellite images of the ocean, some of which contain an oil spill, while others do not.

The images were divided into sections and processed using computer vision algorithms to generate a feature vector describing the content of each image section or patch.

## Data pre-processing

- The initial dataset was provided with features (columns) visually deemed irrelevant to learning performance. These columns include the patch number (first column) and column 22, containing a single unique value, which was removed because columns with only one observation or value are generally ineffective for modeling and are termed zero-variance predictors (as their variance would be null if measured). 

Furthermore, the dataset columns were separated into input and output variables. Column 49, which holds class labels (presence or absence of a spill), was encoded to represent classes 0 and 1.

```Python
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
    # Encodes target variable labels to have classes 0 and 1 using the sklearn LabelEncoder class
    y = LabelEncoder().fit_transform(y)
    return X, y

```
- The data set was sparse, as shown in the figure below.
![Histogramme](https://github.com/A-Mohamed0/Oil-Spill-Detection-Using-Machine-Learning/assets/154687338/71a478c0-7a93-4973-80dd-c04b29c747ca)

In this case, we will normalize the dataset before fitting our model with `MaxAbsScaler()` to create a scaled dataset where parsimony is preserved, a crucial step when training a supervised machine learning model.

- The data is also imbalanced, denoted by the fact that class 0 (no spill) is extremely minority compared to class 1 (Spill). With percentages of 4.376% and 95.624% for each class, respectively. With `ADASYN()` we have sampled the data to better prepare the unbalanced training dataset. 

```Python
## transform the dataset using the ADASYN oversampling method.
oversample = ADASYN()
X_sam, y_sam = oversample.fit_resample(X, y)
```

Using the `ADASYN()` function results in the prompt acquisition of a rebalanced dataset, comprising 887 instances of spill and 896 instances of no-spill.

![Captures](https://github.com/A-Mohamed0/Oil-Spill-Detection-Using-Machine-Learning/assets/154687338/dc6c6db9-df58-483d-b705-3ac74d4c7410)


## fit the model

Following the completion of the data preprocessing stage, the data is now prepared for training. In our study, we employed the `LightGBM` algorithm for training on the dataset.
```Python
estimateur = [('t1', MaxAbsScaler()),

            ('m', lgb.LGBMClassifier(n_estimators=362, num_leaves=1208, min_child_samples=8,
            learning_rate=0.02070742242160566, colsample_bytree=0.37915528071680865,
            reg_alpha=0.002982599447751338, reg_lambda= 1.136605174453919))]
model = Pipeline(estimateur)
```
## Results obtained
We obtained a score of 0.993 as the value for the geometric mean, i.e. the `LightGBM` algorithm manages to link the input data (feature) to the class labels with a success rate of 99%.

![Score](https://github.com/A-Mohamed0/Oil-Spill-Detection-Using-Machine-Learning/assets/154687338/9d9522c2-4dad-4a8d-8011-918d60bcecdb)


## Predictions based on new data

In the final phase, we utilized the model to make predictions on new data. After being tailored to spill data through the `fit()` method, it was then applied to make predictions for fresh data by invoking the `predict()` function. This will result in a class label of 0 indicating no  spill or 1 for a  spill.

The data intended for prediction were gathered from a study conducted by Jason Brownlee on February 21, 2020, in an article focusing on imbalanced classification. A total of 6 images were used, comprising 3 instances where we are aware there is no oil spill and the remaining where we know there is one.

The aim of this experiment is to demonstrate that we can use the fitting model to make label predictions for new data that the model has never seen before.

## Authors
- [@Atoui_Mohamed](https://github.com/A-Mohamed0)
- [@Souadi kamel](skamelmail@gmail.com)
