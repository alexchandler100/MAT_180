# Machine Learning Assisted Khovanov Homology

Project from UC Davis Fall 2022 MAT180 - Mathematics of Machine Learning

By: Yihan (Cecilia) Guo, Keer (Nicole) Ni, and Wei Wu Lu

# Part 1. Project Proposal
This part is the proposal of the final project written initially. We refer to this proposal as a summarize the planning regularly to keep track of our progress.

## Objective

Khovanov Homology is a powerful invariant for knots and links that are expensive to compute. We want to train a machine learning model to predict patterns in Khovanov Homology. 

Khovanov homology can be undestood as a bigraded abelian group associated to a particular knot. As such, each can be decomposed into a *free* group: isomorphic to a direct sum of $\mathbb{Z}$; and a *torsion* group: isomorphic to a direct sum of $\mathbb{Z}_{n_k}$, where each $n_k$ is a distinct prime to some power. 

The bigrading can be represented as plotting the correspoinding subgroup of the homology in Cartesian coordinates.

## Data Sampling

1. Using functions from SageMath, (e.g. Link(), khovanov_homology(), additive_order(), ...) to randomly generate braids. **(done)**
    1. Filter out repeated khovanov homology.
2. Using Khoca to try to get larger dataset (which has much faster performance than SageMath).

## Procedure

1. Import the generated datasets into pandas dataframe. **(done)**
    1. Split the data in training-validation set and test set. **(done)**
2. Implement a gradient descent model.
    1. Linear regression **(done)**
    2. Polynomial regression **(done)**
    3. Neural network
3. Experiment with various representation of the information which the homology has:
    1. Using total number of free groups of a knot to predict its torsion. **(done)**
        - PROBLEM: This might be obscuring the information too much.
    2. Using number of free groups in columns/rows of a knot to predict its torsion. **(done)**
    3. Using every bigrading of free groups as an individual feature to predict its torsion. **(done)**
    4. TBC

## Validation

1. Use the test set accuracy score and compare it with training-validation accuracy score to avoid overfitting/underfitting issues. **(done)**
2. Apply outside model/functions from imported libraries (e.g. sklearn, pytorch, ...) and compare the result with our implemented gradient descent models in procedure step 2. **(done)**

# Part 2. Final Report
This is the report for our final project. We first review to see our achievements, failures, and consider improvment for each step in the proposal in the _review_ sections. The _Result_ section is to summarize the statistical significant finding for our project. The _Discussion_ section is to consider possible future steps for continue our research on Knot Theory, using what we have done as a basis. 

## Data Sampling - review
- Acheivement: Able to generate datasets with different sizes and extract important information (e.g. linke components, number of freegroups in total/per column/per row) using functions from SageMath (see **_dataset_uages.txt_** for description about each dataset in **/data**).
- Failure: Could not use Khoca properly to optimize the data generation process. This is because Khoca does not provide information about bigradings.
    - Improvement: Consider to use Mathematica for computing information about knots.

## Procedure - review
- Achievements: Able to import the data and implement data preprocessing properly and obtain the train-validation-test datasets. Able to visualize the data points. Able to implement Regression models for predicting the total number of torsion groups using (1. total number of free groups 2. total number of free groups per row/column 3. bigrading of free groups).
- Failure: Could not use Neural Networks to predict birading of torsion groups using birading of free groups.
    - Improvement: Use Pytorch&Keras to implement and see the performance result.

## Validation - review
- Achievements: Able to train the model with training dataset, then compare accuracy performance in validation and testing dataset. Able to use regression models from sklearn with the same train-validation-test dataset and compare the accuracy performance.
    - Improvement (1): Implement cross-validation (this is to train-and-validate, then improve weights of the model parameters, after that train-and-validate repetitvely) before the final testing. 
    - Improvement (2): Automate the hyper-parameter tuning pipeline until the best weights for the parameters are selected.

## Reults
1. For predicting total number of torsion groups using total number of free parts:
2. For predicting total number of torsion groups using total number of free parts per column:
3. For predicting total number of torsion groups using total number of free parts per row:
4. For predicting total number of torsion groups using coefficients of Jones Polynomials:
5. For predicting total number of torsion groups using bigrading of freeparts:

## Discussion

# Part 3. Documents usages
This part is to explain the work flow of the project. Users can use this as instruction for how to utlize the code and replicate similar results we have reported in Part 2.

- **/data**: This folder contains all the unused and actively being used datasets. See **_dataset_usages.txt_** for detail descriptions about the datasets.
- **/notebooks**: This folder contains all jupyter notebooks for data generation, data preprocessing, data visualization, model training, and model evaluation processes. 
    - NOTICE: In each of the notebook, change _df = pd.read_csv('data/DATASETNAME.DATASETTYPE')_ to read in and use different datasets.
    - **dataset_generator.ipynb**: In this notebook, we utilize methods to generate dataset wtih different sizes and properties.
    - **data_preprocessing.ipynb**: This notebook provides ways and functions to extract total number of freegroups, total number of torsiongroups, total number of freegroups per row, and total number of freegroups per column.
    - **totalFreePart_Regression.ipynb**: In this notebook, we train Linear regression and Polynomial regression models to predict total number of torsion groups using total number of freeparts for links with different components (1. 1&2&3-components 2. 1-component 3. 2-components 4. 3-components). We also train the regression models from sklearn to compare the result of accuracy.
    - **totalRow&Column_Regression.ipynb**: In this notebook, we train Linear regression and Polynomial regression models to predict total number of torsion groups using total number of freeparts per row and per column for links with different components (1. 1&2&3-components 2. 1-component 3. 2-components 4. 3-components).
    - **jonesPolynomial_Regression.ipynb**: In this notebook, we train Linear regression and Polynomial regression models to predict total number of torsion groups using coefficients of Jones Polynomial for links with different components (1. 1&2&3-components 2. 1-component 3. 2-components 4. 3-components). We also train the regression models from sklearn to compare the result of accuracy.
    - **bigrading_Regression.ipynb**: In this notebook, we train Linear regression and Polynomial regression models to predict total number of torsion groups using bigrading of freeparts for links with different components (1. 1&2&3-components 2. 1-component 3. 2-components 4. 3-components).
    - **Predict_location_torsion.ipynb**: Intended to predcit bigrading of torsion groups using bigrading of free groups. (Future approach: apply CNN and NN models from Pytorch and Keras.)
- **/scripts**: This folder contains all scripts for repetitively used functions in jupyter notebooks. Look at each script for specific usage of each function. Also refer to jupyter notebooks for function usages.
