# Machine Learning Assisted Khovanov Homology

Project from UC Davis Fall 2022 MAT180 - Mathematics of Machine Learning

By: Yihan (Cecilia) Guo, Keer (Nicole) Ni and Wei Wu Lu

## Objective

Khovanov Homology is a powerful invariant for knots and links that are expensive to compute. We want to train a machine learning model to predict patterns in Khovanov Homology. 

Khovanov homology can be undestood as a bigraded abelian group associated to a particular knot. As such, each can be decomposed into a *free* group: isomorphic to a direct sum of $\mathbb{Z}$; and a *torsion* group: isomorphic to a direct sum of $\mathbb{Z}_{n_k}$, where each $n_k$ is a distinct prime to some power. 

The bigrading can be represented as plotting the correspoinding subgroup of the homology in Cartesian coordinates.

## Data Sampling

1. Using functions from SageMath, (e.g. Link(), khovanov_homology(), additive_order(), ...) to randomly generate braids.
    1. Filter out repeated khovanov homology.
2. Using Khoca to try to get larger dataset (which has much faster performance than SageMath).

## Procedure

1. Import the generated datasets into pandas dataframe.
    1. Split the data in training-validation set and test set.
2. Implement a gradient descent model.
    1. Linear regression
    2. Polynomial regression
    3. Neural network (?)
3. Experiment with various representation of the information which the homology has:
    1. Using total number of free groups of a knot to predict its torsion.
        - PROBLEM: This might be obscuring the information too much.
    2. Using number of free groups in columns/rows of a knot to predict its torsion.
    3. Using every bigrading of free groups as an individual feature to predict its torsion.
    4. TBC

## Validation

1. Use the test set accuracy score and compare it with training-validation accuracy score to avoid overfitting/underfitting issues. 
2. Apply outside model/functions from imported libraries (e.g. sklearn, pytorch, ...) and compare the result with our implemented gradient descent models in procedure step 2. 
