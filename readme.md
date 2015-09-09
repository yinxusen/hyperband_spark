---
# Multi-armd Bandit for Hyper-parameter selection

---

It's a very rough prototype with following features:

- Support classification dataset input, partition into training/validation/test set, including standard scaling for dataset.

- Support PartialEstimator, which fits with a training set and a potential model, iterates specified steps then produce a new model.

- LinearRidgeRegression extends from PartialEstimator, provides the ability to train a linear ridge regression model. It calls single step SGD to perform the training.

- ParamSampler is a sampler from a (minVal, maxVal) range, for selecting hyperparameters from a range. Currently I only wrote the IntSampler and DoubleSampler, without categrical sampler.

- Arm just likes the original arms.py. I once want to implement in a funtional style, but it is not easy. So this version is acceptable now.

- ModelFamily provides APIs to generate arms given specified estimator, model, evaluator, etc.

- SearchStrategy provides the ability to pull arms to find best arm. Currently I only implemented the static search strategy.

- BanditValidator is a main entry for multi-arm bandit hyperparameter selection.

And finally, the BanditValidatorExample is a simple example to show its ability.
