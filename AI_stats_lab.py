"""
AI_stats_lab.py

Lab: Bias-Variance Tradeoff

Topics:
- Nonlinear data generation
- Polynomial regression
- Train/dev error comparison
- Model complexity
- Bias-variance diagnosis
- Regularization comparison
- Model improvement recommendations
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ============================================================
# Question 1: Model Complexity and Generalization
# ============================================================

def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate a nonlinear regression dataset.

    True function:
        y = sin(2*pi*x) + Gaussian noise

    Parameters:
        n_samples    : number of samples
        noise        : standard deviation of Gaussian noise
        random_state : random seed

    Returns:
        X : shape (n_samples, 1)
        y : shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    # Draw x uniformly from [0, 1], then reshape to (n_samples, 1)
    X = rng.uniform(0, 1, size=n_samples).reshape(-1, 1)

    # True signal + Gaussian noise
    y = np.sin(2 * np.pi * X.ravel()) + rng.normal(0, noise, size=n_samples)

    return X, y


def create_polynomial_model(degree):
    """
    Create a polynomial regression model using sklearn Pipeline.

    Pipeline steps:
        1. PolynomialFeatures(degree=degree, include_bias=False)
        2. LinearRegression()

    Parameters:
        degree : polynomial degree

    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression())
    ])
    return pipeline


def evaluate_polynomial_degrees(X, y, degrees, test_size=0.3, random_state=0):
    """
    Train polynomial models with different degrees and compute train/dev errors.

    Parameters:
        X            : feature matrix
        y            : target values
        degrees      : list of polynomial degrees
        test_size    : fraction of data used for dev set
        random_state : random seed

    Returns:
        dict with keys:
            "degrees"      : list of degrees
            "train_errors" : list of training MSE values
            "dev_errors"   : list of dev MSE values
            "best_degree"  : degree with lowest dev error
    """
    # Split once; reuse the same split for every degree
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_errors = []
    dev_errors   = []

    for degree in degrees:
        model = create_polynomial_model(degree)
        model.fit(X_train, y_train)

        # Training MSE
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_errors.append(train_mse)

        # Dev (validation) MSE
        y_dev_pred = model.predict(X_dev)
        dev_mse = mean_squared_error(y_dev, y_dev_pred)
        dev_errors.append(dev_mse)

    # Best degree = the one with the minimum dev error
    best_degree = degrees[int(np.argmin(dev_errors))]

    return {
        "degrees":      list(degrees),
        "train_errors": train_errors,
        "dev_errors":   dev_errors,
        "best_degree":  best_degree,
    }


def diagnose_from_errors(train_error, dev_error, high_error_threshold=0.15, gap_threshold=0.05):
    """
    Diagnose model behavior using train and dev error.

    Parameters:
        train_error          : training error
        dev_error            : dev error
        high_error_threshold : threshold for high train error
        gap_threshold        : threshold for high dev-train gap

    Returns:
        dict with keys:
            "train_error"         : train_error
            "dev_error"           : dev_error
            "generalization_gap"  : dev_error - train_error
            "diagnosis"           : one of the four diagnosis strings

    Diagnosis rules:
        high train + small gap  -> "high_bias"
        low  train + large gap  -> "high_variance"
        high train + large gap  -> "high_bias_and_high_variance"
        otherwise               -> "good_fit"
    """
    gap = dev_error - train_error

    high_train = train_error > high_error_threshold
    high_gap   = gap > gap_threshold

    if high_train and not high_gap:
        diagnosis = "high_bias"
    elif not high_train and high_gap:
        diagnosis = "high_variance"
    elif high_train and high_gap:
        diagnosis = "high_bias_and_high_variance"
    else:
        diagnosis = "good_fit"

    return {
        "train_error":        train_error,
        "dev_error":          dev_error,
        "generalization_gap": gap,
        "diagnosis":          diagnosis,
    }


# ============================================================
# Question 2: Regularization and Model Improvement
# ============================================================

def regularization_comparison(X_train, y_train, X_dev, y_dev, alphas):
    """
    Compare Ridge regression models with different regularization strengths.

    Parameters:
        X_train : training features
        y_train : training targets
        X_dev   : dev features
        y_dev   : dev targets
        alphas  : list of Ridge alpha values

    Returns:
        dict with keys:
            "alphas"       : list of alpha values
            "train_errors" : list of training MSE values
            "dev_errors"   : list of dev MSE values
            "best_alpha"   : alpha with lowest dev error
    """
    train_errors = []
    dev_errors   = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        train_mse = mean_squared_error(y_train, model.predict(X_train))
        dev_mse   = mean_squared_error(y_dev,   model.predict(X_dev))

        train_errors.append(train_mse)
        dev_errors.append(dev_mse)

    best_alpha = alphas[int(np.argmin(dev_errors))]

    return {
        "alphas":       list(alphas),
        "train_errors": train_errors,
        "dev_errors":   dev_errors,
        "best_alpha":   best_alpha,
    }


def recommend_action(diagnosis):
    """
    Recommend an action based on bias/variance diagnosis.

    Mapping:
        "high_bias"                   -> "increase_model_complexity"
        "high_variance"               -> "add_regularization_or_more_data"
        "high_bias_and_high_variance" -> "increase_complexity_then_regularize"
        "good_fit"                    -> "keep_model_or_minor_tuning"
        anything else                 -> "unknown_diagnosis"
    """
    action_map = {
        "high_bias":                   "increase_model_complexity",
        "high_variance":               "add_regularization_or_more_data",
        "high_bias_and_high_variance": "increase_complexity_then_regularize",
        "good_fit":                    "keep_model_or_minor_tuning",
    }
    return action_map.get(diagnosis, "unknown_diagnosis")


# ============================================================
# Quick smoke-test (optional, safe to run)
# ============================================================
if __name__ == "__main__":
    # --- Q1 ---
    X, y = generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    degrees = [1, 2, 3, 5, 9, 15]
    results = evaluate_polynomial_degrees(X, y, degrees)
    print("\nPolynomial degree evaluation:")
    for d, tr, dv in zip(results["degrees"], results["train_errors"], results["dev_errors"]):
        print(f"  degree={d:2d}  train_MSE={tr:.4f}  dev_MSE={dv:.4f}")
    print(f"  Best degree: {results['best_degree']}")

    # Diagnose the best model
    best_d = results["best_degree"]
    idx = results["degrees"].index(best_d)
    diag = diagnose_from_errors(results["train_errors"][idx], results["dev_errors"][idx])
    print(f"\nDiagnosis: {diag}")

    # --- Q2 ---
    from sklearn.model_selection import train_test_split
    X_tr, X_dv, y_tr, y_dv = train_test_split(X, y, test_size=0.3, random_state=0)

    # Use degree-9 polynomial features for the Ridge comparison
    poly = PolynomialFeatures(degree=9, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr)
    X_dv_poly = poly.transform(X_dv)

    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    reg_results = regularization_comparison(X_tr_poly, y_tr, X_dv_poly, y_dv, alphas)
    print("\nRegularization comparison:")
    for a, tr, dv in zip(reg_results["alphas"], reg_results["train_errors"], reg_results["dev_errors"]):
        print(f"  alpha={a:<8}  train_MSE={tr:.4f}  dev_MSE={dv:.4f}")
    print(f"  Best alpha: {reg_results['best_alpha']}")

    print(f"\nRecommend for 'high_variance': {recommend_action('high_variance')}")
    print(f"Recommend for 'high_bias':     {recommend_action('high_bias')}")
    print(f"Recommend for 'good_fit':      {recommend_action('good_fit')}")