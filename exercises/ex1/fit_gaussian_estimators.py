import numpy as np
import pandas as pd

import IMLearn.learners.gaussian_estimators as ga
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def univariate():
    # Q1
    X = np.random.normal(10, 1, 1000)
    model = ga.UnivariateGaussian()
    model = model.fit(X)
    print(f"({model.mu_}, {model.var_})")
    # Q2
    dists = np.zeros(100)
    for i in range(10, 1001, 10):
        m = ga.UnivariateGaussian().fit(X[:i])
        dists[int((i / 10) - 1)] = np.abs(m.mu_ - 10)
    plt.plot(np.arange(10, 1001, 10), dists, 'b.')
    plt.xlabel("Number of Samples")
    plt.ylabel("Abs. Distance between Estimated and True Expectation")
    plt.title("Samples Size vs. Distance")
    plt.show()
    plt.close()
    # Q3
    pdf = model.pdf(X)
    plt.scatter(X, pdf)
    plt.show()
    plt.close()


def multivariate():
    # Q1
    mean = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean, sigma, 1000)
    estimator = ga.MultivariateGaussian()
    estimator.fit(X)
    print("Expectation:\n", estimator.mu_)
    print("Covariance:\n", estimator.cov_)
    # Q2
    f1, f3 = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    f1, f3 = f1.flatten(), f3.flatten()
    likelihood = np.zeros(200 * 200)
    for i in range(200 * 200):
        likelihood[i] = ga.MultivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[i], 0]), sigma, X)
    go.Figure(go.Heatmap(x=f1, y=f3, z=likelihood),
              layout=go.Layout(title="Log-Likelihood by f1, f3", xaxis_title="f1", yaxis_title="f3", height=600,
                               width=400)).show()
    # Q3
    print(f"f1: {round(f1[np.argmax(likelihood)], 3)}, f3: {round(f3[np.argmax(likelihood)], 3)}")


if __name__ == "__main__":
    univariate()
    multivariate()
