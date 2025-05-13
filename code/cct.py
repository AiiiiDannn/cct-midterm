# cct.py

# ChatGPT (model: Cognitive Model & ChatGPT o4-mini-high) was used as a reference throughout the development process, assisting in clarifying assignment requirements, shaping the initial code structure and generation, and supporting code debugging.

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os


def load_plant_data(filePath: str = None) -> np.ndarray:
    """
    Load the plant knowledge CSV, drop the first column (Informant ID),
    and return either a DataFrame or a NumPy array.

    Resolving path based on script location.
    """
    if filePath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # directory of this script
        filePath = os.path.join(base_dir, "..", "data", "plant_knowledge.csv")

    df = pd.read_csv(filePath)
    X = df.drop(columns=["Informant"]).values
    return X

def build_cct_model(X: np.ndarray):
    """
    Build the Cultural Consensus Theory model using PyMC.

    Parameters:
    X: np.ndarray
        N x M binary matrix of responses (0 or 1).

    Returns:
    model: pm.Model
        A compiled PyMC model for inference.
    """

    N, M = X.shape    # Number of informants and items
    with pm.Model() as model:
        # Prior for informant competence (between 0.5 and 1)
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)

        # Prior for consensus answers (0 or 1)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Reshape D for broadcasting over items
        D_reshaped = D[:, None]  # Shape becomes (N, 1)

        # Calculate response probability pij using consensus and competence
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)  # Shape (N, M)

        # Likelihood: observed responses follow a Bernoulli distribution
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)

    return model


def run_inference(model, draws, chains, tune, target_accept):
    """
    Run MCMC inference on a given PyMC model.

    Parameters:
    model: pm.Model
        The PyMC model to sample from.
    draws: int
        Number of posterior samples per chain.
    chains: int
        Number of MCMC chains to run.
    tune: int
        Number of tuning steps per chain.
    target_accept: float
        Target acceptance probability for the sampler.

    Returns:
    trace: arviz.InferenceData
        Posterior samples in ArviZ's InferenceData format.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            chains=chains,
            tune=tune,
            target_accept=target_accept,
            return_inferencedata=True
        )
    return trace


def check_convergence(trace):
    """
    Print convergence diagnostics from posterior trace.

    Parameters:
    trace: arviz.InferenceData
        The MCMC samples trace.
    """
    summary = az.summary(trace, var_names=["D", "Z"], round_to=2)
    print(summary)

    # Optional: pair plot for visual inspection
    az.plot_pair(trace, var_names=["D"], kind='kde', marginals=True)
    plt.show()



def analyze_competence(trace):
    """
    Analyze and visualize the posterior distribution of informant competence.

    Parameters:
    trace: arviz.InferenceData
        The MCMC samples trace.
    """
    # Posterior means
    D_means = trace.posterior["D"].mean(dim=["chain", "draw"]).values
    print("Posterior mean competence per informant:")
    for i, d in enumerate(D_means):
        print(f"Informant {i + 1}: {d:.3f}")

    # Plot posterior distributions
    az.plot_posterior(trace, var_names=["D"])
    plt.show()

    # Find most and least competent
    most_competent = np.argmax(D_means)
    least_competent = np.argmin(D_means)
    print(f"Most competent informant: {most_competent + 1}")
    print(f"Least competent informant: {least_competent + 1}")



def analyze_consensus(trace):
    """
    Analyze and visualize the posterior distribution of consensus answers.

    Parameters:
    trace: arviz.InferenceData
        The MCMC samples trace.
    """
    Z_probs = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
    Z_consensus = (Z_probs > 0.5).astype(int)

    print("Posterior mean probability for each consensus answer:")
    print(np.round(Z_probs, 3))
    print("Most likely consensus answer key (rounded):")
    print(Z_consensus)

    # Plot posterior distributions
    az.plot_posterior(trace, var_names=["Z"])
    plt.show()



def compare_with_majority_vote(X, Z_model):
    """
    Compare model-derived consensus answers with simple majority vote.

    Parameters:
    X: np.ndarray
        Original response matrix.
    Z_model: np.ndarray
        Model-inferred consensus answers (binary vector).
    """
    majority_vote = (X.mean(axis=0) > 0.5).astype(int)
    print("Majority vote answers:")
    print(majority_vote)
    print("Model-based consensus answers:")
    print(Z_model)
    print("Agreement rate:", np.mean(majority_vote == Z_model))



if __name__ == "__main__":
    # === Step 1: Load data ===
    # path = "../data/plant_knowledge.csv"
    X = load_plant_data()

    # === Step 2: Build model ===
    model = build_cct_model(X)

    # === Step 3: Run inference ===
    trace = run_inference(
        model,
        draws=2000,
        chains=4,
        tune=1000,
        target_accept=0.95
    )

    # === Step 4: Check convergence ===
    print("\n--- Convergence Diagnostics ---")
    check_convergence(trace)

    # === Step 5: Analyze Informant Competence ===
    print("\n--- Informant Competence ---")
    analyze_competence(trace)

    # === Step 6: Analyze Consensus Answers ===
    print("\n--- Consensus Answers ---")
    analyze_consensus(trace)

    # === Step 7: Compare with Majority Vote ===
    print("\n--- Model vs Majority Vote ---")
    Z_probs = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
    Z_consensus = (Z_probs > 0.5).astype(int)
    compare_with_majority_vote(X, Z_consensus)
