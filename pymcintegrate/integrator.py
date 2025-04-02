import numpy as np
import matplotlib.pyplot as plt
class MonteCarloIntegrator:
    def __init__(self, func, domain, num_samples=10000, seed=None):
        """
        Initialize the Monte Carlo Integrator.

        Parameters:
        func (callable): The function to integrate.
        domain (tuple): The integration limits as (a, b).
        num_samples (int): Default number of samples to use for integration.
        seed (int, optional): Seed for reproducibility.
        """
        self.func = func
        self.domain = domain #assuming domain is (a, b)
        self.num_samples = num_samples
        if seed is not None:
            np.random.seed(seed)
    
    def integrate_simple(self):
        """
        Perform simple Monte Carlo integration using uniform sampling.

        Returns:
            tuple: (estimate, standard error)   
        """
        a, b = self.domain
        samples = np.random.uniform(a, b, self.num_samples)
        function_values = self.func(samples)
        estimate = (b - a) * np.mean(function_values)
        error = (b - a) * np.std(function_values) / np.sqrt(self.num_samples)
        return estimate, error
    
    def integrate_importance(self, proposal_pdf, proposal_sampler):
        """
        Perform Monte Carlo integration using importance sampling.

        Parameters:
            proposal_pdf (callable): Function to evaluarte the proposal density.
            proposal_sampler (callable): Function to generate samples from the proposal distribution.

        Returns:
            tuple: (estimate, standard error)
        """
        samples = proposal_sampler(self.num_samples)
        weights = self.func(samples) / proposal_pdf(samples)
        estimate = np.mean(weights)
        error = np.std(weights) / np.sqrt(self.num_samples)
        return estimate, error
    
    def integrate_antithetic(self):
        """
        Perform Monte Carlo integration using antithetic variates.

        Returns:
            tuple: (estimate, standard error)
        """
        a, b = self.domain
        n = self.num_samples
        half_n = n // 2
        u = np.random.uniform(0, 1, half_n)
        x1 = a + (b - a) * u
        x2 = a + (b - a) * (1 - u)
        values1 = self.func(x1)
        values2 = self.func(x2)
        values = (values1 + values2) / 2
        estimate = (b - a) * np.mean(values)
        error = (b - a) * np.std(values) / np.sqrt(half_n)
        return estimate, error