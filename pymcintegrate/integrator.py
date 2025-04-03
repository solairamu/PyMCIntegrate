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
    
    def plot_convergence(self, method='simple', steps=100, samples_per_step=100, **kwargs):
        """
        Plot the convergence of the integral estimate as the number of samples increases.

        Parameters:
            method (str): The method used for integration ('simple', 'importance', 'antithetic').
            steps (int): Number of convergence steps.
            samples_per_step (int): Number of samples added per step.
            **kwargs: For "importance", provide proposal_pdf and proposal_sampler.
        """
        estimates = []
        total_samples = steps * samples_per_step
        a, b = self.domain

        if method == 'simple':
            for step in range(1, steps + 1):
                n = step * samples_per_step
                samples = np.random.uniform(a, b, n)
                values = self.func(samples)
                est = (b - a) * np.mean(values)
                estimates.append(est)
        elif method == 'antithetic':
            for step in range(1, steps + 1):
                n = step * samples_per_step
                half_n = n // 2
                u = np.random.uniform(0, 1, half_n)
                x1 = a + (b - a) * u
                x2 = a + (b - a) * (1 - u)
                values1 = self.func(x1)
                values2 = self.func(x2)
                values = (values1 + values2) / 2.0
                est = (b - a) * np.mean(values)
                estimates.append(est)
        elif method == 'importance':
            proposal_pdf = kwargs.get('proposal_pdf', None)
            proposal_sampler = kwargs.get('proposal_sampler', None)
            if proposal_pdf is None or proposal_sampler is None:
                raise ValueError("For importance sampling, provide 'proposal_pdf' and 'proposal_sampler'.")
            for step in range(1, steps + 1):
                n = step * samples_per_step
                samples = proposal_sampler(n)
                weights = self.func(samples) / proposal_pdf(samples)
                est = np.mean(weights)
                estimates.append(est)
        else:
            raise ValueError("Invalid method. Choose from 'simple', 'importance', or 'antithetic'.")
        
        # Plot the Convergence
        plt.figure(figsize=(8, 4))
        x_axis = np.arange(samples_per_step, total_samples + 1, samples_per_step)
        plt.plot(x_axis, estimates, marker='o')
        plt.xlabel('Number of Samples')
        plt.ylabel('Integral Estimate')
        plt.title(f'Convergence of Monte Carlo Integration using {method.capitalize()} Method')
        plt.grid(True)
        plt.show
        return estimates