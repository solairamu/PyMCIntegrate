import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    
    def adaptive_integrate(self, bins=10, initial_samples_per_bin=1000, total_samples=10000):
        """
        Perform adaptive Monte Carlo integration by stratifying the integration domain into bins,
        and allocating extra samples based on the variance within each bin.

        Parameters:
            bins (int): Number of bins to divide the integration domain.
            initial_samples_per_bin (int): Number of samples per bin in the initial phase.
            total_samples (int): Total number of samples to be used.

        Returns:
            tuple: (estimate, error)
        """
        a, b = self.domain
        bin_edges = np.linspace(a, b, bins + 1)
        bin_samples = []

        for i in range(bins):
            samples =  np.random.uniform(bin_edges[i], bin_edges[i + 1], initial_samples_per_bin)
            bin_samples.append(samples)

        bin_vars = np.array([np.var(self.func(s)) for s in bin_samples])
        total_initial = bins * initial_samples_per_bin
        remaining_samples = total_samples - total_initial

        if np.sum(bin_vars) > 0:
            extra_allocation = np.round(remaining_samples * (bin_vars / np.sum(bin_vars))).astype(int)
        else:
            extra_allocation = np.full(bins, remaining_samples // bins)

        for i in range(bins):
            if extra_allocation[i] > 0:
                extra_samples = np.random.uniform(bin_edges[i], bin_edges[i+1], extra_allocation[i])
                bin_samples[i] = np.concatenate((bin_samples[i], extra_samples))

        bin_estimates = []
        bin_errors = []
        for i in range(bins):
            samples = bin_samples[i]
            mean_val = np.mean(self.func(samples))
            std_val = np.std(self.func(samples))
            weight = bin_edges[i+1] - bin_edges[i]
            estimate_bin = weight * mean_val
            error_bin = weight * std_val / np.sqrt(len(samples))
            bin_estimates.append(estimate_bin)
            bin_errors.append(error_bin)

        total_estimate = np.sum(bin_estimates)
        total_error = np.sqrt(np.sum(np.array(bin_errors) ** 2))
        return total_estimate, total_error
    
    def plot_convergence(self, method='simple', steps=100, samples_per_step=100, save_folder ='plots', **kwargs):
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
        # Save the plot if a save_folder is specified
        if save_folder:
            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # Create a unique filename using a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"convergence_{method}_{timestamp}.png")
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
        else:
            plt.show()
        return estimates
    
    def animate_convergence_loop(self, method="simple", steps=50, samples_per_step=100, interval=0.1, save_folder ='plots'):
        """
        Animate the convergence of the integral estimate in real time.
        
        Parameters:
            method (str): Integration method ("simple", "antithetic", or "importance").
            steps (int): Number of animation steps.
            samples_per_step (int): Number of samples added per step.
            interval (float): Pause duration between frames (in seconds).
        """
        a, b = self.domain
        estimates = []
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(8, 4))
        line, = ax.plot([], [], marker='o')
        ax.set_xlim(samples_per_step, steps * samples_per_step)
        # Adjust y-axis limits according to your function's range; modify as needed
        ax.set_ylim(0, (b - a) * 1.2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Integral Estimate')
        ax.set_title(f'Convergence Animation: {method.capitalize()} Method')
        ax.grid(True)
        
        for step in range(steps):
            n = (step + 1) * samples_per_step
            if method == "simple":
                samples = np.random.uniform(a, b, n)
                values = self.func(samples)
                est = (b - a) * np.mean(values)
            elif method == "antithetic":
                half_n = n // 2
                u = np.random.uniform(0, 1, half_n)
                x1 = a + (b - a) * u
                x2 = a + (b - a) * (1 - u)
                values1 = self.func(x1)
                values2 = self.func(x2)
                est = (b - a) * np.mean((values1 + values2) / 2.0)
            elif method == "importance":
                # For importance sampling, you’d need to supply proposal_pdf and proposal_sampler
                raise NotImplementedError("Loop-based importance sampling not implemented in this example.")
            else:
                raise ValueError("Unknown method. Choose 'simple', 'antithetic', or 'importance'.")
            
            estimates.append(est)
            x_data = np.arange(samples_per_step, (step + 1) * samples_per_step + 1, samples_per_step)
            line.set_data(x_data, estimates)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(interval)
        
        plt.ioff()
        # Save the plot if a save_folder is specified
        if save_folder:
            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # Create a unique filename using a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"animated_convergence_{method}_{timestamp}.png")
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
        else:
            plt.show()
        plt.show()

