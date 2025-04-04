from pymcintegrate import MonteCarloIntegrator
import numpy as np

def main():
    # Define the function to integrate, e.g., a Gaussian
    def f(x):
        return np.exp(-x**2)
    
    # Define the integration limits
    domain = (-2, 2)

    # Create integrator instance
    integrator = MonteCarloIntegrator(f, domain, num_samples=10000, seed=42)

    # Simple Monte Carlo integration
    estimate, error = integrator.integrate_simple()
    print(f"Simple Monte Carlo Estimate: {estimate}, Error: {error}")

    # Antithetic variates integration
    ant_estimate, ant_error = integrator.integrate_antithetic()
    print(f"Antithetic Variates Estimate: {ant_estimate}, Error: {ant_error}")

    # Importance sampling example
    # Here we define a proposal distribution: using a Beta(2,2) on [0,1] and mapping it to the domain.
    from scipy.stats import beta

    a, b = domain
    proposal_pdf = lambda x: beta.pdf((x - a) / (b - a), 2, 2) / (b - a)
    proposal_sampler = lambda n: a + (b - a) * beta.rvs(2, 2, size=n)

    imp_estimate, imp_error = integrator.integrate_importance(proposal_pdf, proposal_sampler)
    print(f"Importance Sampling Estimate: {imp_estimate}, Error: {imp_error}")

    # Adaptive Monte Carlo integration
    estimate, error = integrator.adaptive_integrate(bins=10, initial_samples_per_bin=100, total_samples=10000)
    print("Adaptive MC Estimate:", estimate, "with error:", error)

    # Plot convergence for simple Monte Carlo
    integrator.plot_convergence(method='simple', steps=50, samples_per_step=100)

    # Animate convergence using the simple method
    integrator.animate_convergence_loop(method="simple", steps=50, samples_per_step=100, interval=0.1)

if __name__ == "__main__":
    main()
