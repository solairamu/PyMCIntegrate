# PyMCIntegrate

PyMC Integrate is a Python package for performing adaptive Monte Carlo integration and uncertainty quantification. It provides several integration methods including:

- **Simple Monte Carlo Integration:** Uniform random sampling.
- **Importance Sampling:** Use a custom proposal distribution. 
- **Antithetic Variates:** Improve estimation efficiency by pairing samples. 
- **Adaptive Monte Carlo Integration:** Stratify the integration domain and allocate extra samples based on variance estimates to improve accuracy in regions with higher variability.
- **Convergence Visualization:** PLot the convergence of the integral estimate as the humber of samples increases.
- **Animated Convergence Visualization:** Real-time animation that dynamically updates the integral estimate, offering an engaging way to observe convergence behavior.