import numpy as np
from scipy.stats import norm


def load_results(csv_path):
    """Load a results CSV into a dict of {column_name: np.ndarray}.

    Works with any CSV written by the experiment scripts (comma-separated,
    header row with column names, no comment prefix).

    Example:
        data = load_results('results/nmse_vs_snr_delta=0.6_...csv')
        plt.semilogy(data['snr_db'], data['nmse_sgl'])
    """
    structured = np.genfromtxt(csv_path, delimiter=',', names=True)
    return {name: structured[name] for name in structured.dtype.names}


def _get_stable_alpha(delta=0.5, kappa=2):
    """Compute the stable step-size alpha for AMP.

    Returns (1/sqrt(delta)) * argmax_{z >= 0} of:
        (1 - kappa/delta * T(z)) / (1 + z^2 - kappa * T(z))
    where T(z) = (1 + z^2) * Phi(-z) - z * phi(z),
    and Phi, phi are the standard normal CDF and PDF.
    """
    z_values = np.arange(0, 20.0 + 0.001, 0.001)

    term = (1 + z_values ** 2) * norm.cdf(-z_values) - z_values * norm.pdf(z_values)
    numerator = 1 - (kappa / delta) * term
    denominator = 1 + z_values ** 2 - kappa * term

    objective = np.where(np.abs(denominator) > 1e-12, numerator / denominator, -np.inf)
    z_opt = z_values[np.argmax(objective)]

    return (1.0 / np.sqrt(delta)) * z_opt
