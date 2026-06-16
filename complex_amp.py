import numpy as np


class ComplexAMP:
    """Complex-valued Approximate Message Passing for sparse group lasso recovery."""

    def __init__(
        self,
        measurements,
        sensing_matrix,
        signal_true=None,
        prior_sample=None,
        group_size=1,
        noise_variance=0.0,
        step_size=1.0,
        tau_estimation='empirical',
    ):
        # Input data
        self.measurements = measurements      # (batch, measurements, 1)
        self.sensing_matrix = sensing_matrix  # (batch, measurements, elements)
        self.signal_true = signal_true        # (batch, elements, 1)
        self.prior_sample = prior_sample      # (batch, elements, 1)
        self.true_support = (np.abs(signal_true) > 0) * 1.0

        # Dimensions
        self.batch_size = measurements.shape[0]
        self.signal_dim = sensing_matrix.shape[-1]
        self.num_measurements = measurements.shape[-2]
        self.measurement_ratio = self.num_measurements / self.signal_dim

        # Fixed solver parameters
        self.noise_variance = noise_variance
        self.step_size = step_size
        self.tau_estimation = tau_estimation

        self.algo = None

        # Group structure
        self.group_size = group_size
        self.num_groups = self.signal_dim // group_size

        # All metric lists share the same indexing:
        #   index 0   — initial state before any iterations (seeded by _init_se)
        #   index 1…N — after each iteration
        self.se_effective_noise = []
        self.se_mse = []
        self.se_pmd = []
        self.se_nmse = []
        self.se_npmd = []

        self.actual_mse = []
        self.actual_pmd = []
        self.actual_nmse = []
        self.actual_nmse_std = []
        self.actual_npmd = []

        # Current algorithm state — set by initialize() before each run
        self.signal_estimate = None
        self.onsager_correction = None
        self.residual = None

    # ------------------------------------------------------------------
    # Core math helpers
    # ------------------------------------------------------------------

    def _soft_threshold(self, x, threshold=0.0):
        """Element-wise complex soft thresholding."""
        if np.any(threshold > 0.0):
            clipped_magnitude = np.maximum(np.abs(x), threshold)
            shrinkage = np.maximum(0.0, 1.0 - threshold / clipped_magnitude)
            x = x * shrinkage
        return x

    def _hermitian(self, matrix):
        """Hermitian (conjugate transpose) of a batched matrix: (batch, m, n) -> (batch, n, m)."""
        return np.conj(matrix).swapaxes(-1, -2)

    def denoise(self, estimate, element_threshold=0.5, group_threshold=0.5):
        """Apply sparse group lasso proximal operator (element then group soft thresholding)."""

        estimate = self._soft_threshold(estimate, threshold=element_threshold)

        if self.group_size > 1 and np.any(group_threshold > 0.0):
            estimate = estimate.reshape(-1, self.num_groups, self.group_size)
            group_norms = np.linalg.vector_norm(
                estimate, axis=-1, ord=2, keepdims=True
            )  # (batch, num_groups, 1)
            clipped_group_norms = np.maximum(group_norms, group_threshold)
            group_shrinkage = np.maximum(0.0, 1.0 - group_threshold / clipped_group_norms)
            estimate = estimate * group_shrinkage
            # reshape() required because array is non-contiguous after broadcasting
            estimate = estimate.reshape(-1, self.num_groups * self.group_size, 1)

        return estimate

    def _denoiser_avg_divergence(self, estimate, element_threshold=0.0, group_threshold=0.0):
        """Average divergence of the sparse group lasso denoiser (for Onsager correction)."""

        input_groups = estimate.reshape(-1, self.num_groups, self.group_size)
        element_magnitudes = np.abs(input_groups)

        element_activity_mask = (element_magnitudes > element_threshold) * 1.0
        clipped_magnitudes = np.maximum(element_magnitudes, element_threshold)
        element_divergence = element_activity_mask * (
            1 - element_threshold / (2 * clipped_magnitudes)
        )
        element_divergence_sum = element_divergence.sum(axis=-1, keepdims=True)

        if np.any(group_threshold > 0.0):
            lasso_denoised_groups = self._soft_threshold(input_groups, threshold=element_threshold)
            group_norms = np.linalg.vector_norm(
                lasso_denoised_groups, axis=-1, ord=2, keepdims=True
            )
            group_activity_mask = (group_norms > group_threshold) * 1.0
            clipped_group_norms = np.maximum(group_norms, group_threshold)
            group_divergence = group_activity_mask * (
                group_threshold / (2 * clipped_group_norms)
                + (1 - group_threshold / clipped_group_norms) * element_divergence_sum
            )
        else:
            group_divergence = element_divergence_sum

        group_divergence = group_divergence.sum(axis=-2, keepdims=True)  # (batch, 1, 1)
        avg_divergence = (1 / self.signal_dim) * group_divergence
        return avg_divergence

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def _amp_step(self, element_threshold, group_threshold):
        """Single AMP iteration: residual -> denoiser input -> denoise -> Onsager update."""

        self.residual = (
            self.measurements
            - self.sensing_matrix @ self.signal_estimate
            + self.onsager_correction
        )
        self.denoiser_input = (
            self.signal_estimate
            + self._hermitian(self.sensing_matrix) @ self.residual
        )
        self.signal_estimate = self.denoise(
            self.denoiser_input, element_threshold, group_threshold
        )
        avg_divergence = self._denoiser_avg_divergence(
            self.denoiser_input, element_threshold, group_threshold
        )
        self.onsager_correction = (
            1 / self.measurement_ratio * avg_divergence * self.residual
        )

    def _ista_step(self, element_threshold, group_threshold):
        """Single ISTA iteration: gradient step -> denoise."""

        self.residual = self.measurements - self.sensing_matrix @ self.signal_estimate
        self.denoiser_input = (
            self.signal_estimate
            + self.step_size * self._hermitian(self.sensing_matrix) @ self.residual
        )
        self.signal_estimate = self.denoise(
            self.denoiser_input, element_threshold, group_threshold
        )

    # ------------------------------------------------------------------
    # Initialization and state evolution
    # ------------------------------------------------------------------

    def initialize(self, warm_start=False):
        """Set algorithm state.

        Cold start (warm_start=False): reset estimate and Onsager term to zero.
        Warm start (warm_start=True): preserve current estimate and Onsager term.

        In both cases, recompute residual from current state so that the
        empirical tau estimate at the start of run() is correct.
        """
        if not warm_start:
            self.signal_estimate = np.zeros_like(self.signal_true)
            self.onsager_correction = 0.0

        # Residual is computed from current state rather than stored directly,
        # because after _amp_step the stored residual corresponds to the
        # pre-update estimate, not the post-update one.
        self.residual = (
            self.measurements
            - self.sensing_matrix @ self.signal_estimate
            + self.onsager_correction
        )

    def _append_actual_metrics(self):
        """Compute and append one entry to all actual performance lists."""
        error = self.signal_estimate - self.signal_true
        per_batch_nmse = (
            np.sum(np.abs(error) ** 2, axis=1)
            / np.maximum(np.sum(np.abs(self.signal_true) ** 2, axis=1), 1e-6)
        )  # (batch, 1)
        self.actual_mse.append(np.mean(np.abs(error) ** 2))
        self.actual_nmse.append(np.mean(per_batch_nmse))
        self.actual_nmse_std.append(np.std(per_batch_nmse))

        estimated_support = (np.abs(self.signal_estimate) > 0) * 1.0
        missed = self.true_support * (1 - estimated_support)
        self.actual_pmd.append(np.mean(missed))
        self.actual_npmd.append(np.mean(
            np.sum(missed, axis=1)
            / np.maximum(np.sum(self.true_support, axis=1), 1e-6)
        ))

    def _init_se(self):
        """Seed all SE lists with index-0 values derived from the prior sample.

        Called once before the first run() when lists are empty. After this,
        all lists have exactly one entry (index 0 = state before any iterations),
        and _se_step() appends one entry per subsequent iteration.
        """
        prior_support = (np.abs(self.prior_sample) > 0) * 1.0
        se_mse_0 = np.mean(np.abs(self.prior_sample) ** 2)
        self.se_effective_noise.append(
            self.noise_variance + (1 / self.measurement_ratio) * se_mse_0
        )
        self.se_mse.append(se_mse_0)
        self.se_pmd.append(np.mean(prior_support))
        self.se_nmse.append(1.0)
        self.se_npmd.append(1.0)

    def _se_step(self, element_threshold, group_threshold):
        """Compute one SE update from the current SE noise level and append results.

        Uses se_effective_noise[-1] as the current tau, simulates denoising on
        prior + Gaussian noise, then appends the next SE noise level and all
        per-iteration SE metrics.
        """
        tau = np.sqrt(self.se_effective_noise[-1])

        denoiser_input = self.prior_sample + (tau / np.sqrt(2)) * (
            np.random.randn(*self.prior_sample.shape)
            + 1j * np.random.randn(*self.prior_sample.shape)
        )
        denoiser_output = self.denoise(denoiser_input, element_threshold, group_threshold)

        prior_support = (np.abs(self.prior_sample) > 0) * 1.0
        denoised_support = (np.abs(denoiser_output) > 0) * 1.0
        error = denoiser_output - self.prior_sample

        se_mse = np.mean(np.abs(error) ** 2)
        self.se_mse.append(se_mse)
        self.se_effective_noise.append(
            self.noise_variance + (1 / self.measurement_ratio) * se_mse
        )
        self.se_pmd.append(np.mean(prior_support * (1 - denoised_support)))
        self.se_nmse.append(np.mean(
            np.sum(np.abs(error) ** 2, axis=1)
            / np.maximum(np.sum(np.abs(self.prior_sample) ** 2, axis=1), 1e-6)
        ))
        self.se_npmd.append(np.mean(
            np.sum(prior_support * (1 - denoised_support), axis=1)
            / np.maximum(np.sum(prior_support, axis=1), 1e-6)
        ))

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        num_iterations,
        algo='amp',
        element_penalty_scale=1.0,
        group_penalty_scale=0.5,
        warm_start=False,
        tau_scale=1.0,
        tolerance=None,
        step_size=None,
        group_threshold_freq=1,
    ):
        """Run the iterative recovery algorithm for num_iterations steps.

        Args:
            num_iterations: Maximum number of iterations to run.
            algo: Algorithm variant. One of 'amp', 'ista', 'amp_element_lasso',
                  'amp_group_lasso', 'ista_element_lasso', 'ista_group_lasso',
                  'amp_sparse_group_lasso'.
                  The _element_lasso and _group_lasso aliases override the
                  penalty scales to their pure-lasso values.
            element_penalty_scale: Scale for the element-wise soft threshold.
                  Ignored when algo alias implies a fixed value.
            group_penalty_scale: Scale for the group soft threshold.
                  Ignored when algo alias implies a fixed value.
            warm_start: If True, continue from current estimate and Onsager term
                        (and SE lists). If False, reset to zero and clear SE lists.
            tau_scale: Multiplicative scale applied to the empirical tau estimate.
            tolerance: If given, stop early when the absolute change in SE MSE
                       between consecutive iterations falls below this value.
                       num_iterations then acts as a maximum cap.
            step_size: If given, overrides self.step_size for this run. Useful for
                       using a different step size between cold and warm-start phases.
                       Always ignored for ISTA, which derives step_size from the
                       Lipschitz constant.
            group_threshold_freq: For 'amp_sparse_group_lasso' only. Integer
                  period controlling how often the group threshold is applied.
                  Group thresholding occurs every group_threshold_freq steps;
                  1 means every step, 5 means every 5th step.
                  0 disables group thresholding entirely.
        """
        # Resolve algo-mode scale overrides
        if algo in ('amp_element_lasso', 'ista_element_lasso'):
            element_penalty_scale, group_penalty_scale = 1.0, 0.0
        elif algo in ('amp_group_lasso', 'ista_group_lasso'):
            element_penalty_scale, group_penalty_scale = 0.0, 1.0
        self.algo = algo

        if not warm_start:
            self.se_effective_noise.clear()
            self.se_mse.clear()
            self.se_pmd.clear()
            self.se_nmse.clear()
            self.se_npmd.clear()
            self.actual_mse.clear()
            self.actual_pmd.clear()
            self.actual_nmse.clear()
            self.actual_nmse_std.clear()
            self.actual_npmd.clear()

        self.initialize(warm_start=warm_start)

        if step_size is not None:
            self.step_size = step_size
        if 'ista' in algo:
            max_eigen_value = (1 + 1 / np.sqrt(self.measurement_ratio)) ** 2
            self.step_size = 1 / max_eigen_value

        if not self.se_effective_noise:
            self._init_se()
            self._append_actual_metrics()  # index 0: state before any iterations

        for t in range(num_iterations):

            # Tau estimates
            se_tau = np.sqrt(self.se_effective_noise[-1])
            empirical_tau = (
                np.linalg.vector_norm(self.residual, axis=(-1, -2), ord=2, keepdims=True)
                / np.sqrt(self.num_measurements)
            )  # (batch, 1, 1)

            tau = tau_scale * empirical_tau if self.tau_estimation == 'empirical' else se_tau

            # Thresholds for the algorithm step (from tau)
            base_threshold = self.step_size * tau
            element_threshold = base_threshold * element_penalty_scale
            group_threshold = group_penalty_scale * base_threshold * np.sqrt(self.group_size)

            # Thresholds for the SE step (always from se_tau for correctness)
            se_base = self.step_size * se_tau
            se_element_threshold = se_base * element_penalty_scale
            se_group_threshold = group_penalty_scale * se_base * np.sqrt(self.group_size)

            # Algorithm step
            if algo == 'amp_sparse_group_lasso':
                use_group = group_threshold_freq > 0 and t % group_threshold_freq == 0
                algo_group = group_threshold if use_group else 0.0
                se_group = se_group_threshold if use_group else 0.0
                self._amp_step(element_threshold, algo_group)
                self._append_actual_metrics()
                self._se_step(se_element_threshold, se_group)
            elif 'ista' in algo:
                self._ista_step(element_threshold, group_threshold)
                self._append_actual_metrics()
                self._se_step(se_element_threshold, se_group_threshold)
            else:
                self._amp_step(element_threshold, group_threshold)
                self._append_actual_metrics()
                self._se_step(se_element_threshold, se_group_threshold)

            if tolerance is not None and len(self.se_mse) >= 2:
                if abs(self.se_mse[-1] - self.se_mse[-2]) < tolerance:
                    break

    # ------------------------------------------------------------------
    # Convenience properties for final-iteration scalars
    # ------------------------------------------------------------------

    @property
    def actual_nmse_final(self):
        return self.actual_nmse[-1]

    @property
    def actual_nmse_final_std(self):
        return self.actual_nmse_std[-1]

    @property
    def actual_npmd_final(self):
        return self.actual_npmd[-1]

    @property
    def se_nmse_final(self):
        return self.se_nmse[-1]

    @property
    def se_npmd_final(self):
        return self.se_npmd[-1]
