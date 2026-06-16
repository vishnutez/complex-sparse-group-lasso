import numpy as np
from complex_amp import ComplexAMP
from real_amp import RealAMP


class ComplexSGL:
    """Sparse group lasso problem setup for complex-valued compressed sensing.

    Responsibilities:
      - Define problem dimensions and sparsity structure
      - Generate sensing matrix, true signal, prior sample, and measurements
      - Instantiate ComplexAMP and RealAMP solvers

    Running the solver is done via self.complex_amp.run() and self.real_amp.run()
    directly, which supports continuation (warm_start=True) without re-instantiating.
    """

    def __init__(
        self,
        delta=0.5,
        rho=0.5,
        group_activity_ratio=0.5,
        num_samples=100,
        num_elements=100,
        num_groups=1,
        meas_noise_variance=0.0,
        step_size=1.0,
        element_penalty_scale=1.0,
        group_penalty_scale=0.5,
        tau_estimation='empirical',
        generation_mode='exact',
        set_sample_to_true=False,
    ):
        # Signal dimensions
        self.num_samples = num_samples
        self.num_elements = num_elements  # N, grows as N -> inf
        self.num_groups = num_groups      # G, fixed as N -> inf
        self.group_size = num_elements // num_groups  # grows as N -> inf

        # Sparsity parameters
        self.delta = delta                              # n/N (measurement ratio)
        self.rho = rho                                  # k/n (sparsity ratio)
        self.element_activity_ratio = rho * delta       # k/N
        self.group_activity_ratio = group_activity_ratio  # num_active_groups/num_groups
        self.element_activity_ratio_per_active_group = np.clip(
            self.element_activity_ratio / self.group_activity_ratio, 0, 1
        )

        # Derived counts
        self.num_active_groups = int(num_groups * group_activity_ratio)
        self.num_measurements = int(num_elements * delta)
        self.num_active_elements_per_group = int(
            self.group_size * self.element_activity_ratio_per_active_group
        )

        # Noise
        self.meas_noise_variance = meas_noise_variance

        # Default solver parameters (may be overridden at run() time)
        self.step_size = step_size
        self.element_penalty_scale = element_penalty_scale
        self.group_penalty_scale = group_penalty_scale
        self.tau_estimation = tau_estimation
        self.generation_mode = generation_mode

        self.set_sample_to_true = set_sample_to_true

        self.complex_amp = None
        self.real_amp = None

    def generate_instance(self):
        """Generate sensing matrix, true signal, prior sample, and noisy measurements."""

        self.sensing_matrix = (
            1 / np.sqrt(2 * self.num_measurements)
            * (
                np.random.randn(self.num_samples, self.num_measurements, self.num_elements)
                + 1j * np.random.randn(self.num_samples, self.num_measurements, self.num_elements)
            )
        )  # (batch, measurements, elements)

        self._generate_signal(role='actual')
        self._generate_signal(role='sample')

        if self.set_sample_to_true:
            self.prior_sample = self.signal_true

        self.measurements = self.sensing_matrix @ self.signal_true

        if self.meas_noise_variance > 0:
            noise = np.sqrt(self.meas_noise_variance / 2) * (
                np.random.randn(self.num_samples, self.num_measurements, 1)
                + 1j * np.random.randn(self.num_samples, self.num_measurements, 1)
            )
            self.measurements = self.measurements + noise

    def setup_amp(self):
        """Instantiate self.complex_amp and self.real_amp from the problem data.

        Algorithm variant and penalty scales are passed to .run(), so they can
        differ between a cold start and a warm continuation:

            sgl.setup_amp()
            sgl.complex_amp.run(100, algo='amp', element_penalty_scale=1.0, group_penalty_scale=0.5)
            sgl.complex_amp.run(50, algo='amp_group_lasso', warm_start=True)
        """
        self.complex_amp = ComplexAMP(
            measurements=self.measurements,
            sensing_matrix=self.sensing_matrix,
            signal_true=self.signal_true,
            prior_sample=self.prior_sample,
            group_size=self.group_size,
            noise_variance=self.meas_noise_variance,
            step_size=self.step_size,
            tau_estimation=self.tau_estimation,
        )
        self.real_amp = RealAMP(
            measurements=self.measurements,
            sensing_matrix=self.sensing_matrix,
            signal_true=self.signal_true,
            prior_sample=self.prior_sample,
            group_size=self.group_size,
            noise_variance=self.meas_noise_variance,
            step_size=self.step_size,
            tau_estimation=self.tau_estimation,
        )

    def _generate_signal(self, role='actual'):
        """Generate a sparse group-structured complex signal.

        Args:
            role: 'actual' for the true signal, 'sample' for the prior sample.
        """
        assert self.num_elements % self.num_groups == 0

        if self.generation_mode == 'exact':
            group_support = np.zeros(
                (self.num_samples, self.num_active_groups), dtype=np.int32
            )
            per_group_element_support = np.zeros(
                (self.num_samples, self.num_active_groups, self.num_active_elements_per_group),
                dtype=np.int32,
            )

            for b in range(self.num_samples):
                group_support[b] = np.random.choice(
                    self.num_groups, size=self.num_active_groups, replace=False
                )
                for g in range(self.num_active_groups):
                    per_group_element_support[b, g] = np.random.choice(
                        self.group_size, size=self.num_active_elements_per_group, replace=False
                    )

            support_mask = np.zeros(
                (self.num_samples, self.num_groups, self.group_size), dtype=bool
            )
            support_mask[
                np.arange(self.num_samples)[:, None, None],
                group_support[:, :, None],
                per_group_element_support,
            ] = True

        elif self.generation_mode == 'probabilistic':
            element_activity_prob = self.num_active_elements_per_group / self.group_size
            element_support_mask = (
                np.random.rand(self.num_samples, self.num_groups, self.group_size)
                < element_activity_prob
            )
            group_activity_prob = self.num_active_groups / self.num_groups
            group_support_mask = (
                np.random.rand(self.num_samples, self.num_groups) < group_activity_prob
            )[:, :, None]  # (B, G, 1)
            support_mask = group_support_mask * element_support_mask

        else:
            raise ValueError(f"Invalid generation_mode: {self.generation_mode}")

        active_elements = np.exp(
            1j * 2 * np.pi * np.random.rand(self.num_samples, self.num_groups, self.group_size)
        )

        signal = (
            (support_mask * active_elements)
            .reshape(self.num_samples, self.num_elements, 1)
            .astype(np.complex64)
        )

        if role == 'actual':
            self.signal_true = signal
        elif role == 'sample':
            self.prior_sample = signal
        else:
            raise ValueError(f"Invalid role: {role}")
