# complex-sparse-group-lasso

Implementation of complex-valued Sparse Group LASSO (SGL) signal recovery using Approximate Message Passing (AMP) with non-separable denoisers, together with State Evolution (SE) predictions of AMP performance.

---

## Background

The problem is underdetermined complex-valued compressed sensing:

```
y = A x + w
```

where `y ∈ ℂⁿ` are measurements, `A ∈ ℂⁿˣᴺ` is the sensing matrix, `x ∈ ℂᴺ` is the unknown sparse signal, and `w` is complex Gaussian noise.

The signal `x` has **sparse group structure**: it is partitioned into `G` groups of equal size `N/G`. Only a fraction `r_ga` of groups are active, and within each active group only a fraction of elements are nonzero.

Recovery uses the **Sparse Group LASSO** proximal operator as an AMP denoiser:

```
prox(u) = soft_threshold_element(u, λ_e) followed by soft_threshold_group(·, λ_g)
```

The element penalty enforces individual sparsity; the group penalty enforces group sparsity. Setting `λ_e = 0` gives pure Group LASSO; setting `λ_g = 0` gives pure (element) LASSO.

**State Evolution (SE)** tracks theoretical predictions of AMP's NMSE and misdetection probability per iteration, validated against empirical batch averages.

---

## File Structure

```
complex-sparse-group-lasso/
├── complex_sgl.py         # Problem setup: dimensions, signal generation, measurements
├── complex_amp.py         # ComplexAMP solver (complex-valued AMP + SE)
├── real_amp.py            # RealAMP solver (complex inputs lifted to real via isomorphism)
├── utils.py               # _get_stable_alpha(), load_results()
├── run_nmse_vs_snr.py     # Experiment: NMSE vs SNR, four algorithms compared
├── run_nmse_vs_iter.py    # Experiment: NMSE vs iteration, AMP vs ISTA at fixed SNR
├── environment.yaml       # Conda environment (Python 3.10, numpy, scipy, matplotlib)
├── docs/
│   └── 2026-06-15-clean-implementation-plan.md
├── LICENSE
└── README.md
```

### Core classes

| Class | File | Role |
|-------|------|------|
| `ComplexSGL` | `complex_sgl.py` | Problem factory: generates `A`, `x`, `y`; holds solver instances |
| `ComplexAMP` | `complex_amp.py` | AMP/ISTA solver operating on complex arrays directly |
| `RealAMP` | `real_amp.py` | AMP/ISTA solver that internally lifts complex inputs to real via block-real isomorphism |

---

## Setup

```bash
conda env create -f environment.yaml
conda activate csgl
```

---

## Running the Experiments

### NMSE vs SNR

Compares Lasso, Group Lasso, Complex SGL, and Real SGL across an SNR sweep:

```bash
python run_nmse_vs_snr.py
```

**Default configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `delta` | 0.6 | Measurement ratio n/N |
| `rho` | 0.5 | Sparsity ratio k/n |
| `group_activity_ratio` | 0.6 | Fraction of groups that are active |
| `num_elements` | 1000 | Signal dimension N |
| `num_groups` | 10 | Number of groups G (group size = 100) |
| `num_samples` | 200 | Batch size for Monte Carlo averaging |
| `snr_db_values` | 10, 15, 20, 25, 30, 35, 40 dB | SNR sweep |
| `signal_power` | `rho × delta = 0.3` | Expected signal power per element |
| `element_penalty_scale` | 0.8 | λ_e scale (multiplies τ) |
| `group_penalty_scale` | 0.2 | λ_g scale (multiplies τ√(group_size)) |
| `group_threshold_freq` | 5 | Group threshold applied every 5 AMP steps |
| `num_iters` | 200 | Max iterations (early-stopped by SE tolerance 1e-5) |
| `step_size` | 1.0 | AMP step size α |
| `tau_estimation` | `'empirical'` | τ estimated from residual norm |
| `generation_mode` | `'exact'` | Exact sparsity counts (not probabilistic) |
| `seed` | 42 | Random seed |

**Outputs:**
- `results/nmse_vs_snr_<params>.csv` — all algorithm results in one file (single source of truth)
- `plots/nmse_vs_snr_<params>.png` — matplotlib figure
- `plots/nmse_vs_snr_<params>.tex` — standalone pgfplots figure (references the CSV directly via `col sep=comma`)

CSV columns: `snr_db`, `nmse_lasso`, `se_lasso`, `nmse_group_lasso`, `se_group_lasso`, `nmse_sgl`, `se_sgl`, `std_sgl`, `nmse_real_sgl`, `se_real_sgl`

### NMSE vs Iteration

Compares AMP-SGL and ISTA-SGL convergence at a fixed SNR of 30 dB:

```bash
python run_nmse_vs_iter.py
```

**Default configuration** (same as above, plus):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `snr_db` | 30 | Fixed SNR |
| `step_size` | `1 / (1 + σ²_w)` | Noise-scaled step size |
| `group_threshold_freq` | 1 | Group threshold every AMP step |

**Output:** `results/nmse_vs_iter_<params>.csv` — per-iteration NMSE for AMP and ISTA.

CSV columns: `iteration`, `nmse_amp`, `se_amp`, `nmse_ista`

---

## API Reference

### `ComplexSGL`

```python
from complex_sgl import ComplexSGL

sgl = ComplexSGL(
    delta=0.6,               # measurement ratio n/N
    rho=0.5,                 # sparsity ratio k/n
    group_activity_ratio=0.6,# fraction of active groups
    num_samples=200,         # batch size
    num_elements=1000,       # signal dimension N
    num_groups=10,           # number of groups G
    meas_noise_variance=1e-3,# measurement noise σ²_w
    step_size=1.0,           # AMP step size α
    element_penalty_scale=0.8,
    group_penalty_scale=0.2,
    tau_estimation='empirical',
    generation_mode='exact', # 'exact' or 'probabilistic'
)

sgl.generate_instance()   # draw A, x, y
sgl.setup_amp()           # create sgl.complex_amp and sgl.real_amp
```

### `ComplexAMP.run()`

```python
sgl.complex_amp.run(
    num_iterations=200,
    algo='amp_sparse_group_lasso', # see table below
    element_penalty_scale=0.8,
    group_penalty_scale=0.2,
    group_threshold_freq=5,        # amp_sparse_group_lasso only
    warm_start=False,              # True to continue from current state
    tolerance=1e-5,                # early stop on SE MSE change
    step_size=None,                # override step size for this run
)
```

**`algo` options:**

| `algo` | Element threshold | Group threshold | Notes |
|--------|------------------|-----------------|-------|
| `'amp'` | yes | yes | AMP with both penalties |
| `'amp_element_lasso'` | yes (scale=1.0) | no | Pure LASSO via AMP |
| `'amp_group_lasso'` | no | yes (scale=1.0) | Pure Group LASSO via AMP |
| `'amp_sparse_group_lasso'` | yes | yes, every `group_threshold_freq` steps | Sparse Group LASSO via AMP |
| `'ista'` | yes | yes | ISTA with both penalties |
| `'ista_element_lasso'` | yes (scale=1.0) | no | Pure LASSO via ISTA |
| `'ista_group_lasso'` | no | yes (scale=1.0) | Pure Group LASSO via ISTA |

### Reading results

```python
# Per-iteration arrays (length = 1 + num_iterations_run)
sgl.complex_amp.actual_nmse    # list of floats
sgl.complex_amp.se_nmse        # list of floats (SE prediction)
sgl.complex_amp.actual_npmd    # normalized probability of misdetection

# Final-iteration scalars
sgl.complex_amp.actual_nmse_final      # float
sgl.complex_amp.actual_nmse_final_std  # std across batch
sgl.complex_amp.se_nmse_final          # float
```

The same attributes exist on `sgl.real_amp`.

### Loading saved results

```python
from utils import load_results

data = load_results('results/nmse_vs_snr_delta=0.6_rho=0.5_...csv')
# data is a dict of {column_name: np.ndarray}

snr   = data['snr_db']
nmse  = data['nmse_sgl']
se    = data['se_sgl']
std   = data['std_sgl']
```

---

## Expected Results

With the default configuration (`δ=0.6`, `ρ=0.5`, `r_ga=0.6`, `λ_e=0.8`, `λ_g=0.2`):

- At **30 dB SNR**, Complex SGL reaches NMSE ≈ 1–5 × 10⁻³ after convergence.
- **Complex SGL** outperforms both pure Lasso and pure Group Lasso across all SNR values because the joint penalty exploits both element and group sparsity simultaneously.
- **Real SGL** (the complex system lifted to its real equivalent) serves as a baseline; it is expected to match or slightly underperform Complex SGL.
- **State Evolution** (dashed lines) closely tracks the empirical AMP NMSE, validating the theoretical predictions.
- AMP converges in ~20–50 iterations; ISTA requires ~100–200 iterations for comparable NMSE.

---

## Warm-Start Example

Run the same instance with two different algorithm phases:

```python
sgl.generate_instance()
sgl.setup_amp()

# Phase 1: element lasso to get close
sgl.complex_amp.run(100, algo='amp_element_lasso')

# Phase 2: switch to sparse group lasso, continuing from current estimate
sgl.complex_amp.run(100, algo='amp_sparse_group_lasso',
                    element_penalty_scale=0.8, group_penalty_scale=0.2,
                    warm_start=True)

print(sgl.complex_amp.actual_nmse_final)
```

---

## Signal Model

Active element magnitudes are unit-norm with uniform random phase: `x_i = e^{j φ_i}` for `φ_i ~ Uniform[0, 2π)`. The sensing matrix uses i.i.d. entries `A_{ij} ~ (1/√(2n)) CN(0,1)`. Measurement noise is `w ~ (σ_w / √2) CN(0, I)`.

Signal power is `E[|x_i|²] = ρδ` (element activity ratio), so `σ²_w = ρδ / 10^{SNR/10}`.
