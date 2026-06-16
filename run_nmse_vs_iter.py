"""NMSE vs iteration experiment: compare AMP and ISTA convergence at a fixed SNR.

Output:
  results/<stem>.csv   — per-iteration NMSE for AMP and ISTA; single source of truth
"""
import numpy as np
import os

from complex_sgl import ComplexSGL


# ── Problem geometry ──────────────────────────────────────────────────────────
delta = 0.6                         # measurement ratio n/N
rho = 0.5                           # sparsity ratio k/n
group_activity_ratio = 0.6          # fraction of groups that are active
num_elements = 1000                 # signal dimension N
num_groups = 10                     # group size = num_elements / num_groups = 100
num_samples = 200                   # Monte Carlo batch size
generation_mode = 'exact'           # 'exact' (fixed counts) or 'probabilistic'

# ── Algorithm ─────────────────────────────────────────────────────────────────
element_penalty_scale = 0.8         # λ_e scale (multiplies τ)
group_penalty_scale = 0.2           # λ_g scale (multiplies τ · sqrt(group_size))
group_threshold_freq = 1            # apply group threshold every AMP step
num_iters = 200                     # max iterations (early-stopped by tolerance)
tau_estimation = 'empirical'        # 'empirical' (residual norm) or 'se' (state evolution)

# ── Experiment ─────────────────────────────────────────────────────────────────
snr_db = 30                                 # fixed SNR for convergence comparison
signal_power = rho * delta                  # E[|x_i|^2] per element
seed = 42

meas_noise_variance = signal_power / (10 ** (snr_db / 10))
step_size = 1.0 / (1 + meas_noise_variance) # noise-scaled step size

print(f'SNR = {snr_db} dB, noise variance = {meas_noise_variance:.4e}, step size = {step_size:.4f}')

np.random.seed(seed)

sgl = ComplexSGL(
    delta=delta,
    rho=rho,
    group_activity_ratio=group_activity_ratio,
    num_samples=num_samples,
    num_elements=num_elements,
    num_groups=num_groups,
    meas_noise_variance=meas_noise_variance,
    step_size=step_size,
    element_penalty_scale=element_penalty_scale,
    group_penalty_scale=group_penalty_scale,
    tau_estimation=tau_estimation,
    generation_mode=generation_mode,
)
sgl.generate_instance()
sgl.setup_amp()

# ── AMP sparse group lasso ────────────────────────────────────────────────────
sgl.complex_amp.run(
    num_iters,
    algo='amp_sparse_group_lasso',
    element_penalty_scale=element_penalty_scale,
    group_penalty_scale=group_penalty_scale,
    group_threshold_freq=group_threshold_freq,
)
nmse_amp = np.array(sgl.complex_amp.actual_nmse)
se_amp   = np.array(sgl.complex_amp.se_nmse)
print(f'AMP  final: NMSE={nmse_amp[-1]:.4e}, SE NMSE={se_amp[-1]:.4e}')

# ── ISTA sparse group lasso ───────────────────────────────────────────────────
sgl.complex_amp.run(
    num_iters,
    algo='ista',
    element_penalty_scale=element_penalty_scale,
    group_penalty_scale=group_penalty_scale,
    warm_start=False,
)
nmse_ista = np.array(sgl.complex_amp.actual_nmse)
print(f'ISTA final: NMSE={nmse_ista[-1]:.4e}')


# ── Save CSV (single source of truth) ─────────────────────────────────────────
os.makedirs('results', exist_ok=True)

stem = (
    f'nmse_vs_iter'
    f'_delta={delta}_rho={rho}_ga={group_activity_ratio}'
    f'_snr={snr_db}dB'
    f'_gps={group_penalty_scale}_eps={element_penalty_scale}'
    f'_gtr={group_threshold_freq}_seed={seed}'
)
csv_path = f'results/{stem}.csv'

# Pad shorter array to the longer length (early stopping can make them differ)
max_len = max(len(nmse_amp), len(nmse_ista))


def _pad(arr, length):
    if len(arr) < length:
        return np.concatenate([arr, np.full(length - len(arr), arr[-1])])
    return arr


nmse_amp  = _pad(nmse_amp,  max_len)
se_amp    = _pad(se_amp,    max_len)
nmse_ista = _pad(nmse_ista, max_len)

np.savetxt(
    csv_path,
    np.column_stack([np.arange(max_len), nmse_amp, se_amp, nmse_ista]),
    delimiter=',',
    header='iteration,nmse_amp,se_amp,nmse_ista',
    comments='',
    fmt=['%d', '%.8g', '%.8g', '%.8g'],
)
print(f'Saved: {csv_path}')
