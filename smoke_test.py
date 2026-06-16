"""Smoke test: verify all components work end-to-end with tiny dimensions.

Runs in under 30 seconds. Checks:
  1. Imports
  2. ComplexSGL signal generation (shapes, dtype)
  3. ComplexAMP — all seven algo variants, metric list lengths, warm start
  4. RealAMP — basic run and metrics
  5. CSV save / load roundtrip (load_results)
  6. utils._get_stable_alpha
  7. End-to-end NMSE vs SNR (2 SNR points, mini dims)
  8. End-to-end NMSE vs iter (AMP + ISTA, mini dims)
"""
import os
import sys
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

failures = []

def check(name, cond, msg=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}' + (f'  ({msg})' if msg else ''))
        failures.append(name)

def section(title):
    print(f'\n[{len(failures) == 0 and "·" or "!"}] {title}')


# ── Tiny problem dimensions used throughout ───────────────────────────────────
N  = 100    # signal dimension
G  = 5      # groups  →  group_size = 20
B  = 10     # batch (Monte Carlo samples)
T  = 5      # AMP/ISTA iterations

delta = 0.6
rho   = 0.5
ga    = 0.6
eps   = 0.8   # element_penalty_scale
lam_g = 0.2   # group_penalty_scale
n     = int(N * delta)          # 60 measurements
signal_power = rho * delta      # 0.3


# ── 1. Imports ────────────────────────────────────────────────────────────────
section('Imports')
try:
    from complex_sgl import ComplexSGL
    from complex_amp import ComplexAMP
    from real_amp import RealAMP
    from utils import load_results, _get_stable_alpha
    print('  PASS  all imports')
except Exception as e:
    print(f'  FAIL  imports: {e}')
    sys.exit(1)


# ── 2. ComplexSGL — signal generation ────────────────────────────────────────
section('ComplexSGL — signal generation')
try:
    sgl = ComplexSGL(
        delta=delta, rho=rho, group_activity_ratio=ga,
        num_samples=B, num_elements=N, num_groups=G,
        meas_noise_variance=1e-2, step_size=1.0,
        element_penalty_scale=eps, group_penalty_scale=lam_g,
        tau_estimation='empirical', generation_mode='exact',
    )
    sgl.generate_instance()

    check('signal_true shape',    sgl.signal_true.shape    == (B, N, 1))
    check('prior_sample shape',   sgl.prior_sample.shape   == (B, N, 1))
    check('sensing_matrix shape', sgl.sensing_matrix.shape == (B, n, N))
    check('measurements shape',   sgl.measurements.shape   == (B, n, 1))
    check('signal is complex',    np.iscomplexobj(sgl.signal_true))
    check('signal is sparse',     np.mean(np.abs(sgl.signal_true) > 0) < 1.0)
except Exception as e:
    print(f'  FAIL  generate_instance: {e}')


# ── 3. ComplexAMP — all algo variants ────────────────────────────────────────
section('ComplexAMP — algo variants, metric lengths, warm start')
try:
    sgl.setup_amp()
    amp = sgl.complex_amp

    algos = [
        'amp', 'amp_element_lasso', 'amp_group_lasso', 'amp_sparse_group_lasso',
        'ista', 'ista_element_lasso', 'ista_group_lasso',
    ]
    for algo in algos:
        try:
            amp.run(T, algo=algo, element_penalty_scale=eps, group_penalty_scale=lam_g)
            check(f'{algo}: list length == T+1',  len(amp.actual_nmse) == T + 1)
            check(f'{algo}: actual_nmse finite',  np.isfinite(amp.actual_nmse_final))
            check(f'{algo}: se_nmse finite',      np.isfinite(amp.se_nmse_final))
            check(f'{algo}: nmse_std finite',     np.isfinite(amp.actual_nmse_final_std))
        except Exception as e:
            print(f'  FAIL  {algo}: {e}')
            failures.append(algo)

    # warm start should extend lists, not reset them
    amp.run(T, algo='amp', warm_start=False)
    before = len(amp.actual_nmse)
    amp.run(T, algo='amp', warm_start=True)
    check('warm_start extends list', len(amp.actual_nmse) == before + T)

except Exception as e:
    print(f'  FAIL  ComplexAMP setup: {e}')


# ── 4. RealAMP ────────────────────────────────────────────────────────────────
section('RealAMP — basic run')
try:
    ramp = sgl.real_amp
    ramp.run(T, algo='amp_sparse_group_lasso',
             element_penalty_scale=eps, group_penalty_scale=lam_g)
    check('list length == T+1',  len(ramp.actual_nmse) == T + 1)
    check('actual_nmse finite',  np.isfinite(ramp.actual_nmse_final))
    check('se_nmse finite',      np.isfinite(ramp.se_nmse_final))
    check('nmse_std finite',     np.isfinite(ramp.actual_nmse_final_std))
except Exception as e:
    print(f'  FAIL  RealAMP: {e}')


# ── 5. CSV save / load roundtrip ──────────────────────────────────────────────
section('CSV save / load roundtrip')
try:
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/_smoke_test_roundtrip.csv'

    snr_ref  = np.array([20.0, 30.0])
    nmse_ref = np.array([0.12, 0.034])
    se_ref   = np.array([0.11, 0.031])
    std_ref  = np.array([0.01, 0.004])

    np.savetxt(
        csv_path,
        np.column_stack([snr_ref, nmse_ref, se_ref, std_ref]),
        delimiter=',',
        header='snr_db,nmse_sgl,se_sgl,std_sgl',
        comments='',
        fmt='%.8g',
    )
    check('file written', os.path.exists(csv_path))

    data = load_results(csv_path)
    expected_cols = {'snr_db', 'nmse_sgl', 'se_sgl', 'std_sgl'}
    check('all columns present', expected_cols.issubset(data.keys()))
    check('snr_db roundtrips',   np.allclose(data['snr_db'],   snr_ref))
    check('nmse_sgl roundtrips', np.allclose(data['nmse_sgl'], nmse_ref))
    check('std_sgl roundtrips',  np.allclose(data['std_sgl'],  std_ref))

    os.remove(csv_path)
except Exception as e:
    print(f'  FAIL  CSV roundtrip: {e}')


# ── 6. utils ──────────────────────────────────────────────────────────────────
section('utils._get_stable_alpha')
try:
    alpha = _get_stable_alpha(delta=0.6, kappa=2)
    check('returns positive finite float', np.isfinite(alpha) and alpha > 0)
except Exception as e:
    print(f'  FAIL  _get_stable_alpha: {e}')


# ── 7. End-to-end: NMSE vs SNR (mini) ────────────────────────────────────────
section('End-to-end: NMSE vs SNR  (2 SNR points, mini dims)')
try:
    np.random.seed(0)
    snr_db_values = np.array([20.0, 30.0])
    nmse_sgl, se_sgl, std_sgl = [], [], []

    for snr_db in snr_db_values:
        meas_noise_variance = signal_power / (10 ** (snr_db / 10))
        s = ComplexSGL(
            delta=delta, rho=rho, group_activity_ratio=ga,
            num_samples=B, num_elements=N, num_groups=G,
            meas_noise_variance=meas_noise_variance, step_size=1.0,
            element_penalty_scale=eps, group_penalty_scale=lam_g,
            tau_estimation='empirical', generation_mode='exact',
        )
        s.generate_instance()
        s.setup_amp()
        s.complex_amp.run(T, algo='amp_sparse_group_lasso',
                          element_penalty_scale=eps, group_penalty_scale=lam_g,
                          group_threshold_freq=5)
        nmse_sgl.append(s.complex_amp.actual_nmse_final)
        se_sgl.append(s.complex_amp.se_nmse_final)
        std_sgl.append(s.complex_amp.actual_nmse_final_std)

    csv_path = 'results/_smoke_nmse_vs_snr.csv'
    columns = {
        'snr_db': snr_db_values, 'nmse_sgl': nmse_sgl,
        'se_sgl': se_sgl, 'std_sgl': std_sgl,
    }
    np.savetxt(
        csv_path,
        np.column_stack([np.array(v, dtype=float) for v in columns.values()]),
        delimiter=',', header=','.join(columns.keys()), comments='', fmt='%.8g',
    )
    data = load_results(csv_path)
    check('CSV columns present',  {'snr_db', 'nmse_sgl', 'se_sgl', 'std_sgl'}.issubset(data.keys()))
    check('correct number of rows', len(data['snr_db']) == len(snr_db_values))
    check('all NMSE values finite', np.all(np.isfinite(data['nmse_sgl'])))
    check('all SE values finite',   np.all(np.isfinite(data['se_sgl'])))

    os.remove(csv_path)
except Exception as e:
    print(f'  FAIL  NMSE vs SNR end-to-end: {e}')


# ── 8. End-to-end: NMSE vs iter (mini) ───────────────────────────────────────
section('End-to-end: NMSE vs iter  (AMP + ISTA, mini dims)')
try:
    np.random.seed(0)
    meas_noise_variance = signal_power / (10 ** (30 / 10))
    step_size = 1.0 / (1 + meas_noise_variance)

    s = ComplexSGL(
        delta=delta, rho=rho, group_activity_ratio=ga,
        num_samples=B, num_elements=N, num_groups=G,
        meas_noise_variance=meas_noise_variance, step_size=step_size,
        element_penalty_scale=eps, group_penalty_scale=lam_g,
        tau_estimation='empirical', generation_mode='exact',
    )
    s.generate_instance()
    s.setup_amp()

    s.complex_amp.run(T, algo='amp_sparse_group_lasso',
                      element_penalty_scale=eps, group_penalty_scale=lam_g,
                      group_threshold_freq=1)
    nmse_amp = np.array(s.complex_amp.actual_nmse)
    se_amp   = np.array(s.complex_amp.se_nmse)

    s.complex_amp.run(T, algo='ista',
                      element_penalty_scale=eps, group_penalty_scale=lam_g)
    nmse_ista = np.array(s.complex_amp.actual_nmse)

    max_len = max(len(nmse_amp), len(nmse_ista))

    def _pad(arr, length):
        return np.concatenate([arr, np.full(length - len(arr), arr[-1])]) if len(arr) < length else arr

    nmse_amp  = _pad(nmse_amp,  max_len)
    se_amp    = _pad(se_amp,    max_len)
    nmse_ista = _pad(nmse_ista, max_len)

    csv_path = 'results/_smoke_nmse_vs_iter.csv'
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(max_len), nmse_amp, se_amp, nmse_ista]),
        delimiter=',', header='iteration,nmse_amp,se_amp,nmse_ista',
        comments='', fmt=['%d', '%.8g', '%.8g', '%.8g'],
    )
    data = load_results(csv_path)
    check('CSV columns present',     {'iteration', 'nmse_amp', 'se_amp', 'nmse_ista'}.issubset(data.keys()))
    check('correct number of rows',  len(data['iteration']) == max_len)
    check('all AMP values finite',   np.all(np.isfinite(data['nmse_amp'])))
    check('all ISTA values finite',  np.all(np.isfinite(data['nmse_ista'])))
    check('iteration column is 0..T', list(data['iteration'].astype(int)) == list(range(max_len)))

    os.remove(csv_path)
except Exception as e:
    print(f'  FAIL  NMSE vs iter end-to-end: {e}')


# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"─" * 52}')
if failures:
    print(f'FAILED — {len(failures)} check(s): {failures}')
    sys.exit(1)
else:
    print(f'All checks passed.')
