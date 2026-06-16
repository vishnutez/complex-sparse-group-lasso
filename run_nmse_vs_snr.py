"""NMSE vs SNR experiment: compare Lasso, Group Lasso, Complex SGL, and Real SGL.

Outputs (written to results/ and plots/):
  results/<stem>.csv   — all algorithm results; single source of truth
  plots/<stem>.png     — matplotlib figure
  plots/<stem>.tex     — standalone pgfplots figure referencing the CSV
"""
import numpy as np
import matplotlib.pyplot as plt
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
group_threshold_freq = 5            # apply group threshold every N AMP steps
step_size = 1.0                     # AMP step size α
num_iters = 200                     # max iterations (early-stopped by tolerance)
tau_estimation = 'empirical'        # 'empirical' (residual norm) or 'se' (state evolution)
set_sample_to_true = False          # if True, prior sample = true signal (oracle test)

# ── Experiment sweep ──────────────────────────────────────────────────────────
snr_db_values = np.arange(10, 41, 5)        # SNR values in dB
signal_power = rho * delta                  # E[|x_i|^2] per element
seed = 42

nmse_lasso = []
nmse_pgi_lasso = []
nmse_group_lasso = []
nmse_sgl = []
nmse_real_sgl = []

se_lasso = []
se_pgi_lasso = []
se_group_lasso = []
se_sgl = []
se_real_sgl = []

std_sgl = []

np.random.seed(seed)

algos_to_run = ['lasso', 'group_lasso', 'sgl', 'real_sgl']

for snr_db in snr_db_values:
    meas_noise_variance = signal_power / (10 ** (snr_db / 10))
    step_size = 1.0

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
        set_sample_to_true=set_sample_to_true,
    )
    sgl.generate_instance()
    sgl.setup_amp()

    if 'lasso' in algos_to_run:
        sgl.complex_amp.run(num_iters, algo='amp_element_lasso', tolerance=1e-5)
        nmse_lasso.append(sgl.complex_amp.actual_nmse_final)
        se_lasso.append(sgl.complex_amp.se_nmse_final)

    if 'group_lasso' in algos_to_run:
        sgl.complex_amp.run(
            num_iters, algo='amp_group_lasso',
            element_penalty_scale=element_penalty_scale,
            group_penalty_scale=group_penalty_scale,
            tolerance=1e-5,
        )
        nmse_group_lasso.append(sgl.complex_amp.actual_nmse_final)
        se_group_lasso.append(sgl.complex_amp.se_nmse_final)

    if 'sgl' in algos_to_run:
        sgl.complex_amp.run(
            num_iters, algo='amp_sparse_group_lasso',
            element_penalty_scale=element_penalty_scale,
            group_penalty_scale=group_penalty_scale,
            group_threshold_freq=group_threshold_freq,
            tolerance=1e-5,
        )
        nmse_sgl.append(sgl.complex_amp.actual_nmse_final)
        se_sgl.append(sgl.complex_amp.se_nmse_final)
        std_sgl.append(sgl.complex_amp.actual_nmse_final_std)

    if 'real_sgl' in algos_to_run:
        sgl.real_amp.run(
            num_iters, algo='amp_sparse_group_lasso',
            element_penalty_scale=element_penalty_scale,
            group_penalty_scale=group_penalty_scale,
            group_threshold_freq=group_threshold_freq,
            tolerance=1e-5,
        )
        nmse_real_sgl.append(sgl.real_amp.actual_nmse_final)
        se_real_sgl.append(sgl.real_amp.se_nmse_final)

    if 'pgi_lasso' in algos_to_run:
        delta_pgi = delta / group_activity_ratio
        num_elements_pgi = int(num_elements * group_activity_ratio)
        sgl_pgi = ComplexSGL(
            delta=delta_pgi,
            rho=rho,
            group_activity_ratio=1.0,
            num_samples=num_samples,
            num_elements=num_elements_pgi,
            num_groups=num_groups,
            meas_noise_variance=meas_noise_variance,
            step_size=step_size,
            tau_estimation=tau_estimation,
            generation_mode=generation_mode,
        )
        sgl_pgi.generate_instance()
        sgl_pgi.setup_amp()
        sgl_pgi.complex_amp.run(num_iters, algo='amp_element_lasso', tolerance=1e-5)
        nmse_pgi_lasso.append(sgl_pgi.complex_amp.actual_nmse_final)
        se_pgi_lasso.append(sgl_pgi.complex_amp.se_nmse_final)

    display_nmse = ''
    if 'lasso' in algos_to_run:
        display_nmse += f'Lasso={nmse_lasso[-1]:.4e} SE={se_lasso[-1]:.4e}  '
    if 'group_lasso' in algos_to_run:
        display_nmse += f'GroupLasso={nmse_group_lasso[-1]:.4e} SE={se_group_lasso[-1]:.4e}  '
    if 'sgl' in algos_to_run:
        display_nmse += f'SGL={nmse_sgl[-1]:.4e} SE={se_sgl[-1]:.4e}  '
    if 'real_sgl' in algos_to_run:
        display_nmse += f'RealSGL={nmse_real_sgl[-1]:.4e} SE={se_real_sgl[-1]:.4e}'
    print(f'SNR={snr_db:2d}dB  {display_nmse}')


# ── Save CSV (single source of truth) ─────────────────────────────────────────
os.makedirs('results', exist_ok=True)

stem = (
    f'nmse_vs_snr'
    f'_delta={delta}_rho={rho}_ga={group_activity_ratio}'
    f'_gps={group_penalty_scale}_eps={element_penalty_scale}'
    f'_gtr={group_threshold_freq}_gen_mode={generation_mode}_seed={seed}'
)
csv_path = f'results/{stem}.csv'

columns = {'snr_db': snr_db_values}
if 'lasso' in algos_to_run:
    columns['nmse_lasso'] = nmse_lasso
    columns['se_lasso'] = se_lasso
if 'group_lasso' in algos_to_run:
    columns['nmse_group_lasso'] = nmse_group_lasso
    columns['se_group_lasso'] = se_group_lasso
if 'sgl' in algos_to_run:
    columns['nmse_sgl'] = nmse_sgl
    columns['se_sgl'] = se_sgl
    columns['std_sgl'] = std_sgl
if 'real_sgl' in algos_to_run:
    columns['nmse_real_sgl'] = nmse_real_sgl
    columns['se_real_sgl'] = se_real_sgl
if 'pgi_lasso' in algos_to_run:
    columns['nmse_pgi_lasso'] = nmse_pgi_lasso
    columns['se_pgi_lasso'] = se_pgi_lasso

np.savetxt(
    csv_path,
    np.column_stack([np.array(v, dtype=float) for v in columns.values()]),
    delimiter=',',
    header=','.join(columns.keys()),
    comments='',
    fmt='%.8g',
)
print(f'Saved: {csv_path}')


# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE   = '#264A73'
ORANGE = '#CC5500'
GRAY   = '#777777'

ses_to_plot = ['sgl']
stds_to_plot = ['sgl']
algos_to_plot = ['group_lasso', 'lasso', 'real_sgl', 'sgl']

font_size = 16
lw = 2
plt.rcParams.update({'font.size': font_size, 'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(10, 6))

if 'group_lasso' in algos_to_plot:
    ax.semilogy(snr_db_values, nmse_group_lasso,
                color=GRAY, linewidth=lw, marker='^', label='Group Lasso')
    if 'group_lasso' in ses_to_plot:
        ax.semilogy(snr_db_values, se_group_lasso,
                    color=GRAY, linewidth=lw, marker='^', linestyle='--', label='Group Lasso SE')

if 'real_sgl' in algos_to_plot:
    ax.semilogy(snr_db_values, nmse_real_sgl,
                color='black', linewidth=lw, marker='s', label='Real SGL')
    if 'real_sgl' in ses_to_plot:
        ax.semilogy(snr_db_values, se_real_sgl,
                    color='black', linewidth=lw, marker='s', linestyle='--', label='Real SGL SE')

if 'lasso' in algos_to_plot:
    ax.semilogy(snr_db_values, nmse_lasso,
                color=BLUE, linewidth=lw, marker='o', label='Lasso')
    if 'lasso' in ses_to_plot:
        ax.semilogy(snr_db_values, se_lasso,
                    color=BLUE, linewidth=lw, marker='o', linestyle='--', label='Lasso SE')

if 'pgi_lasso' in algos_to_plot:
    ax.semilogy(snr_db_values, nmse_pgi_lasso,
                color='red', linewidth=lw, marker='s', label='PGI Lasso')

if 'sgl' in algos_to_plot:
    if 'sgl' in stds_to_plot and len(std_sgl) > 0:
        ax.errorbar(snr_db_values, nmse_sgl, yerr=std_sgl,
                    color=ORANGE, linewidth=lw, marker='D', label='Complex SGL', capsize=4)
    else:
        ax.semilogy(snr_db_values, nmse_sgl,
                    color=ORANGE, linewidth=lw, marker='D', label='Complex SGL')
    if 'sgl' in ses_to_plot:
        ax.semilogy(snr_db_values, se_sgl,
                    color=ORANGE, linewidth=lw, marker='D', linestyle='--', label='Complex SGL SE')

ax.set_xlabel('SNR (dB)')
ax.set_ylabel('NMSE')
handles, labels = ax.get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))
legend_order = ['Group Lasso', 'Complex SGL', 'Real SGL', 'Complex SGL SE', 'Lasso']
ordered = [(label_to_handle[l], l) for l in legend_order if l in label_to_handle]
ax.legend([h for h, _ in ordered], [l for _, l in ordered],
          loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(True, alpha=0.5)
ax.set_title(
    r'$\delta$={}, $\rho$={}, $r_{{\mathrm{{ga}}}}$={}, $\lambda_g$={}, $\kappa$={}'.format(
        delta, rho, group_activity_ratio, group_penalty_scale, group_threshold_freq
    )
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)

os.makedirs('plots', exist_ok=True)
plot_path = f'plots/{stem}.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Saved: {plot_path}')
plt.close(fig)


# ── pgfplots .tex (references the CSV directly) ───────────────────────────────
tex_path = f'plots/{stem}.tex'
csv_name = os.path.basename(csv_path)

addplot_blocks = []
if 'group_lasso' in algos_to_plot:
    addplot_blocks.append([
        r'\addplot[color=mygray, line width=2pt, mark=triangle*]',
        f'    table[x=snr_db, y=nmse_group_lasso, col sep=comma] {{../results/{csv_name}}};',
        r'\addlegendentry{Group Lasso}',
    ])
if 'lasso' in algos_to_plot:
    addplot_blocks.append([
        r'\addplot[color=myblue, line width=2pt, mark=*]',
        f'    table[x=snr_db, y=nmse_lasso, col sep=comma] {{../results/{csv_name}}};',
        r'\addlegendentry{Lasso}',
    ])
if 'real_sgl' in algos_to_plot:
    addplot_blocks.append([
        r'\addplot[color=black, line width=2pt, mark=square*]',
        f'    table[x=snr_db, y=nmse_real_sgl, col sep=comma] {{../results/{csv_name}}};',
        r'\addlegendentry{Real SGL}',
    ])
if 'sgl' in algos_to_plot:
    if 'sgl' in stds_to_plot and len(std_sgl) > 0:
        addplot_blocks.append([
            r'\addplot[color=myorange, line width=2pt, mark=diamond*,',
            r'    error bars/.cd, y dir=both, y explicit]',
            f'    table[x=snr_db, y=nmse_sgl, y error=std_sgl, col sep=comma] {{../results/{csv_name}}};',
            r'\addlegendentry{Complex SGL}',
        ])
    else:
        addplot_blocks.append([
            r'\addplot[color=myorange, line width=2pt, mark=diamond*]',
            f'    table[x=snr_db, y=nmse_sgl, col sep=comma] {{../results/{csv_name}}};',
            r'\addlegendentry{Complex SGL}',
        ])
    if 'sgl' in ses_to_plot:
        addplot_blocks.append([
            r'\addplot[color=myorange, line width=2pt, mark=diamond*, dashed]',
            f'    table[x=snr_db, y=se_sgl, col sep=comma] {{../results/{csv_name}}};',
            r'\addlegendentry{Complex SGL SE}',
        ])

title_tex = (
    rf'$\delta={delta}$, $\rho={rho}$, '
    rf'$r_{{\mathrm{{ga}}}}={group_activity_ratio}$, '
    rf'$\lambda_g={group_penalty_scale}$, $\kappa={group_threshold_freq}$'
)

tex_lines = [
    r'\documentclass[tikz]{standalone}',
    r'\usepackage{pgfplots}',
    r'\pgfplotsset{compat=1.18}',
    r'\usepackage{xcolor}',
    '',
    r'\definecolor{myblue}{HTML}{264A73}',
    r'\definecolor{myorange}{HTML}{CC5500}',
    r'\definecolor{mygray}{HTML}{777777}',
    '',
    r'\begin{document}',
    r'\begin{tikzpicture}',
    r'\begin{semilogyaxis}[',
    r'    xlabel={SNR (dB)},',
    r'    ylabel={NMSE},',
    f'    title={{{title_tex}}},',
    r'    grid=both,',
    r'    minor grid style={opacity=0.25},',
    r'    major grid style={opacity=0.5},',
    r'    axis line style={rounded corners=4pt},',
    r'    legend style={',
    r'        at={(0.5,-0.25)},',
    r'        anchor=north,',
    r'        legend columns=3,',
    r'        rounded corners=4pt,',
    r'        /tikz/column sep=1.5em,',
    r'        row sep=0.4em,',
    r'    },',
    r'    width=12cm,',
    r'    height=8cm,',
    r'    font=\large,',
    r']',
    '',
]
for i, block in enumerate(addplot_blocks):
    tex_lines.extend(block)
    if i < len(addplot_blocks) - 1:
        tex_lines.append('')
tex_lines += [
    '',
    r'\end{semilogyaxis}',
    r'\end{tikzpicture}',
    r'\end{document}',
]

with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')
print(f'Saved: {tex_path}')
