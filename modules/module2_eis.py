"""
Module 2: Electrochemical Impedance Spectroscopy (EIS) Analysis
===============================================================
Uses REAL NASA PCOE Battery Dataset — impedance measurements.
NASA Ames Research Center, Prognostics Center of Excellence.

Dataset: B0005, B0006, B0007, B0018
- 278 real EIS measurements per cell across full cycle life
- EIS frequency sweep: 0.1 Hz to 5 kHz
- Randles circuit parameters extracted by NASA:
    Re  = electrolyte/ohmic resistance (Ohms)
    Rct = charge transfer resistance (Ohms)
- Full complex impedance spectrum per measurement

Source:
    Saha, B. & Goebel, K. (2007). Battery Data Set.
    NASA Ames Prognostics Data Repository.

ICD Context:
    EIS is used in battery qualification protocols for implantable
    devices. Rising charge transfer resistance (Rct) indicates SEI
    thickening and active material loss. Impedance rise PRECEDES
    capacity fade by many cycles — for a 7-10 year implanted device,
    early detection is the only clinically meaningful detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import scipy.io as sio

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────
# STEP 1: LOAD REAL NASA EIS DATA
# ─────────────────────────────────────────────

def load_eis_data(filepath, cell_name):
    """
    Load real EIS measurements from NASA .mat file.

    For each impedance cycle extracts:
    - Re:  electrolyte resistance (Ohms)
    - Rct: charge transfer resistance (Ohms)
    - Full complex Battery_impedance spectrum
    - Cycle index within the full experiment

    Also loads discharge capacity to correlate
    impedance rise with capacity fade.
    """
    mat = sio.loadmat(filepath)
    battery = mat[cell_name]
    cycles = battery['cycle'][0, 0]

    impedance_data = []
    discharge_data = []
    discharge_count = 0
    impedance_count = 0

    for i in range(cycles.shape[1]):
        cycle_type = str(cycles[0, i]['type'][0])
        data = cycles[0, i]['data'][0, 0]

        if cycle_type == 'discharge':
            discharge_count += 1
            capacity = float(data['Capacity'][0, 0])
            discharge_data.append({
                'discharge_cycle': discharge_count,
                'capacity_Ah': capacity,
                'cycle_index': i
            })

        elif cycle_type == 'impedance':
            impedance_count += 1

            # Real measured Re and Rct from NASA
            Re  = float(data['Re'][0, 0])
            Rct = float(data['Rct'][0, 0])

            # Full complex impedance spectrum
            Z_complex = data['Battery_impedance'].flatten()

            impedance_data.append({
                'impedance_count': impedance_count,
                'cycle_index': i,
                'Re': Re,
                'Rct': Rct,
                'Z_real': Z_complex.real,
                'Z_imag': Z_complex.imag
            })

    df_imp = pd.DataFrame([{
        'impedance_count': d['impedance_count'],
        'cycle_index':     d['cycle_index'],
        'Re':              d['Re'],
        'Rct':             d['Rct']
    } for d in impedance_data])

    df_dis = pd.DataFrame(discharge_data)

    # Store full spectra separately
    spectra = [(d['impedance_count'],
                d['Z_real'],
                d['Z_imag'])
               for d in impedance_data]

    return df_imp, df_dis, spectra


def load_all_eis():
    """
    Load EIS data for all 4 NASA cells.
    """
    cells = {
        'B0005': 'data/B0005.mat',
        'B0006': 'data/B0006.mat',
        'B0007': 'data/B0007.mat',
        'B0018': 'data/B0018.mat',
    }

    all_imp  = {}
    all_dis  = {}
    all_spec = {}

    for cell_name, filepath in cells.items():
        print(f"  Loading {cell_name}...", end="")
        df_imp, df_dis, spectra = load_eis_data(
            filepath, cell_name
        )
        all_imp[cell_name]  = df_imp
        all_dis[cell_name]  = df_dis
        all_spec[cell_name] = spectra

        print(f" {len(df_imp)} EIS measurements | "
              f"Re: {df_imp['Re'].iloc[0]:.4f} -> "
              f"{df_imp['Re'].iloc[-1]:.4f} Ohm | "
              f"Rct: {df_imp['Rct'].iloc[0]:.4f} -> "
              f"{df_imp['Rct'].iloc[-1]:.4f} Ohm")

    return all_imp, all_dis, all_spec


# ─────────────────────────────────────────────
# STEP 2: GENERATE ALL PLOTS
# ─────────────────────────────────────────────

def plot_nyquist(all_spec, output_path):
    """
    Real Nyquist plots from NASA impedance measurements.

    Plots full complex impedance spectra at early, mid,
    and late cycle life for all 4 cells.

    Nyquist plot: -Im(Z) vs Re(Z)
    - High frequency intercept = Re (ohmic resistance)
    - Semicircle diameter = Rct (charge transfer)
    - Low frequency tail = Warburg diffusion
    As cell ages: semicircle grows, curve shifts right.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    axes = axes.flatten()

    stage_colors = ['#34d399', '#f97316', '#ef4444']
    stage_labels = ['Early', 'Mid', 'Late']

    for ax, cell_name in zip(
            axes, ['B0005', 'B0006', 'B0007', 'B0018']):
        ax.set_facecolor('#0d1117')
        spectra = all_spec[cell_name]
        total = len(spectra)

        # Pick early (5%), mid (50%), late (90%)
        indices = [
            int(total * 0.05),
            int(total * 0.50),
            int(total * 0.90)
        ]
        indices = [min(i, total - 1) for i in indices]

        for idx, color, label in zip(
                indices, stage_colors, stage_labels):
            count, Z_real, Z_imag = spectra[idx]
            # Filter to capacitive arc region
            mask = -Z_imag >= -0.02
            if mask.sum() > 2:
                ax.plot(Z_real[mask], -Z_imag[mask],
                        color=color, linewidth=2.0,
                        alpha=0.85,
                        label=f'{label} '
                              f'(meas. {count})')
                ax.scatter(Z_real[mask][0],
                           -Z_imag[mask][0],
                           color=color, s=50, zorder=5)

        ax.set_xlabel('Re(Z)  [Ohm]',
                      color='#e6edf3', fontsize=10)
        ax.set_ylabel('-Im(Z)  [Ohm]',
                      color='#e6edf3', fontsize=10)
        ax.set_title(f'{cell_name} — Real Nyquist Plot',
                     color='#f59e0b', fontsize=11)
        ax.tick_params(colors='#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=8)

    fig.suptitle(
        'Real Nyquist Plots — NASA PCOE EIS Measurements\n'
        'Semicircle growth = Rct increase (SEI + active '
        'material loss)\n'
        'ICD context: impedance qualification protocol '
        'before implantation',
        color='#e6edf3', fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_re_rct_evolution(all_imp, output_path):
    """
    Real Re and Rct evolution across cycle life.

    Re  = electrolyte resistance — rises as SEI layer grows
    Rct = charge transfer resistance — rises as active
          material degrades

    Both measured directly by NASA EIS equipment.
    This is the key plot showing impedance precedes
    capacity fade as an early warning signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    colors = {
        'B0005': '#f97316',
        'B0006': '#38bdf8',
        'B0007': '#34d399',
        'B0018': '#a855f7'
    }

    for cell_name, df in all_imp.items():
        color = colors[cell_name]

        axes[0].plot(df['impedance_count'], df['Re'],
                     color=color, linewidth=2.0,
                     alpha=0.85, marker='o',
                     markersize=2.5, label=cell_name)

        axes[1].plot(df['impedance_count'], df['Rct'],
                     color=color, linewidth=2.0,
                     alpha=0.85, marker='o',
                     markersize=2.5, label=cell_name)

    titles = [
        'Re (Electrolyte Resistance) vs Measurement Index\n'
        'SEI layer growth — square-root kinetics',
        'Rct (Charge Transfer Resistance) vs Measurement Index\n'
        'Active material loss — exponential growth'
    ]
    ylabels = ['Re (Ohm)', 'Rct (Ohm)']

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_facecolor('#0d1117')
        ax.set_xlabel('EIS Measurement Index',
                      color='#e6edf3', fontsize=11)
        ax.set_ylabel(ylabel, color='#e6edf3', fontsize=11)
        ax.set_title(title, color='#e6edf3', fontsize=10)
        ax.tick_params(colors='#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=10)

    fig.suptitle(
        'Real Impedance Parameter Evolution — NASA PCOE Data\n'
        'Impedance rise precedes capacity fade — '
        'ICD early warning signal',
        color='#e6edf3', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_impedance_vs_capacity(all_imp, all_dis, output_path):
    """
    Correlation between real impedance rise and capacity fade.

    Aligns impedance measurements with nearest discharge cycle
    to show the relationship between Re, Rct and capacity.

    Key finding: impedance rises before capacity drops below
    the EOL threshold — giving advance warning of cell failure.
    Critical for ICD batteries where in-vivo capacity testing
    is not possible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    colors = {
        'B0005': '#f97316',
        'B0006': '#38bdf8',
        'B0007': '#34d399',
        'B0018': '#a855f7'
    }

    for cell_name in all_imp.keys():
        df_imp = all_imp[cell_name]
        df_dis = all_dis[cell_name]
        color  = colors[cell_name]

        # Align: match impedance measurements to
        # nearest discharge cycle by cycle_index
        merged_Re  = []
        merged_Rct = []
        merged_cap = []

        for _, imp_row in df_imp.iterrows():
            # Find nearest discharge cycle
            diffs = (df_dis['cycle_index'] -
                     imp_row['cycle_index']).abs()
            nearest = diffs.idxmin()
            cap = df_dis.loc[nearest, 'capacity_Ah']
            merged_Re.append(imp_row['Re'])
            merged_Rct.append(imp_row['Rct'])
            merged_cap.append(cap)

        axes[0].scatter(merged_Re, merged_cap,
                        color=color, alpha=0.5,
                        s=15, label=cell_name)
        axes[1].scatter(merged_Rct, merged_cap,
                        color=color, alpha=0.5,
                        s=15, label=cell_name)

    # EOL threshold
    for ax in axes:
        ax.axhline(y=1.4, color='#ef4444',
                   linestyle='--', linewidth=1.5,
                   alpha=0.8, label='EOL (1.4 Ah)')

    axes[0].set_xlabel('Re — Electrolyte Resistance (Ohm)',
                       color='#e6edf3', fontsize=11)
    axes[0].set_ylabel('Discharge Capacity (Ah)',
                       color='#e6edf3', fontsize=11)
    axes[0].set_title('Capacity vs Re\nSEI growth correlation',
                      color='#e6edf3', fontsize=11)

    axes[1].set_xlabel(
        'Rct — Charge Transfer Resistance (Ohm)',
        color='#e6edf3', fontsize=11)
    axes[1].set_ylabel('Discharge Capacity (Ah)',
                       color='#e6edf3', fontsize=11)
    axes[1].set_title(
        'Capacity vs Rct\nActive material loss correlation',
        color='#e6edf3', fontsize=11)

    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=9)

    fig.suptitle(
        'Real Impedance vs Capacity Correlation — NASA Data\n'
        'Non-destructive state-of-health estimation '
        'for ICD batteries',
        color='#e6edf3', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bode(all_spec, output_path):
    """
    Bode plot: impedance magnitude vs frequency.
    Uses real NASA impedance spectra.
    Shows how impedance magnitude increases with aging
    across the full frequency range.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    axes = axes.flatten()

    stage_colors = ['#34d399', '#f97316', '#ef4444']
    stage_labels = ['Early', 'Mid', 'Late']

    for ax, cell_name in zip(
            axes, ['B0005', 'B0006', 'B0007', 'B0018']):
        ax.set_facecolor('#0d1117')
        spectra = all_spec[cell_name]
        total = len(spectra)

        indices = [
            int(total * 0.05),
            int(total * 0.50),
            int(total * 0.90)
        ]
        indices = [min(i, total - 1) for i in indices]

        for idx, color, label in zip(
                indices, stage_colors, stage_labels):
            count, Z_real, Z_imag = spectra[idx]
            Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
            # Use index as frequency proxy
            # (NASA data doesn't store freq array separately)
            freq_idx = np.arange(len(Z_mag))

            ax.plot(freq_idx, Z_mag,
                    color=color, linewidth=2.0,
                    alpha=0.85,
                    label=f'{label} (meas. {count})')

        ax.set_xlabel('Frequency Index (low to high)',
                      color='#e6edf3', fontsize=10)
        ax.set_ylabel('|Z| (Ohm)',
                      color='#e6edf3', fontsize=10)
        ax.set_title(
            f'{cell_name} — Impedance Magnitude',
            color='#f59e0b', fontsize=11)
        ax.tick_params(colors='#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=8)

    fig.suptitle(
        'Impedance Magnitude Evolution — Real NASA EIS Data\n'
        'Overall impedance increases with aging across '
        'full frequency range\n'
        'ICD context: rising |Z| reduces available '
        'energy for defibrillation pulse',
        color='#e6edf3', fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("MODULE 2: EIS Analysis — Real NASA PCOE Data")
    print("B0005, B0006, B0007, B0018 | NASA Ames")
    print("ICD Context: Pre-implant qualification protocol")
    print("=" * 60)

    print("\n[1/5] Loading real NASA EIS data...")
    all_imp, all_dis, all_spec = load_all_eis()

    # Save parameter summary
    summary_rows = []
    for cell_name, df in all_imp.items():
        summary_rows.append({
            'cell': cell_name,
            'n_measurements': len(df),
            'Re_initial': df['Re'].iloc[0],
            'Re_final': df['Re'].iloc[-1],
            'Re_increase_pct': (df['Re'].iloc[-1] /
                                df['Re'].iloc[0] - 1) * 100,
            'Rct_initial': df['Rct'].iloc[0],
            'Rct_final': df['Rct'].iloc[-1],
            'Rct_increase_pct': (df['Rct'].iloc[-1] /
                                 df['Rct'].iloc[0] - 1) * 100
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("data/eis_summary.csv", index=False)

    print("\n  EIS Summary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['cell']}: "
              f"Re +{row['Re_increase_pct']:.1f}% | "
              f"Rct +{row['Rct_increase_pct']:.1f}%")

    print("\n[2/5] Plotting real Nyquist plots...")
    plot_nyquist(all_spec, "outputs/module2_nyquist.png")

    print("[3/5] Plotting Re and Rct evolution...")
    plot_re_rct_evolution(all_imp,
                          "outputs/module2_parameter_evolution.png")

    print("[4/5] Plotting impedance vs capacity...")
    plot_impedance_vs_capacity(
        all_imp, all_dis,
        "outputs/module2_capacity_vs_impedance.png")

    print("[5/5] Plotting impedance magnitude...")
    plot_bode(all_spec, "outputs/module2_bode.png")

    print("\n" + "=" * 60)
    print("MODULE 2 COMPLETE — Real NASA EIS Results")
    for _, row in summary_df.iterrows():
        print(f"  {row['cell']}: "
              f"Re {row['Re_initial']:.4f} -> "
              f"{row['Re_final']:.4f} Ohm "
              f"(+{row['Re_increase_pct']:.1f}%) | "
              f"Rct {row['Rct_initial']:.4f} -> "
              f"{row['Rct_final']:.4f} Ohm "
              f"(+{row['Rct_increase_pct']:.1f}%)")
    print("=" * 60)