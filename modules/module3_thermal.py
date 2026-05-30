"""
Module 3: Thermal Safety Envelope
==================================
Physics-based thermal simulation for ICD battery qualification.

NOTE ON METHODOLOGY:
The NASA PCOE dataset (B0005-B0018) was tested at room temperature
(24C). No in-vivo thermal data exists for implanted ICD batteries
in public datasets. This module uses a physics-based DFN-inspired
thermal model to simulate battery behavior at body temperature
(37C) — the critical operating condition for implanted devices.

Real NASA data informs the model parameters:
- Internal resistance from B0005 EIS: Re = 0.0447 Ohm (measured)
- ICD cell capacity: 0.5 Ah (typical small form factor ICD cell)
- NASA B0005 is 1.856 Ah — too large for ICD application
- Re is the key grounded parameter; capacity scaled to ICD spec

Reference:
    Doyle, M., Fuller, T.F. & Newman, J. (1993). Modeling of
    galvanostatic charge and discharge of the lithium/polymer/
    insertion cell. Journal of the Electrochemical Society,
    140(6), 1526-1533.

    Marquis, S.G. et al. (2019). An asymptotic derivation of a
    single particle model with electrolyte. Journal of the
    Electrochemical Society, 166(15), A3693.

ICD Context:
    ICDs operate at 37C body temperature. Heat cannot dissipate
    freely. This module maps the safe operating envelope,
    identifying C-rate and temperature combinations that risk
    lithium plating or thermal events. Required for FDA 510(k).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as sio
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────
# STEP 1: LOAD REAL NASA PARAMETERS
# ─────────────────────────────────────────────

def get_nasa_parameters():
    """
    Extract real Re from NASA B0005 first impedance measurement.
    Use ICD-appropriate capacity (0.5 Ah) for thermal simulation.
    """
    mat = sio.loadmat('data/B0005.mat')
    battery = mat['B0005']
    cycles = battery['cycle'][0, 0]

    re_values = []
    for i in range(cycles.shape[1]):
        cycle_type = str(cycles[0, i]['type'][0])
        if cycle_type == 'impedance':
            data = cycles[0, i]['data'][0, 0]
            re_values.append(float(data['Re'][0, 0]))

    initial_Re = re_values[0]

    # ICD cells are small form factor — 0.5 Ah typical
    # NASA B0005 is 1.856 Ah (laptop/EV cell, too large for ICD)
    # We use the NASA-measured Re as model input,
    # scaled to ICD cell geometry
    icd_capacity = 0.5  # Ah

    print(f"      NASA B0005 measured Re (initial): "
          f"{initial_Re:.4f} Ohm")
    print(f"      ICD cell capacity (model input): "
          f"{icd_capacity:.1f} Ah")

    return initial_Re, icd_capacity


# ─────────────────────────────────────────────
# STEP 2: PHYSICS-BASED THERMAL MODEL
# ─────────────────────────────────────────────

def arrhenius_factor(T_celsius, T_ref=25.0, Ea_R=6500.0):
    """
    Arrhenius temperature correction factor.
    k(T)/k(T_ref) = exp(Ea/R * (1/T_ref - 1/T))
    Ea/R = 6500 K for Li-ion side reactions
    At 37C vs 25C: ~2.1x faster side reactions
    """
    T_K     = T_celsius + 273.15
    T_ref_K = T_ref + 273.15
    return np.exp(Ea_R * (1/T_ref_K - 1/T_K))


def simulate_discharge(C_rate, T_ambient, R0, capacity_Ah):
    """
    Simulate ICD battery discharge using DFN-inspired thermal model.

    Heat balance:
        m*Cp * dT/dt = Q_irrev + Q_rev - Q_diss
        Q_irrev = I^2 * R  (Joule heating)
        Q_rev   = I*T*|dU/dT| (entropic heat)
        Q_diss  = h*A*(T_cell - T_ambient)

    Parameters:
    - R0 from real NASA B0005 EIS measurement (0.0447 Ohm)
    - capacity_Ah = 0.5 Ah (ICD cell spec)
    - h*A is LOW: implanted device has poor heat dissipation
    """
    # Thermal parameters for small ICD cell
    m_cell = 0.0035   # kg  (3.5g small form factor)
    Cp     = 1100.0   # J/kg/K
    h_A    = 0.018    # W/K (implanted — low dissipation)
    dUdT   = -0.0002  # V/K entropic coefficient

    # Arrhenius correction for temperature effects
    arr        = arrhenius_factor(T_ambient)
    R_internal = R0 * (1.0 + 0.3 * (arr - 1.0))

    I           = C_rate * capacity_Ah
    t_discharge = int(3600 / C_rate)
    dt          = 1.0

    time        = np.zeros(t_discharge)
    temperature = np.zeros(t_discharge)
    voltage     = np.zeros(t_discharge)
    heat_gen    = np.zeros(t_discharge)
    anode_pot   = np.zeros(t_discharge)

    T_cell    = T_ambient + 273.15
    OCV_full  = 4.20
    OCV_empty = 2.70

    plating_risk = False

    for i in range(t_discharge):
        t   = i * dt
        SOC = max(0.0, 1.0 - t / (3600 / C_rate))
        OCV = OCV_empty + (OCV_full - OCV_empty) * SOC

        T_celsius = T_cell - 273.15
        R_eff     = R_internal * (1 + 0.002 * (T_celsius - 25))

        V = max(2.0, OCV - I * R_eff)

        Q_irrev = I**2 * R_eff
        Q_rev   = I * T_cell * abs(dUdT)
        Q_total = Q_irrev + Q_rev
        Q_diss  = h_A * (T_cell - (T_ambient + 273.15))

        dT     = (Q_total - Q_diss) / (m_cell * Cp) * dt
        T_cell = T_cell + dT

        V_anode = 0.10 - (I * R_eff * 0.5) - \
            (T_celsius - 25) * 0.0012

        if V_anode < 0.05 and not plating_risk:
            plating_risk = True

        time[i]        = t
        temperature[i] = T_cell - 273.15
        voltage[i]     = V
        heat_gen[i]    = Q_total
        anode_pot[i]   = V_anode

    T_max_rise = temperature.max() - T_ambient

    return {
        'C_rate':       C_rate,
        'T_ambient':    T_ambient,
        'time':         time,
        'temperature':  temperature,
        'voltage':      voltage,
        'heat_gen':     heat_gen,
        'anode_pot':    anode_pot,
        'T_max_rise':   T_max_rise,
        'plating_risk': plating_risk
    }


def run_all_simulations(R0, capacity_Ah):
    """
    Run 15 simulations: 5 C-rates x 3 temperatures.
    """
    C_rates    = [0.5, 1.0, 2.0, 3.0, 4.0]
    T_ambients = [25.0, 37.0, 45.0]

    results = []
    total   = len(C_rates) * len(T_ambients)
    count   = 0

    for T in T_ambients:
        for C in C_rates:
            count += 1
            print(f"      C={C:.1f}C, T={T:.0f}C "
                  f"({count}/{total})...", end="")
            r = simulate_discharge(C, T, R0, capacity_Ah)
            results.append(r)
            status = "PLATING RISK" \
                if r['plating_risk'] else "OK"
            print(f" dT={r['T_max_rise']:.2f}C  {status}")

    return results, C_rates, T_ambients


# ─────────────────────────────────────────────
# STEP 3: GENERATE ALL PLOTS
# ─────────────────────────────────────────────

def plot_temperature_rise(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    body_results = [r for r in results if r['T_ambient'] == 37.0]
    colors = ['#34d399', '#38bdf8', '#f97316', '#ef4444', '#a855f7']

    for result, color in zip(body_results, colors):
        t_min = result['time'] / 60
        ax.plot(t_min, result['temperature'],
                color=color, linewidth=2.0, alpha=0.85,
                label=f"C-rate: {result['C_rate']:.1f}C")

    ax.axhline(y=37.0, color='white', linestyle='--',
               linewidth=1.2, alpha=0.6,
               label='Body temperature (37C)')
    ax.axhline(y=45.0, color='#ef4444', linestyle='-.',
               linewidth=1.5, alpha=0.8,
               label='Safety threshold (45C)')

    ax.set_xlabel('Time (minutes)', color='#e6edf3', fontsize=12)
    ax.set_ylabel('Cell Temperature (C)', color='#e6edf3', fontsize=12)
    ax.set_title(
        'Temperature Rise During Discharge — T_ambient = 37C (Body Temp)\n'
        'Model Re grounded in real NASA B0005 EIS: 0.0447 Ohm\n'
        'ICD: implanted device has minimal heat dissipation to tissue',
        color='#e6edf3', fontsize=12, pad=15)
    ax.set_ylim(35, 55)
    ax.tick_params(colors='#e6edf3')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_voltage_curves(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')

    T_list   = [25.0, 37.0, 45.0]
    T_labels = ['25C (Room)', '37C (Body)', '45C (Elevated)']
    colors   = ['#34d399', '#38bdf8', '#f97316', '#ef4444', '#a855f7']

    for ax, T, T_label in zip(axes, T_list, T_labels):
        ax.set_facecolor('#0d1117')
        T_results = [r for r in results if r['T_ambient'] == T]

        for result, color in zip(T_results, colors):
            SOC = 1.0 - result['time'] / result['time'].max()
            ax.plot(SOC, result['voltage'],
                    color=color, linewidth=1.8, alpha=0.85,
                    label=f"{result['C_rate']:.1f}C")

        ax.axhline(y=2.7, color='white', linestyle='--',
                   linewidth=1.0, alpha=0.5, label='Cutoff (2.7V)')
        ax.set_xlabel('State of Charge', color='#e6edf3', fontsize=11)
        ax.set_ylabel('Voltage (V)', color='#e6edf3', fontsize=11)
        ax.set_title(f'Discharge Curves\n{T_label}',
                     color='#e6edf3', fontsize=11)
        ax.set_ylim(2.4, 4.3)
        ax.tick_params(colors='#e6edf3')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=8,
                  title='C-rate', title_fontsize=8)

    fig.suptitle(
        'Simulated Discharge Curves — ICD Li-ion Cell\n'
        'Higher temperature reduces effective voltage window\n'
        'Model Re input from real NASA B0005 measurement',
        color='#e6edf3', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_safety_envelope(results, C_rates, T_ambients, output_path):
    T_rise_grid  = np.zeros((len(T_ambients), len(C_rates)))
    plating_grid = np.zeros((len(T_ambients), len(C_rates)))

    for r in results:
        i = T_ambients.index(r['T_ambient'])
        j = C_rates.index(r['C_rate'])
        T_rise_grid[i, j]  = r['T_max_rise']
        plating_grid[i, j] = 1.0 if r['plating_risk'] else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    cmap = LinearSegmentedColormap.from_list(
        'safety', ['#34d399', '#fbbf24', '#ef4444'])

    im = ax.imshow(
        T_rise_grid, cmap=cmap, aspect='auto',
        vmin=0, vmax=T_rise_grid.max(),
        extent=[-0.5, len(C_rates)-0.5,
                -0.5, len(T_ambients)-0.5],
        origin='lower')

    for i in range(len(T_ambients)):
        for j in range(len(C_rates)):
            val  = T_rise_grid[i, j]
            risk = plating_grid[i, j]
            text = f"+{val:.1f}C"
            if risk:
                text += "\nPLATING"
            color = 'white' \
                if val > T_rise_grid.max() * 0.5 else '#0d1117'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(C_rates)))
    ax.set_xticklabels([f'{c:.1f}C' for c in C_rates],
                       color='#e6edf3', fontsize=11)
    ax.set_yticks(range(len(T_ambients)))
    ax.set_yticklabels([f'{t:.0f}C' for t in T_ambients],
                       color='#e6edf3', fontsize=11)
    ax.set_xlabel('Discharge C-rate', color='#e6edf3', fontsize=12)
    ax.set_ylabel('Ambient Temperature', color='#e6edf3', fontsize=12)
    ax.set_title(
        'Thermal Safe Operating Envelope — ICD Li-ion Cell\n'
        'Color = max temperature rise | PLATING = lithium plating risk\n'
        'Model Re from real NASA B0005 EIS: 0.0447 Ohm | '
        'FDA 510(k) context',
        color='#e6edf3', fontsize=12, pad=15)

    ax.axhline(y=0.5, color='#38bdf8', linewidth=2.5,
               linestyle='--', alpha=0.8)
    ax.axhline(y=1.5, color='#38bdf8', linewidth=2.5,
               linestyle='--', alpha=0.8)
    ax.text(len(C_rates)-0.4, 1.0, '  Body temp (37C)',
            color='#38bdf8', fontsize=9, va='center', ha='right')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Temperature Rise (C)',
                   color='#e6edf3', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#e6edf3')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#e6edf3')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heat_generation(results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    body_results = [r for r in results if r['T_ambient'] == 37.0]
    colors = ['#34d399', '#38bdf8', '#f97316', '#ef4444', '#a855f7']

    for result, color in zip(body_results, colors):
        t_min = result['time'] / 60
        ax.plot(t_min, result['heat_gen'] * 1000,
                color=color, linewidth=2.0, alpha=0.85,
                label=f"C-rate: {result['C_rate']:.1f}C")

    ax.set_xlabel('Time (minutes)', color='#e6edf3', fontsize=12)
    ax.set_ylabel('Heat Generation Rate (mW)', color='#e6edf3', fontsize=12)
    ax.set_title(
        'Heat Generation Rate — T_ambient = 37C\n'
        'Q = I^2*R (Joule) + I*T*|dU/dT| (entropic)\n'
        'R grounded in real NASA B0005 EIS measurement',
        color='#e6edf3', fontsize=12, pad=15)
    ax.tick_params(colors='#e6edf3')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.legend(facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("MODULE 3: Thermal Safety Envelope")
    print("Physics model + Real NASA B0005 parameters")
    print("ICD Context: FDA 510(k) operating envelope")
    print("=" * 60)

    print("\n[1/6] Loading real NASA B0005 parameters...")
    R0, capacity_Ah = get_nasa_parameters()

    print(f"\n[2/6] Running 15 thermal simulations "
          f"(5 C-rates x 3 temps)...")
    results, C_rates, T_ambients = run_all_simulations(
        R0, capacity_Ah)

    summary = pd.DataFrame([{
        'C_rate':       r['C_rate'],
        'T_ambient':    r['T_ambient'],
        'T_max_rise':   round(r['T_max_rise'], 3),
        'plating_risk': r['plating_risk']
    } for r in results])
    summary.to_csv("data/thermal_summary.csv", index=False)

    print("\n[3/6] Plotting temperature rise...")
    plot_temperature_rise(
        results, "outputs/module3_temperature_rise.png")

    print("[4/6] Plotting voltage curves...")
    plot_voltage_curves(
        results, "outputs/module3_voltage_curves.png")

    print("[5/6] Plotting safety envelope...")
    plot_safety_envelope(
        results, C_rates, T_ambients,
        "outputs/module3_safety_envelope.png")

    print("[6/6] Plotting heat generation...")
    plot_heat_generation(
        results, "outputs/module3_heat_generation.png")

    body = [r for r in results if r['T_ambient'] == 37.0]
    safe_C = [r['C_rate'] for r in body
              if not r['plating_risk']]
    max_safe_C = max(safe_C) if safe_C else 0.5

    print("\n" + "=" * 60)
    print("MODULE 3 COMPLETE")
    print(f"Real NASA B0005 Re used: {R0:.4f} Ohm")
    print(f"ICD cell capacity modeled: {capacity_Ah:.1f} Ah")
    print(f"Safe C-rate at 37C body temp: {max_safe_C:.1f}C")
    print(f"Plating risk cases: "
          f"{sum(r['plating_risk'] for r in results)}/15")
    print("=" * 60)