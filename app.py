"""
Li-ion Battery Degradation and State Estimation Toolkit
=======================================================
Streamlit dashboard — dark navy/amber theme
Real NASA PCOE Battery Dataset — B0005, B0006, B0007, B0018
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Battery Degradation Toolkit",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0f1e; color: #e6edf3; }
    [data-testid="stSidebar"] { background-color: #0d1427; border-right: 1px solid #1e2a45; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="metric-container"] { background-color: #0d1427; border: 1px solid #1e2a45; border-radius: 8px; padding: 16px; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f59e0b !important; font-size: 28px !important; font-weight: 700 !important; }
    h1, h2, h3 { color: #f59e0b !important; }
    h4, h5, h6 { color: #e6edf3 !important; }
    .stAlert { background-color: #0d1427; border: 1px solid #1e2a45; border-radius: 8px; }
    hr { border-color: #1e2a45; }
    .stRadio label { color: #e6edf3 !important; }
    code { background-color: #0d1427 !important; color: #f59e0b !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #f59e0b !important; border-bottom-color: #f59e0b !important; }
    .result-box { background-color: #0d1427; border: 1px solid #1e2a45; border-radius: 8px; padding: 16px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔋 Battery Toolkit")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📈  Module 1: RUL Prediction",
            "⚡  Module 2: EIS Analysis",
            "🌡️  Module 3: Thermal Safety",
            "👤  About"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Context**")
    st.markdown(
        "Tools developed during MS thesis on "
        "Li-ion cells for implantable cardiac "
        "defibrillators (ICD) at Purdue University."
    )
    st.markdown("---")
st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Narain115-f59e0b?"
        "style=flat&logo=github)](https://github.com/Narain115/battery-toolkit)"
    )
    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown(
        "[NASA PCOE Battery Dataset](https://www.nasa.gov/intelligent-systems-division/"
        "discovery-and-systems-health/pcoe/pcoe-data-set-repository/)"
    )
    st.markdown(
        "[Dataset README](https://github.com/Narain115/battery-toolkit/blob/master/"
        "README_NASA_B0005_B0006_B0007_B0018.txt)"
    )


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "🏠  Overview":

    st.title("Li-ion Battery Degradation & State Estimation Toolkit")
    st.markdown("**MS Thesis Project — Materials Science & Engineering, Purdue University**")

    st.markdown("""
    During my MS thesis designing Li-ion cells for **implantable cardiac
    defibrillators (ICD)**, I needed analytical tools to predict battery
    end-of-life, characterize degradation mechanisms, and verify safe
    operating envelopes. I built this toolkit to solve those problems.

    An ICD is surgically implanted in the chest. The battery cannot be
    easily replaced. It must last **7-10 years**, operate at **37°C body
    temperature**, and never fail. There is no low-battery warning.
    The analytical techniques demonstrated here — capacity fade tracking,
    impedance monitoring, and thermal modeling — are the exact workflows
    used to qualify batteries for safety-critical implantable devices.
    """)

    st.markdown("---")
    st.markdown("### Key Results — All Numbers from Real NASA Measurements")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fastest degrading cell", "B0018",
                  delta="EOL at cycle 97")
    with col2:
        st.metric("Longest lasting cell", "B0007",
                  delta="EOL at cycle 168")
    with col3:
        st.metric("Safe C-rate at 37°C", "1.0C",
                  delta="Body temperature limit")
    with col4:
        st.metric("Real EIS measurements", "1,112",
                  delta="278 per cell × 4 cells")

    st.markdown("---")
    st.markdown("### Dataset — NASA PCOE Battery Dataset")
    st.markdown("""
    All data in this project comes from **real laboratory measurements**
    at NASA Ames Research Center, Prognostics Center of Excellence.
    Four 18650 Li-ion cells were cycled to end-of-life under controlled conditions.

    | Cell | Initial Capacity | EOL Cycle | Total Fade | EIS Measurements |
    |------|-----------------|-----------|------------|-----------------|
    | B0005 | 1.842 Ah | Cycle 125 | 28.0% | 278 |
    | B0006 | 2.018 Ah | Cycle 109 | 41.2% | 278 |
    | B0007 | 1.883 Ah | Cycle 168 | ~24% | 278 |
    | B0018 | 1.840 Ah | Cycle 97 | ~27% | 278 |

    **Protocol:** 1.5A CC-CV charge to 4.2V | 2A CC discharge | 24°C room temperature
    """)

    st.markdown("---")
    st.markdown("### Three Modules, Three Engineering Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Module 1: RUL Prediction")
        st.markdown("""
        **Question:** Can we detect which battery will fail
        sooner, using only the first 30 cycles?

        **Finding:** B0018 shows a capacity fade slope of
        -0.0035 Ah/cycle in early cycles — 4x steeper than
        B0005. This signal is detectable before implantation.

        **Dataset:** 636 real discharge cycles across 4 cells
        """)

    with col2:
        st.markdown("#### ⚡ Module 2: EIS Analysis")
        st.markdown("""
        **Question:** Can impedance measurements detect
        degradation before capacity visibly drops?

        **Finding:** B0006 Re rose +20.2% and Rct rose +27.3%
        across its life — both rising well before capacity
        crossed the 1.4 Ah EOL threshold at cycle 109.

        **Dataset:** 1,112 real EIS spectra across 4 cells
        """)

    with col3:
        st.markdown("#### 🌡️ Module 3: Thermal Safety")
        st.markdown("""
        **Question:** How does body temperature (37°C) change
        the safe operating envelope?

        **Finding:** Safe C-rate drops from 2.0C at 25°C to
        1.0C at 37°C. Arrhenius kinetics (Ea/R = 6500K) mean
        side reactions are ~2x faster at body temperature.

        **Model input:** Real NASA B0005 Re = 0.0447 Ohm
        """)

    st.markdown("---")
    st.markdown("### References")
    st.markdown("""
    1. Saha, B. & Goebel, K. (2007). Battery Data Set. NASA PCOE Data Repository.
       [Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
    2. Doyle, M., Fuller, T.F. & Newman, J. (1993). J. Electrochem. Soc., 140(6), 1526-1533.
       [Paper](https://doi.org/10.1149/1.2221597)
    3. Severson, K.A. et al. (2019). Nature Energy, 4, 383-391.
       [Paper](https://doi.org/10.1038/s41560-019-0356-8)
    4. Birkl, C.R. et al. (2017). Journal of Power Sources, 341, 373-386.
       [Paper](https://doi.org/10.1016/j.jpowsour.2016.09.073)
    """)


# ─────────────────────────────────────────────
# PAGE: MODULE 1
# ─────────────────────────────────────────────

elif page == "📈  Module 1: RUL Prediction":

    st.title("Module 1: Remaining Useful Life Prediction")
    st.markdown(
        "**Dataset:** NASA PCOE — Real measurements from NASA Ames Research Center | "
        "B0005, B0006, B0007, B0018 | 636 real discharge cycles"
    )
    st.info(
        "**ICD Context:** Before a battery is packaged into an implantable device, "
        "it undergoes formation cycling. This module demonstrates that early-cycle "
        "measurements can identify which cells degrade faster — enabling a "
        "qualification go/no-go decision before implantation."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("B0005 EOL", "Cycle 125",
                  delta="Initial: 1.842 Ah")
    with col2:
        st.metric("B0006 EOL", "Cycle 109",
                  delta="Initial: 2.018 Ah")
    with col3:
        st.metric("B0007 EOL", "Cycle 168",
                  delta="Initial: 1.883 Ah")
    with col4:
        st.metric("B0018 EOL", "Cycle 97",
                  delta="Fastest degrading")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Capacity Fade", "Voltage Evolution",
        "Formation Window", "Feature Correlation"
    ])

    with tab1:
        st.markdown("#### Real Capacity Fade — All 4 NASA Cells")
        if os.path.exists("outputs/module1_capacity_fade.png"):
            st.image("outputs/module1_capacity_fade.png",
                     use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        Every point on this plot is a **real discharge measurement** from
        NASA Ames Research Center. Each cell was discharged at 2A until
        the voltage dropped to its cutoff, and the total charge delivered
        was recorded as the capacity for that cycle.

        **Key findings from the real data:**

        - **B0007 lasted longest** — 168 cycles before hitting the 1.4 Ah
          EOL threshold, despite starting with a lower initial capacity
          (1.883 Ah) than B0006 (2.018 Ah). This shows that initial
          capacity alone does not predict cycle life.

        - **B0018 died earliest** — reached EOL at cycle 97, 73% fewer
          cycles than B0007. Its capacity dropped below 1.6 Ah by cycle 50,
          while B0007 was still above 1.7 Ah at the same point.

        - **B0006 showed the steepest total fade** — from 2.018 Ah to
          1.186 Ah, a 41.2% total capacity loss by the end of the dataset.
          This is the most degraded cell in the set.

        - **B0005 degraded most uniformly** — relatively steady linear
          fade from 1.856 Ah to 1.325 Ah (28.0% total fade), reaching
          EOL at cycle 125.

        **Engineering implication:** The 73-cycle gap between B0018 (worst)
        and B0007 (best) from the same cell type means cell-to-cell
        variability is significant. For ICD manufacturing, every cell must
        be individually qualified — you cannot assume all cells from the
        same batch behave identically.
        """)

    with tab2:
        st.markdown("#### Discharge Voltage Curve Evolution")
        if os.path.exists("outputs/module1_predicted_vs_actual.png"):
            st.image("outputs/module1_predicted_vs_actual.png",
                     use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        These plots show how the **shape of the discharge voltage curve
        changes** as the cell ages, comparing early (cycle 5), mid (50%
        of life), and late (90% of life) cycles for each cell.

        **Key findings:**

        - **Voltage depression with aging** — the mid and late curves sit
          consistently lower than the early curve across the entire
          discharge. This happens because internal resistance increases
          with aging (confirmed by the EIS data in Module 2), causing
          a larger voltage drop under the same 2A discharge current.

        - **Shorter discharge time** — late-life curves end earlier
          (lower capacity), visible as the curves terminating further
          left. This directly maps to the capacity fade seen in the
          previous plot.

        - **The shape change is detectable early** — by mid-life, the
          voltage depression is already visible. An engineer monitoring
          voltage curve shape in real time could detect degradation
          before capacity crosses the EOL threshold.

        **ICD implication:** ICD batteries must deliver a precise
        high-voltage pulse for defibrillation. A depressed voltage curve
        means less available energy for the pulse. Voltage curve shape
        monitoring is a real diagnostic used in ICD battery management
        systems to track state-of-health without running a full discharge.
        """)

    with tab3:
        st.markdown("#### Formation Window — First 30 Cycles")
        if os.path.exists("outputs/module1_shap.png"):
            st.image("outputs/module1_shap.png",
                     use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This zooms into the **first 30 cycles only** — the formation
        cycling window that happens in the factory before a battery
        is packaged into a device.

        **Key findings from the real early-cycle data:**

        - **B0005 capacity fade slope: -0.000904 Ah/cycle** — the
          gentlest degradation rate in the first 30 cycles. It went
          on to last until cycle 125.

        - **B0006 capacity fade slope: -0.005481 Ah/cycle** — 6x
          steeper than B0005 in the same early window, with much
          higher variance (0.00327 vs 0.000237). It died at cycle 109.

        - **The temperature plot (right)** shows real cell temperature
          during discharge. All cells run slightly above room temperature
          (24°C ambient) due to Joule heating. Higher temperature
          readings correlate with higher internal resistance —
          consistent with the EIS data.

        **Engineering implication:** The difference in early-cycle
        fade slope between cells is measurable with just 30 cycles of
        data. In an ICD manufacturing line, this means a qualification
        engineer can flag underperforming cells before they are
        implanted — without waiting for the cells to reach end-of-life.
        """)

    with tab4:
        st.markdown("#### Feature Correlation with Cycle Life")
        if os.path.exists("outputs/module1_deltaQ.png"):
            st.image("outputs/module1_deltaQ.png",
                     use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This shows the **Pearson correlation coefficient** between each
        early-cycle feature and the total cycle life of the cell,
        calculated from real NASA data across all 4 cells.

        **Key findings:**

        - **Capacity fade slope: correlation = +0.815** — the strongest
          predictor. A steeper capacity fade in the first 30 cycles
          strongly predicts shorter total life. Higher slope value
          (less negative) = gentler fade = longer life.

        - **Capacity variance: correlation = -0.768** — second strongest.
          More variability in early-cycle capacity measurements indicates
          instability in the cell chemistry, predicting shorter life.

        - **Mean temperature: correlation = +0.735** — cells that run
          cooler in early cycles (lower internal resistance) tend to
          last longer.

        - **Initial capacity: correlation = -0.124** — weakest predictor.
          This confirms that how much charge a cell can hold initially
          tells you almost nothing about how long it will last.
          B0006 had the highest initial capacity (2.018 Ah) but died
          at cycle 109, while B0007 had a lower initial capacity
          (1.883 Ah) but lasted to cycle 168.

        **Engineering implication:** Capacity fade slope and variance
        in the first 30 cycles are the most informative screening
        metrics for ICD battery qualification. Initial capacity
        specifications alone are insufficient for predicting cell life.
        """)


# ─────────────────────────────────────────────
# PAGE: MODULE 2
# ─────────────────────────────────────────────

elif page == "⚡  Module 2: EIS Analysis":

    st.title("Module 2: Electrochemical Impedance Spectroscopy")
    st.markdown(
        "**Dataset:** NASA PCOE — Real EIS measurements | "
        "278 spectra per cell × 4 cells = 1,112 total | "
        "Frequency sweep 0.1 Hz to 5 kHz"
    )
    st.info(
        "**ICD Context:** EIS is a non-destructive measurement — "
        "you do not need to fully discharge the battery to assess its health. "
        "This is critical for implanted devices where full discharge testing "
        "is not possible in vivo. Rising Re and Rct are early warning signals "
        "that appear before capacity crosses the EOL threshold."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("B0005 Re increase", "+12.0%",
                  delta="0.0447 → 0.0500 Ohm")
    with col2:
        st.metric("B0006 Re increase", "+20.2%",
                  delta="0.0612 → 0.0736 Ohm")
    with col3:
        st.metric("B0005 Rct increase", "+7.7%",
                  delta="0.0695 → 0.0748 Ohm")
    with col4:
        st.metric("B0006 Rct increase", "+27.3%",
                  delta="0.0785 → 0.1000 Ohm")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Nyquist Plots", "Re & Rct Evolution",
        "Capacity Correlation", "Impedance Magnitude"
    ])

    with tab1:
        st.markdown("#### Real Nyquist Plots — NASA EIS Measurements")
        if os.path.exists("outputs/module2_nyquist.png"):
            st.image("outputs/module2_nyquist.png",
                     use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        A Nyquist plot shows the **real part of impedance (Re(Z)) on the
        x-axis** and the **negative imaginary part (-Im(Z)) on the y-axis**
        at each frequency in the sweep (0.1 Hz to 5 kHz).

        **How to read it:**
        - The point where the curve intersects the x-axis at high frequency
          = Re (ohmic/electrolyte resistance, dominated by SEI layer)
        - The diameter of the semicircular arc = Rct (charge transfer
          resistance at the electrode-electrolyte interface)
        - The 45° line at low frequency = Warburg diffusion element
          (solid-state lithium-ion diffusion in the electrode)

        **Key findings from the real NASA data:**

        - **B0018 shows the clearest aging signature** — the late-life
          spectrum (red) shows a distinctly larger semicircle and a
          rightward shift compared to early life (green), confirming
          both Rct growth and Re growth over its 97-cycle life.

        - **B0005 and B0006 show consistent semicircle growth** — mid
          and late spectra cluster together with larger arcs than early
          measurements, consistent with the Re and Rct increase confirmed
          in the parameter evolution plot.

        - **Some early measurements (green curves) show noise spikes** —
          this is real measurement artifact from the NASA EIS equipment
          during the first few cycles when the cell is still stabilizing
          from formation. This is documented in the NASA dataset README.

        **ICD implication:** A single EIS measurement takes minutes and
        does not require discharging the battery. In an ICD monitoring
        system, periodic impedance checks can track Re and Rct trends
        and flag the device for replacement before the battery fails.
        """)

    with tab2:
        st.markdown("#### Re and Rct Evolution — Real Measurements")
        if os.path.exists("outputs/module2_parameter_evolution.png"):
            st.image("outputs/module2_parameter_evolution.png",
                     use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        These plots track the **two key Randles circuit parameters**
        across all 278 EIS measurements for each cell. Every point
        is a real NASA laboratory measurement.

        **Re (left plot) — electrolyte/ohmic resistance:**
        - Re rises because the **SEI (solid electrolyte interphase)
          layer grows thicker** on the graphite anode with each cycle.
          SEI growth follows approximate square-root kinetics
          (diffusion-limited process).
        - **B0006 has the highest absolute Re** (starts at 0.0612 Ohm
          vs B0005's 0.0447 Ohm) and the largest increase (+20.2%).
          This explains why B0006 degrades more aggressively.
        - **B0005 has the lowest Re** throughout its life (0.0447 →
          0.0500 Ohm, +12.0%), consistent with it having a more
          stable SEI and living to cycle 125.

        **Rct (right plot) — charge transfer resistance:**
        - Rct rises because **active material is lost** from the
          electrodes — lithium plating, particle cracking, and
          loss of electrical contact all reduce the electrochemically
          active surface area.
        - **B0006 shows the largest Rct increase** (+27.3%, from
          0.0785 to 0.1000 Ohm). A Rct doubling would indicate
          severe active material loss.
        - **B0007 and B0018 show noisier Rct trends** — the early
          measurements have higher variance, which is consistent
          with the NASA README noting measurement variability
          during initial cycles.

        **Critical finding:** Both Re and Rct begin rising from the
        very first measurements, well before capacity crosses the
        1.4 Ah EOL threshold. This confirms that impedance is a
        leading indicator of degradation, not a lagging one.
        """)

    with tab3:
        st.markdown("#### Real Impedance vs Capacity Correlation")
        if os.path.exists("outputs/module2_capacity_vs_impedance.png"):
            st.image("outputs/module2_capacity_vs_impedance.png",
                     use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This plot aligns each impedance measurement with its nearest
        discharge cycle to show the **direct relationship between
        rising impedance and falling capacity** — both from real
        NASA measurements.

        **Key findings:**

        - **Clear negative correlation in both plots** — as Re and
          Rct increase (x-axis moves right), capacity decreases
          (y-axis moves down). This relationship holds consistently
          across all 4 cells.

        - **B0006 (blue) spans the widest Re range** — from 0.04 Ohm
          (healthy) to 0.08 Ohm (degraded), the widest spread of
          any cell. Its capacity dropped from 2.018 Ah to 1.186 Ah
          over this range — a 41.2% total fade.

        - **The EOL threshold (red dashed line at 1.4 Ah)** is crossed
          at different impedance levels for different cells, confirming
          that there is no single universal impedance threshold for EOL.
          Each cell has its own degradation trajectory.

        - **Practical implication for ICD monitoring:** You cannot
          define a single Re or Rct cutoff that applies to all cells.
          Instead, you track the rate of change of impedance for each
          individual device and flag it when the slope of Re or Rct
          increase accelerates beyond a threshold.
        """)

    with tab4:
        st.markdown("#### Impedance Magnitude Evolution")
        if os.path.exists("outputs/module2_bode.png"):
            st.image("outputs/module2_bode.png",
                     use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This shows **|Z| (impedance magnitude)** plotted against
        frequency index across early, mid, and late cycle life
        for each cell.

        **How to read it:**
        - At low frequency indices (left side), the impedance
          magnitude is dominated by diffusion (Warburg element).
        - At high frequency indices (right side), the magnitude
          rises again as the capacitive behavior breaks down.
        - The middle flat region represents the resistive plateau
          where Re + Rct dominate.

        **Key findings:**

        - **The resistive plateau rises with aging** in B0005, B0006,
          and B0018 — mid and late curves sit higher than early curves
          in the flat region. This is consistent with the Re and Rct
          increases confirmed in the parameter evolution plot.

        - **B0007 shows more noise in early measurements** — the
          green curve has larger amplitude variations. This is
          real measurement noise from the NASA equipment, not
          an artifact of our analysis.

        - **Rising |Z| means the battery delivers less power**
          at the same voltage. For an ICD, this directly affects
          the energy available for the defibrillation pulse —
          a clinically critical performance parameter.
        """)


# ─────────────────────────────────────────────
# PAGE: MODULE 3
# ─────────────────────────────────────────────

elif page == "🌡️  Module 3: Thermal Safety":

    st.title("Module 3: Thermal Safe Operating Envelope")
    st.markdown(
        "**Model:** DFN-inspired thermal simulation | "
        "Re input from real NASA B0005 EIS: 0.0447 Ohm | "
        "ICD cell 0.5 Ah | 15 simulations (5 C-rates × 3 temps)"
    )
    st.info(
        "**Methodology note:** The NASA dataset was tested at 24°C room temperature. "
        "No public dataset exists for batteries tested at 37°C body temperature inside "
        "an implanted device. This module uses a physics-based thermal model with the "
        "NASA-measured resistance as input to simulate ICD operating conditions. "
        "This is standard practice in battery engineering for safety-critical applications."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Safe C-rate at 37°C", "1.0C",
                  delta="Body temperature limit")
    with col2:
        st.metric("Safe C-rate at 25°C", "2.0C",
                  delta="Room temperature")
    with col3:
        st.metric("Plating risk cases", "7 / 15",
                  delta="Simulations flagged")
    with col4:
        st.metric("NASA Re input", "0.0447 Ω",
                  delta="Real B0005 measurement")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Safety Envelope", "Temperature Rise",
        "Voltage Curves", "Heat Generation"
    ])

    with tab1:
        st.markdown("#### Thermal Safe Operating Envelope")
        if os.path.exists("outputs/module3_safety_envelope.png"):
            st.image("outputs/module3_safety_envelope.png",
                     use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This is the **primary deliverable of Module 3** — a map of maximum
        temperature rise and lithium plating risk across all combinations
        of C-rate and ambient temperature simulated.

        **How to read it:**
        - Each cell in the grid = one simulation run
        - Color = maximum temperature rise above ambient (green = safe, red = unsafe)
        - "PLATING" = the anode potential dropped below 0.05V vs Li/Li+,
          indicating lithium plating risk
        - Blue dashed lines = the 37°C body temperature row (ICD operating condition)

        **Key findings from the simulation:**

        - **At 25°C (room temperature):** Safe up to 2.0C. At 3.0C and
          above, temperature rise exceeds 10°C and lithium plating risk
          appears. This is the standard lab testing condition.

        - **At 37°C (body temperature):** Safe only up to 1.0C. The safe
          operating window is narrower because the Arrhenius factor
          (~2.1x at 37°C vs 25°C) accelerates side reactions including
          lithium plating at lower current rates.

        - **At 45°C (elevated, e.g. fever):** Safe only at 0.5C. Even
          moderate C-rates produce plating risk at this temperature.
          This is critical for ICD design — a patient running a fever
          narrows the safe operating window further.

        - **The ICD design implication:** ICDs are designed to operate
          at well below 0.5C during normal pacing, with brief high-rate
          pulses only during defibrillation events. This simulation
          confirms why that design choice is necessary.
        """)

    with tab2:
        st.markdown("#### Temperature Rise — T_ambient = 37°C (Body Temp)")
        if os.path.exists("outputs/module3_temperature_rise.png"):
            st.image("outputs/module3_temperature_rise.png",
                     use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        This shows how **cell temperature evolves over time** during
        discharge at different C-rates, with the ambient temperature
        fixed at 37°C (body temperature). Model resistance input is
        the real NASA B0005 measured Re = 0.0447 Ohm.

        **Key findings:**

        - **At 0.5C:** Temperature rise is only +1.09°C above ambient,
          reaching 38.09°C. This is well within safe limits and explains
          why ICD normal pacing operates at very low C-rates.

        - **At 1.0C:** Temperature rise is +2.63°C, reaching 39.63°C.
          Still safe — just below the 45°C safety threshold with margin.
          This is the maximum safe C-rate at body temperature per
          this simulation.

        - **At 2.0C and above:** Temperature approaches and exceeds
          the 45°C safety threshold. At 4.0C the simulated temperature
          rises by +21.80°C above body temperature, reaching 58.8°C —
          thermally catastrophic for an implanted device.

        - **Heat dissipation matters:** The low h*A value (0.018 W/K)
          models the poor heat dissipation of an implanted device
          surrounded by tissue. The same cell in a laptop with forced
          air cooling would have h*A ~10x higher and be safe at
          much higher C-rates.
        """)

    with tab3:
        st.markdown("#### Simulated Discharge Voltage Curves")
        if os.path.exists("outputs/module3_voltage_curves.png"):
            st.image("outputs/module3_voltage_curves.png",
                     use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        Simulated discharge voltage curves at three ambient temperatures
        for all 5 C-rates. Each subplot shows how temperature affects
        the voltage response.

        **Key findings:**

        - **Higher C-rates produce lower voltage** — the increased
          current causes a larger I×R drop across the internal
          resistance, reducing terminal voltage. At 4.0C, the
          initial voltage is visibly lower than at 0.5C.

        - **Higher temperature slightly reduces voltage** — the
          Arrhenius correction increases effective resistance at
          higher temperatures (via accelerated side reactions),
          causing additional voltage depression.

        - **The 2.7V cutoff line** represents the minimum discharge
          voltage. Cells at higher C-rates reach this cutoff sooner
          (lower SOC on x-axis), delivering less total charge —
          consistent with the capacity fade seen at high C-rates.

        - **ICD pulse implication:** The defibrillation pulse
          requires the battery to deliver high current briefly.
          At elevated temperature, the available voltage during
          the pulse is lower, which affects the energy delivered
          to the heart. This is why ICD batteries must maintain
          low internal resistance throughout their life.
        """)

    with tab4:
        st.markdown("#### Heat Generation Rate — T_ambient = 37°C")
        if os.path.exists("outputs/module3_heat_generation.png"):
            st.image("outputs/module3_heat_generation.png",
                     use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

        st.markdown("#### What This Shows")
        st.markdown("""
        Heat generation rate (in milliwatts) during discharge at
        37°C body temperature. Two mechanisms contribute:

        **Q_irrev = I² × R** (irreversible Joule heating)
        Scales with the square of current — doubling the C-rate
        quadruples the Joule heating. This is the dominant term
        at high C-rates.

        **Q_rev = I × T × |dU/dT|** (reversible entropic heat)
        From the entropy change during lithium intercalation/
        deintercalation. Scales linearly with current and is
        significant at body temperature (T = 310K).

        **Key findings:**

        - **At 0.5C:** Total heat generation is only ~2 mW —
          the implanted device can dissipate this to surrounding
          tissue without meaningful temperature rise.

        - **At 2.0C:** Heat generation is ~30 mW — exceeds the
          dissipation capacity of the implanted device (h*A = 18 mW/K),
          causing temperature to accumulate over the discharge.

        - **At 4.0C:** Heat generation peaks above 120 mW —
          6x the safe dissipation limit. This produces the
          +21.8°C temperature rise seen in the safety envelope.

        - **Model input:** R = 0.0447 Ohm from real NASA B0005
          EIS measurement. This grounds the heat generation
          calculation in real measured data.
        """)

    if os.path.exists("data/thermal_summary.csv"):
        st.markdown("---")
        st.markdown("#### Full Simulation Results Table")
        df = pd.read_csv("data/thermal_summary.csv")
        df['plating_risk'] = df['plating_risk'].map(
            {True: 'PLATING RISK', False: 'Safe'})
        df.columns = ['C-rate', 'T Ambient (°C)',
                      'Max ΔT (°C)', 'Status']
        st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────

elif page == "👤  About":

    st.title("About This Project")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Background

        This toolkit was developed as part of my MS thesis in Materials
        Science & Engineering at Purdue University, where I designed
        Li-ion cells for implantable cardiac defibrillators (ICD).

        The core engineering problem: an ICD battery cannot be easily
        replaced once implanted. It must operate reliably for 7-10 years
        at 37°C body temperature, in a confined space with minimal
        heat dissipation, and must never fail.

        The analytical techniques in this toolkit — capacity fade
        characterization, electrochemical impedance analysis, and
        thermal modeling — are the exact workflows used by battery
        engineers to qualify cells for safety-critical applications.

        ### Technical Methodology

        **Module 1 — Capacity Fade & RUL**
        Real discharge capacity data from NASA PCOE B0005-B0018.
        636 real cycles across 4 cells. Feature extraction from
        first 30 cycles (formation window). Key finding: capacity
        fade slope in early cycles is the strongest predictor of
        total cycle life (Pearson r = 0.815).

        **Module 2 — EIS Analysis**
        1,112 real EIS measurements (278 per cell). Randles circuit
        parameters Re and Rct tracked across full cell life.
        B0006 showed Re +20.2% and Rct +27.3% over its lifetime.
        Both parameters rise before capacity crosses EOL threshold —
        confirmed as early warning indicators.

        **Module 3 — Thermal Safety**
        Physics-based DFN thermal model. Re input from real NASA
        B0005 measurement (0.0447 Ohm). 15 simulations across 5
        C-rates × 3 temperatures. Safe C-rate drops from 2.0C at
        25°C to 1.0C at 37°C body temperature.
        """)

        st.markdown("### Dataset & Citations")
        st.markdown("""
        - **Saha, B. & Goebel, K. (2007).** NASA PCOE Battery Dataset.
          Cells B0005, B0006, B0007, B0018. NASA Ames Research Center.
          [Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

        - **Doyle, Fuller, Newman (1993).** J. Electrochem. Soc., 140(6).
          DFN model for thermal simulation.
          [Paper](https://doi.org/10.1149/1.2221597)

        - **Severson et al. (2019).** Nature Energy, 4, 383-391.
          Early-cycle feature engineering methodology.
          [Paper](https://doi.org/10.1038/s41560-019-0356-8)

        - **Birkl et al. (2017).** Journal of Power Sources, 341, 373-386.
          Degradation diagnostics and impedance parameters.
          [Paper](https://doi.org/10.1016/j.jpowsour.2016.09.073)
        """)

    with col2:
        st.markdown("### Contact")
        st.markdown("""
        **Narain Karthikeyan**

        MS Materials Science & Engineering
        Purdue University (GPA 3.7)

        📧 nkarthik4004@gmail.com
        🔗 linkedin.com/in/narainkarthi21
        💻 github.com/Narain115

        ---

        **Industry Experience**

        AMD — Packaging Engineer
        HBM wafer yield optimization
        Python/JMP/SQL analytics

        Plastic Recycling Inc — R&D Engineer
        FTIR, DSC, TGA, TMA, DMA
        Polymer characterization

        ---

        **Target Roles**

        Battery Engineer
        Process Engineer
        Materials Engineer
        R&D Engineer
        """)

        st.markdown("---")
        st.markdown("### Stack")
        st.code("""
Python 3.11
Streamlit 1.35
scipy (NASA .mat parsing)
matplotlib 3.8
pandas 2.2
numpy 1.26
scikit-learn 1.4
        """)