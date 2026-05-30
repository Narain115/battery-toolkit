"""
Li-ion Battery Degradation and State Estimation Toolkit
=======================================================
Streamlit dashboard — dark navy/amber theme

Narain Karthikeyan | MS Materials Science, Purdue University
Thesis: Li-ion cells for implantable cardiac defibrillators (ICD)
GitHub: github.com/Narain115/battery-toolkit
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


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "🏠  Overview":

    st.title("Li-ion Battery Degradation & State Estimation Toolkit")
    st.markdown("**Thesis project — MS Materials Science & Engineering, Purdue University**")

    st.markdown("""
    During my MS thesis designing Li-ion cells for **implantable cardiac
    defibrillators (ICD)**, I needed analytical tools to predict battery
    end-of-life, characterize degradation mechanisms, and verify safe
    operating envelopes. I built this toolkit to solve those problems.

    An ICD is surgically implanted in the chest. The battery cannot be
    easily replaced. It must last **7-10 years**, operate at **37°C body
    temperature**, and never fail. There is no low-battery warning.
    Every module maps to a real engineering challenge in safety-critical
    battery qualification.
    """)

    st.markdown("---")
    st.markdown("### Key Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="NASA Cells Analyzed", value="4 cells",
                  delta="B0005, B0006, B0007, B0018")
    with col2:
        st.metric(label="Real Discharge Cycles", value="636 cycles",
                  delta="Real NASA measurements")
    with col3:
        st.metric(label="Safe C-rate at 37°C", value="1.0C",
                  delta="Body temperature limit")
    with col4:
        st.metric(label="Real EIS Measurements", value="1,112",
                  delta="Across 4 NASA cells")

    st.markdown("---")
    st.markdown("### Three Modules, Three Engineering Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Module 1: RUL Prediction")
        st.markdown("""
        **Question:** Can we predict when this battery will die,
        before the device is implanted?

        **Method:** Real capacity fade analysis from NASA cycling
        data. Early-cycle feature extraction from first 30 cycles.
        Voltage curve evolution across full cell life.

        **Dataset:** NASA PCOE B0005-B0018 | 636 real cycles
        """)

    with col2:
        st.markdown("#### ⚡ Module 2: EIS Analysis")
        st.markdown("""
        **Question:** How do we detect internal degradation
        before capacity drops?

        **Method:** Real Nyquist plots from NASA impedance
        measurements. Re and Rct evolution across 278
        measurements per cell. Capacity correlation analysis.

        **Dataset:** NASA PCOE — 1,112 real EIS measurements
        """)

    with col3:
        st.markdown("#### 🌡️ Module 3: Thermal Safety")
        st.markdown("""
        **Question:** What operating conditions are safe at
        37°C body temperature?

        **Method:** DFN-inspired thermal model. Re input from
        real NASA B0005 EIS (0.0447 Ohm). 15 simulations
        across 5 C-rates × 3 temperatures.

        **Reference:** Doyle-Fuller-Newman (1993)
        """)

    st.markdown("---")
    st.markdown("### Dataset — NASA PCOE Battery Dataset")
    st.markdown("""
    All cycling and impedance data comes from real laboratory measurements
    at **NASA Ames Research Center, Prognostics Center of Excellence**.

    - **Cells:** B0005, B0006, B0007, B0018 — 18650 Li-ion, 2Ah nominal
    - **Protocol:** 1.5A CC-CV charge to 4.2V | 2A CC discharge
    - **EIS:** Frequency sweep 0.1 Hz to 5 kHz, periodic throughout life
    - **EOL criterion:** 30% capacity fade (2Ah to 1.4Ah)
    - **Temperature:** 24°C room temperature

    Saha, B. & Goebel, K. (2007). Battery Data Set.
    [NASA PCOE Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
    """)

    st.markdown("---")
    st.markdown("### Literature References")
    st.markdown("""
    1. Saha, B. & Goebel, K. (2007). Battery Data Set. NASA PCOE Data Repository.
       [Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

    2. Doyle, M., Fuller, T.F. & Newman, J. (1993). Modeling of galvanostatic charge
       and discharge. *Journal of the Electrochemical Society*, 140(6), 1526-1533.
       [Paper](https://doi.org/10.1149/1.2221597)

    3. Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life.
       *Nature Energy*, 4, 383-391. (Feature engineering methodology)
       [Paper](https://doi.org/10.1038/s41560-019-0356-8)

    4. Birkl, C.R. et al. (2017). Degradation diagnostics for lithium ion cells.
       *Journal of Power Sources*, 341, 373-386.
       [Paper](https://doi.org/10.1016/j.jpowsour.2016.09.073)
    """)


# ─────────────────────────────────────────────
# PAGE: MODULE 1
# ─────────────────────────────────────────────

elif page == "📈  Module 1: RUL Prediction":

    st.title("Module 1: Remaining Useful Life Prediction")
    st.markdown(
        "**Dataset:** NASA PCOE Battery Dataset — "
        "Real measurements from NASA Ames Research Center | "
        "B0005, B0006, B0007, B0018"
    )
    st.info(
        "**ICD Context:** Formation cycling happens in the factory before "
        "the cell is packaged into a device. Early-cycle analysis enables "
        "a qualification go/no-go decision before implantation. "
        "B0018 shows the fastest degradation — it would be flagged for rejection."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("B0005 EOL", "Cycle 125", "1.856 → 1.325 Ah")
    with col2:
        st.metric("B0006 EOL", "Cycle 109", "2.035 → 1.186 Ah")
    with col3:
        st.metric("B0007 EOL", "Cycle 168", "1.891 → 1.432 Ah")
    with col4:
        st.metric("B0018 EOL", "Cycle 97", "1.855 → 1.341 Ah")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Capacity Fade", "Voltage Evolution",
        "Formation Window", "Feature Correlation"
    ])

    with tab1:
        st.markdown("#### Real Capacity Fade — All 4 NASA Cells")
        st.markdown(
            "Every point is a real discharge measurement from NASA Ames lab. "
            "Dashed red line = EOL threshold (1.4 Ah = 30% fade). "
            "Dashed white line = formation window cutoff (30 cycles)."
        )
        if os.path.exists("outputs/module1_capacity_fade.png"):
            st.image("outputs/module1_capacity_fade.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

    with tab2:
        st.markdown("#### Discharge Voltage Curve Evolution")
        st.markdown(
            "Real voltage profiles at early, mid, and late cycle life. "
            "Voltage depression with aging indicates rising internal resistance "
            "and active material loss — affects ICD defibrillation pulse delivery."
        )
        if os.path.exists("outputs/module1_predicted_vs_actual.png"):
            st.image("outputs/module1_predicted_vs_actual.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

    with tab3:
        st.markdown("#### Formation Window — First 30 Cycles")
        st.markdown(
            "Real measurements from the first 30 cycles only. "
            "B0018 already shows distinctly lower capacity by cycle 10 — "
            "an early warning signal detectable before implantation."
        )
        if os.path.exists("outputs/module1_shap.png"):
            st.image("outputs/module1_shap.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")

    with tab4:
        st.markdown("#### Feature Correlation with Cycle Life")
        st.markdown(
            "Capacity fade slope has the strongest correlation (0.815) "
            "with total cycle life. Higher slope = faster degradation = "
            "shorter life. Detectable from the first 30 cycles."
        )
        if os.path.exists("outputs/module1_deltaQ.png"):
            st.image("outputs/module1_deltaQ.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first.")


# ─────────────────────────────────────────────
# PAGE: MODULE 2
# ─────────────────────────────────────────────

elif page == "⚡  Module 2: EIS Analysis":

    st.title("Module 2: Electrochemical Impedance Spectroscopy")
    st.markdown(
        "**Dataset:** NASA PCOE Battery Dataset — "
        "Real EIS measurements | 278 spectra per cell | "
        "Frequency sweep 0.1 Hz to 5 kHz"
    )
    st.info(
        "**ICD Context:** EIS is used in battery qualification protocols "
        "for implantable devices. Impedance rise PRECEDES capacity fade. "
        "For a 7-10 year implanted device, early detection of degradation "
        "is the only clinically meaningful detection strategy."
    )

    st.markdown("---")

    if os.path.exists("data/eis_summary.csv"):
        df = pd.read_csv("data/eis_summary.csv")
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        for col, (_, row) in zip(cols, df.iterrows()):
            with col:
                st.metric(
                    f"{row['cell']} Re increase",
                    f"+{row['Re_increase_pct']:.1f}%",
                    f"Rct +{row['Rct_increase_pct']:.1f}%"
                )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("EIS Measurements", "278 per cell", "Real NASA data")
        with col2:
            st.metric("Cells analyzed", "4 cells", "B0005-B0018")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Nyquist Plots", "Re & Rct Evolution",
        "Capacity Correlation", "Impedance Magnitude"
    ])

    with tab1:
        st.markdown("#### Real Nyquist Plots — NASA EIS Measurements")
        st.markdown(
            "Real complex impedance spectra at early, mid, and late cycle life. "
            "Semicircle diameter = Rct. High-frequency intercept = Re. "
            "Growing semicircle = increasing charge transfer resistance with aging."
        )
        if os.path.exists("outputs/module2_nyquist.png"):
            st.image("outputs/module2_nyquist.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

    with tab2:
        st.markdown("#### Re and Rct Evolution — Real Measurements")
        st.markdown(
            "Re increases as the SEI layer grows on the anode. "
            "Rct increases as active material degrades. "
            "Both trends are visible long before capacity reaches EOL threshold."
        )
        if os.path.exists("outputs/module2_parameter_evolution.png"):
            st.image("outputs/module2_parameter_evolution.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

    with tab3:
        st.markdown("#### Real Impedance vs Capacity Correlation")
        st.markdown(
            "Shows that rising Re and Rct correlate directly with capacity fade. "
            "Non-destructive state-of-health estimation — critical for ICD "
            "monitoring where full discharge testing is not possible in vivo."
        )
        if os.path.exists("outputs/module2_capacity_vs_impedance.png"):
            st.image("outputs/module2_capacity_vs_impedance.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")

    with tab4:
        st.markdown("#### Impedance Magnitude Evolution")
        st.markdown(
            "Overall impedance magnitude increases with aging across "
            "the full frequency range. Rising |Z| reduces available "
            "energy for the defibrillation pulse."
        )
        if os.path.exists("outputs/module2_bode.png"):
            st.image("outputs/module2_bode.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first.")


# ─────────────────────────────────────────────
# PAGE: MODULE 3
# ─────────────────────────────────────────────

elif page == "🌡️  Module 3: Thermal Safety":

    st.title("Module 3: Thermal Safe Operating Envelope")
    st.markdown(
        "**Model:** DFN-inspired thermal simulation | "
        "Re input from real NASA B0005 EIS (0.0447 Ohm) | "
        "ICD cell 0.5 Ah | 15 simulations"
    )
    st.info(
        "**ICD Context:** ICDs operate at 37°C body temperature. "
        "Heat cannot dissipate freely — the device is surrounded by tissue. "
        "Model resistance grounded in real NASA B0005 measurements. "
        "Required for FDA 510(k) submissions."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Safe C-rate at 37°C", "1.0C", "Body temperature limit")
    with col2:
        st.metric("Safe C-rate at 25°C", "2.0C", "Room temperature")
    with col3:
        st.metric("Plating Risk Cases", "7 / 15", "Simulations flagged")
    with col4:
        st.metric("NASA Re Input", "0.0447 Ω", "Real B0005 measurement")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Safety Envelope", "Temperature Rise",
        "Voltage Curves", "Heat Generation"
    ])

    with tab1:
        st.markdown("#### Thermal Safe Operating Envelope")
        st.markdown(
            "Color = max temperature rise. PLATING = lithium plating risk. "
            "Blue lines highlight the 37°C body temperature row. "
            "Safe C-rate drops from 2.0C at 25°C to 1.0C at body temperature."
        )
        if os.path.exists("outputs/module3_safety_envelope.png"):
            st.image("outputs/module3_safety_envelope.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

    with tab2:
        st.markdown("#### Temperature Rise — T_ambient = 37°C")
        st.markdown(
            "At body temperature, higher C-rates push the cell toward "
            "the 45°C safety threshold. Model resistance from real "
            "NASA B0005 EIS measurement (Re = 0.0447 Ohm)."
        )
        if os.path.exists("outputs/module3_temperature_rise.png"):
            st.image("outputs/module3_temperature_rise.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

    with tab3:
        st.markdown("#### Simulated Discharge Voltage Curves")
        st.markdown(
            "Discharge curves at 25°C, 37°C, and 45°C. "
            "Higher temperature reduces the effective voltage window "
            "via Arrhenius kinetics (Ea/R = 6500 K)."
        )
        if os.path.exists("outputs/module3_voltage_curves.png"):
            st.image("outputs/module3_voltage_curves.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

    with tab4:
        st.markdown("#### Heat Generation Rate — T_ambient = 37°C")
        st.markdown(
            "Q = I²R (Joule) + IT|dU/dT| (entropic). "
            "Higher C-rates generate heat faster than the implanted "
            "device can dissipate to surrounding tissue."
        )
        if os.path.exists("outputs/module3_heat_generation.png"):
            st.image("outputs/module3_heat_generation.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first.")

    if os.path.exists("data/thermal_summary.csv"):
        st.markdown("---")
        st.markdown("#### Full Simulation Results")
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

        Each module addresses a distinct qualification challenge.

        ### Technical Methodology

        **Module 1 — RUL Prediction**
        Real capacity fade analysis from NASA PCOE B0005-B0018.
        636 real discharge cycles across 4 cells. Early-cycle feature
        extraction from first 30 cycles. Voltage curve evolution
        showing degradation mechanisms across full cell life.

        **Module 2 — EIS Analysis**
        1,112 real EIS measurements from NASA PCOE dataset.
        Real Nyquist plots, Re and Rct evolution, capacity correlation.
        Impedance rise precedes capacity fade — early warning signal
        for ICD battery state-of-health monitoring.

        **Module 3 — Thermal Safety**
        DFN-inspired thermal model with Arrhenius-corrected kinetics.
        Model resistance grounded in real NASA B0005 EIS measurement
        (Re = 0.0447 Ohm). 15 simulations across 5 C-rates × 3 temps.
        Safe C-rate ceiling at 37°C body temperature: 1.0C.
        """)

        st.markdown("### Dataset & Citations")
        st.markdown("""
        - **Saha, B. & Goebel, K. (2007).** NASA PCOE Battery Dataset.
          Cells B0005, B0006, B0007, B0018. NASA Ames Research Center.
          [Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

        - **Doyle, Fuller, Newman (1993).** J. Electrochem. Soc., 140(6).
          [Paper](https://doi.org/10.1149/1.2221597)

        - **Severson et al. (2019).** Nature Energy, 4, 383-391.
          Feature engineering methodology.
          [Paper](https://doi.org/10.1038/s41560-019-0356-8)

        - **Birkl et al. (2017).** Journal of Power Sources, 341, 373-386.
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