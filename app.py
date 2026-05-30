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
    .stCaption { color: #94a3b8 !important; }
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
        st.metric(label="Cycle Life Prediction R²", value="0.966", delta="ElasticNet on 124 cells")
    with col2:
        st.metric(label="Prediction RMSE", value="90 cycles", delta="From first 50 cycles only")
    with col3:
        st.metric(label="Safe C-rate at 37°C", value="2.0C", delta="Body temperature limit")
    with col4:
        st.metric(label="EIS Rs Increase", value="+16.2%", delta="Over 150 cycles (SEI growth)")

    st.markdown("---")
    st.markdown("### Three Modules, Three Engineering Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Module 1: RUL Prediction")
        st.markdown("""
        **Question:** Can we predict when this battery will die,
        before the device is implanted?

        **Method:** Physics-informed features from Severson et al.
        (Nature Energy 2019) extracted from cycles 1-50.
        ElasticNet + Random Forest + SHAP explainability.

        **Dataset:** 124 LFP cells, Stanford-MIT
        """)

    with col2:
        st.markdown("#### ⚡ Module 2: EIS Analysis")
        st.markdown("""
        **Question:** How do we detect internal degradation
        before capacity drops?

        **Method:** Randles equivalent circuit fitting.
        Track Rs (SEI growth) and Rct (active material loss)
        across cycle life.

        **Dataset:** Calibrated to NASA PCOE B0005-B0018
        """)

    with col3:
        st.markdown("#### 🌡️ Module 3: Thermal Safety")
        st.markdown("""
        **Question:** What operating conditions are safe at
        37°C body temperature?

        **Method:** DFN-inspired thermal model, 15 simulations
        across 5 C-rates × 3 temperatures. Safe operating
        envelope with lithium plating onset boundary.

        **Reference:** Doyle-Fuller-Newman (1993)
        """)

    st.markdown("---")
    st.markdown("### Literature References & Datasets")
    st.markdown("""
    1. Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life
       before capacity degradation. *Nature Energy*, 4, 383-391.
       [Paper](https://doi.org/10.1038/s41560-019-0356-8) |
       [Dataset — Braatz Lab, MIT](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

    2. Saha, B. & Goebel, K. (2007). Battery Data Set. NASA PCOE Data Repository.
       [Dataset — NASA PCOE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

    3. Doyle, M., Fuller, T.F. & Newman, J. (1993). Modeling of galvanostatic charge
       and discharge. *Journal of the Electrochemical Society*, 140(6), 1526-1533.
       [Paper](https://doi.org/10.1149/1.2221597)

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
        "**Dataset:** Severson et al. (2019), Nature Energy — "
        "Stanford-MIT | 124 LFP/graphite cells"
    )
    st.info(
        "**ICD Context:** Formation cycling happens in the factory before "
        "the cell is packaged into a device. This model predicts full cycle "
        "life from the first 50 cycles only — enabling a qualification "
        "go/no-go decision before implantation."
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ElasticNet R²", "0.966", "RMSE: 90 cycles")
    with col2:
        st.metric("Random Forest R²", "0.929", "RMSE: 129 cycles")
    with col3:
        st.metric("Training Data", "First 50 cycles", "Formation window only")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Capacity Fade", "Predicted vs Actual", "SHAP Importance", "ΔQ(V) Curves"
    ])

    with tab1:
        st.markdown("#### Capacity Fade Trajectories — 20 Sample Cells")
        st.markdown("Color = cycle life (red = short-lived, blue = long-lived). Dashed line = 80% nominal capacity (end-of-life threshold).")
        if os.path.exists("outputs/module1_capacity_fade.png"):
            st.image("outputs/module1_capacity_fade.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first to generate this plot.")

    with tab2:
        st.markdown("#### Predicted vs Actual Cycle Life")
        st.markdown("Points on the diagonal = perfect prediction. ElasticNet outperforms Random Forest here, consistent with Severson et al. finding that the feature-life relationship is largely linear.")
        if os.path.exists("outputs/module1_predicted_vs_actual.png"):
            st.image("outputs/module1_predicted_vs_actual.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first to generate this plot.")

    with tab3:
        st.markdown("#### SHAP Feature Importance")
        st.markdown("Log variance of ΔQ(V) is the dominant predictor — consistent with the key finding of Severson et al. 2019. Internal resistance at cycle 2 is the second most important feature.")
        if os.path.exists("outputs/module1_shap.png"):
            st.image("outputs/module1_shap.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first to generate this plot.")

    with tab4:
        st.markdown("#### ΔQ(V) Curves by Cycle Life Bucket")
        st.markdown("Cells with shorter lives show a deeper negative dip in their ΔQ(V) curves during early cycling. This physical signature is the basis for early-cycle prediction.")
        if os.path.exists("outputs/module1_deltaQ.png"):
            st.image("outputs/module1_deltaQ.png", use_column_width=True)
        else:
            st.warning("Run module1_rul.py first to generate this plot.")


# ─────────────────────────────────────────────
# PAGE: MODULE 2
# ─────────────────────────────────────────────

elif page == "⚡  Module 2: EIS Analysis":

    st.title("Module 2: Electrochemical Impedance Spectroscopy")
    st.markdown(
        "**Dataset:** Calibrated to NASA PCOE Battery Dataset "
        "(Saha & Goebel, 2007) | Cells B0005-B0018"
    )
    st.info(
        "**ICD Context:** EIS is used in battery qualification protocols "
        "for implantable devices. Impedance rise PRECEDES capacity fade "
        "by hundreds of cycles. For a 7-10 year implanted device, early "
        "detection of degradation is the only clinically meaningful "
        "detection strategy."
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rs Increase", "+16.2%", "SEI layer growth")
    with col2:
        st.metric("Rct Increase", "+2.7%", "Active material loss")
    with col3:
        st.metric("EIS Spectra", "30 measurements", "Every 5 cycles")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Nyquist Plot", "Bode Plot", "Parameter Evolution", "Capacity Correlation"
    ])

    with tab1:
        st.markdown("#### Nyquist Plot — EIS Spectra Across Cycle Life")
        st.markdown("The semicircle diameter = Rct. The high-frequency x-intercept = Rs. As the cell ages, the semicircle grows and shifts right.")
        if os.path.exists("outputs/module2_nyquist.png"):
            st.image("outputs/module2_nyquist.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first to generate this plot.")

    with tab2:
        st.markdown("#### Bode Plot — Magnitude and Phase")
        st.markdown("Complementary to Nyquist. Shows how impedance magnitude and phase shift evolve with frequency across cycle life.")
        if os.path.exists("outputs/module2_bode.png"):
            st.image("outputs/module2_bode.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first to generate this plot.")

    with tab3:
        st.markdown("#### Randles Circuit Parameter Evolution")
        st.markdown("Rs follows square-root growth kinetics (SEI diffusion-limited). Rct grows exponentially (active material loss). Cdl decreases linearly (electroactive area reduction).")
        if os.path.exists("outputs/module2_parameter_evolution.png"):
            st.image("outputs/module2_parameter_evolution.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first to generate this plot.")

    with tab4:
        st.markdown("#### Impedance vs Capacity Correlation")
        st.markdown("Shows that Rs and Rct can predict remaining capacity without running a full discharge test — critical for ICD monitoring where full discharge is not possible in vivo.")
        if os.path.exists("outputs/module2_capacity_vs_impedance.png"):
            st.image("outputs/module2_capacity_vs_impedance.png", use_column_width=True)
        else:
            st.warning("Run module2_eis.py first to generate this plot.")


# ─────────────────────────────────────────────
# PAGE: MODULE 3
# ─────────────────────────────────────────────

elif page == "🌡️  Module 3: Thermal Safety":

    st.title("Module 3: Thermal Safe Operating Envelope")
    st.markdown(
        "**Model:** DFN-inspired thermal simulation | "
        "Li/MnO₂ ICD primary cell | 15 simulations"
    )
    st.info(
        "**ICD Context:** ICDs operate at 37°C body temperature. "
        "Heat cannot dissipate freely — the device is surrounded by tissue. "
        "This analysis maps the safe operating envelope and identifies "
        "C-rate/temperature combinations that risk lithium plating or "
        "thermal events. Required for FDA 510(k) submissions."
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Safe C-rate at 37°C", "2.0C", "Body temperature limit")
    with col2:
        st.metric("Safe C-rate at 25°C", "3.0C", "Room temperature limit")
    with col3:
        st.metric("Plating Risk Cases", "6 / 15", "Simulations flagged")
    with col4:
        st.metric("Max ΔT at 4.0C / 37°C", "+40.75°C", "Unsafe region")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Safety Envelope", "Temperature Rise", "Voltage Curves", "Heat Generation"
    ])

    with tab1:
        st.markdown("#### Thermal Safe Operating Envelope")
        st.markdown("The key deliverable. Color = max temperature rise. ⚠ = lithium plating risk. Blue dashed lines highlight the 37°C body temperature row. This plot maps directly to FDA 510(k) analysis.")
        if os.path.exists("outputs/module3_safety_envelope.png"):
            st.image("outputs/module3_safety_envelope.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first to generate this plot.")

    with tab2:
        st.markdown("#### Temperature Rise — T_ambient = 37°C")
        st.markdown("At body temperature, higher C-rates push the cell toward the 45°C safety threshold. The 4.0C case reaches +40.75°C rise — catastrophic for an implanted device.")
        if os.path.exists("outputs/module3_temperature_rise.png"):
            st.image("outputs/module3_temperature_rise.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first to generate this plot.")

    with tab3:
        st.markdown("#### Voltage Discharge Curves")
        st.markdown("Discharge curves at 25°C, 37°C, and 45°C. Higher temperature reduces the effective voltage window and accelerates side reactions via Arrhenius kinetics.")
        if os.path.exists("outputs/module3_voltage_curves.png"):
            st.image("outputs/module3_voltage_curves.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first to generate this plot.")

    with tab4:
        st.markdown("#### Heat Generation Rate — T_ambient = 37°C")
        st.markdown("Q_total = I²R (irreversible Joule heating) + IT|dU/dT| (reversible entropic heat). Higher C-rates generate heat faster than the implanted device can dissipate.")
        if os.path.exists("outputs/module3_heat_generation.png"):
            st.image("outputs/module3_heat_generation.png", use_column_width=True)
        else:
            st.warning("Run module3_thermal.py first to generate this plot.")

    if os.path.exists("data/thermal_summary.csv"):
        st.markdown("---")
        st.markdown("#### Full Simulation Results")
        df = pd.read_csv("data/thermal_summary.csv")
        df['plating_risk'] = df['plating_risk'].map({True: '⚠ Yes', False: '✓ Safe'})
        df.columns = ['C-rate', 'T Ambient (°C)', 'Max ΔT (°C)', 'Plating Risk']
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

        Each module in this toolkit addresses a distinct qualification
        challenge that arises when designing batteries for this
        safety-critical context.

        ### Technical Methodology

        **Module 1 — RUL Prediction**
        Physics-informed feature extraction from cycles 1-50, based on
        the delta_Q(V) methodology of Severson et al. (2019).
        ElasticNet regression (R²=0.966, RMSE=90 cycles) and
        Random Forest with SHAP explainability on 124 LFP cells.

        **Module 2 — EIS Analysis**
        Randles equivalent circuit fitting: Rs + (Rct ∥ Cdl) + Warburg.
        Synthetic data calibrated to NASA PCOE B0005-B0018 published
        parameter trends. Tracks SEI growth (Rs) and active material
        loss (Rct) as early degradation indicators.

        **Module 3 — Thermal Safety**
        DFN-inspired thermal model with Arrhenius-corrected kinetics.
        15 simulations across 5 C-rates × 3 temperatures.
        Safe C-rate ceiling at 37°C body temperature: 2.0C.
        """)

        st.markdown("### Dataset Citations")
        st.markdown("""
        - **Severson et al. (2019).** Nature Energy, 4, 383-391. Stanford-MIT, 124 LFP/graphite cells.
          [Paper](https://doi.org/10.1038/s41560-019-0356-8) |
          [Dataset](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

        - **Saha, B. & Goebel, K. (2007).** NASA PCOE Battery Dataset. Cells B0005, B0006, B0007, B0018.
          [Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

        - **Birkl et al. (2017).** Journal of Power Sources, 341, 373-386.
          [Paper](https://doi.org/10.1016/j.jpowsour.2016.09.073)

        - **Doyle, Fuller, Newman (1993).** J. Electrochem. Soc., 140(6).
          [Paper](https://doi.org/10.1149/1.2221597)
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

        • AMD — Packaging Engineer
          HBM wafer yield optimization
          Python/JMP/SQL analytics

        • Plastic Recycling Inc — R&D Engineer
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
scikit-learn 1.4.2
SHAP 0.45
matplotlib 3.8
pandas 2.2
numpy 1.26
scipy 1.13
        """)