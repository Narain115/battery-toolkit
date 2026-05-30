# Li-ion Battery Degradation & State Estimation Toolkit

**MS Thesis Project — Materials Science & Engineering, Purdue University**  
**Author:** Narain Karthikeyan | nkarthik4004@gmail.com  
**Live Demo:** [battery-toolkit-ojufrsyarlaw6ykssemseq.streamlit.app](https://battery-toolkit-ojufrsyarlaw6ykssemseq.streamlit.app)

---

## Project Motivation

During my MS thesis designing Li-ion cells for implantable cardiac defibrillators (ICD), I needed analytical tools to characterize battery degradation, predict end-of-life, and verify safe operating envelopes. An ICD battery cannot be easily replaced once implanted — it must last 7-10 years, operate at 37°C body temperature, and never fail.

The analytical techniques in this toolkit — capacity fade characterization, electrochemical impedance spectroscopy (EIS), and thermal modeling — are the exact workflows used by battery engineers to qualify cells for safety-critical applications. This toolkit demonstrates all three workflows using real laboratory data from NASA Ames Research Center.

---

## Dataset

**NASA PCOE Battery Dataset**  
Saha, B. & Goebel, K. (2007). NASA Ames Prognostics Data Repository.  
[https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Four 18650 Li-ion cells cycled to end-of-life under controlled conditions:

| Cell | Initial Capacity | EOL Cycle | Total Fade | EIS Measurements |
|------|-----------------|-----------|------------|-----------------|
| B0005 | 1.842 Ah | Cycle 125 | 28.0% | 278 |
| B0006 | 2.018 Ah | Cycle 109 | 41.2% | 278 |
| B0007 | 1.883 Ah | Cycle 168 | ~24% | 278 |
| B0018 | 1.840 Ah | Cycle 97 | ~27% | 278 |

**Protocol:** 1.5A CC-CV charge to 4.2V | 2A CC discharge | 24°C room temperature  
**EOL criterion:** 30% capacity fade from nominal (2Ah to 1.4Ah)  
**EIS:** Frequency sweep 0.1 Hz to 5 kHz, periodic throughout cycle life

---

## Three Modules

### Module 1: Remaining Useful Life Prediction
**File:** `modules/module1_rul.py`

Analyzes real capacity fade trajectories across 636 discharge cycles. Extracts early-cycle features from the first 30 cycles (formation window) to characterize degradation behavior before full cycle life is known.

**Key results from real NASA data:**
- B0007 lasted 73% longer than B0018 (168 vs 97 cycles) despite similar initial capacity — confirming that initial capacity alone does not predict cycle life
- B0005 early-cycle capacity fade slope: -0.000904 Ah/cycle
- B0006 early-cycle capacity fade slope: -0.005481 Ah/cycle (6x steeper than B0005)
- Capacity fade slope (Pearson r = 0.815) is the strongest early predictor of total cycle life
- Initial capacity correlation with cycle life: r = -0.124 (weakest predictor)
- Voltage curve shape change is detectable by mid-life, preceding measurable capacity fade

**Engineering implication:** Cell-to-cell variability is significant — a 73-cycle gap between best and worst performing cells from the same batch. Individual cell qualification is required for safety-critical applications.

**Outputs:** `outputs/module1_*.png`

---

### Module 2: Electrochemical Impedance Spectroscopy
**File:** `modules/module2_eis.py`

Analyzes 1,112 real EIS measurements (278 per cell) across full cycle life. Extracts Re (electrolyte resistance) and Rct (charge transfer resistance) from the Randles circuit model and tracks their evolution with degradation.

**Randles circuit model:**
```
Z(ω) = Re + 1 / (jωCdl + 1/(Rct + Z_Warburg))
```

**Key results from real NASA data:**

| Cell | Re Initial | Re Final | Re Change | Rct Initial | Rct Final | Rct Change |
|------|-----------|---------|-----------|------------|---------|-----------|
| B0005 | 0.0447 Ω | 0.0500 Ω | +12.0% | 0.0695 Ω | 0.0748 Ω | +7.7% |
| B0006 | 0.0612 Ω | 0.0736 Ω | +20.2% | 0.0785 Ω | 0.1000 Ω | +27.3% |

- Re rises due to SEI layer growth on the graphite anode (diffusion-limited, square-root kinetics)
- Rct rises due to active material loss (reduced electrochemically active surface area)
- Both parameters rise before capacity crosses the 1.4 Ah EOL threshold — confirmed as leading indicators
- B0006 shows the largest impedance increase, consistent with its steepest capacity fade

**Engineering implication:** EIS is non-destructive and takes minutes. Impedance monitoring can detect degradation before capacity visibly drops — critical for ICD batteries where full discharge testing in vivo is not possible.

**Outputs:** `outputs/module2_*.png`

---

### Module 3: Thermal Safe Operating Envelope
**File:** `modules/module3_thermal.py`

Physics-based DFN-inspired thermal model simulating ICD battery behavior at body temperature (37°C). Model resistance input grounded in real NASA B0005 EIS measurement (Re = 0.0447 Ω).

**Thermal model:**
```
m·Cp·dT/dt = Q_irrev + Q_rev - Q_diss
Q_irrev = I²·R  (Joule heating)
Q_rev   = I·T·|dU/dT|  (entropic heat)
Q_diss  = h·A·(T_cell - T_ambient)  (low — implanted device)
```

**Arrhenius correction:** Ea/R = 6500 K → side reactions ~2.1x faster at 37°C vs 25°C

**Key results:**

| C-rate | 25°C ΔT | Status | 37°C ΔT | Status | 45°C ΔT | Status |
|--------|---------|--------|---------|--------|---------|--------|
| 0.5C | +0.99°C | Safe | +1.09°C | Safe | +1.19°C | Safe |
| 1.0C | +2.29°C | Safe | +2.63°C | Safe | +3.00°C | Safe |
| 2.0C | +5.89°C | Safe | +7.12°C | Safe | +8.56°C | Safe |
| 3.0C | +10.81°C | Safe | +13.54°C | **PLATING** | +16.79°C | **PLATING** |
| 4.0C | +16.97°C | Safe | +21.80°C | **PLATING** | +27.63°C | **PLATING** |

- Safe C-rate ceiling drops from 2.0C at 25°C to 1.0C at 37°C body temperature
- At 45°C (patient fever), safe ceiling drops further to 1.0C with plating risk at 3.0C+
- Model note: NASA cells are 1.856 Ah; ICD cell capacity modeled at 0.5 Ah (typical small form factor)

**Engineering implication:** Body temperature significantly narrows the safe operating window. ICD batteries operate below 0.5C during normal pacing for this reason, with brief high-rate pulses only during defibrillation events.

**Outputs:** `outputs/module3_*.png`

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Narain115/battery-toolkit.git
cd battery-toolkit

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NASA dataset files to data/
# B0005.mat, B0006.mat, B0007.mat, B0018.mat
# From: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

# Run modules to generate plots
python modules/module1_rul.py
python modules/module2_eis.py
python modules/module3_thermal.py

# Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
battery_toolkit/
├── data/                    # NASA .mat files (download separately)
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
├── modules/
│   ├── module1_rul.py       # Capacity fade & RUL prediction
│   ├── module2_eis.py       # EIS analysis
│   └── module3_thermal.py   # Thermal safety envelope
├── outputs/                 # Generated plots (PNG)
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## References

1. Saha, B. & Goebel, K. (2007). Battery Data Set. NASA Ames Prognostics Data Repository. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

2. Doyle, M., Fuller, T.F. & Newman, J. (1993). Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell. *Journal of the Electrochemical Society*, 140(6), 1526-1533. https://doi.org/10.1149/1.2221597

3. Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383-391. https://doi.org/10.1038/s41560-019-0356-8

4. Birkl, C.R. et al. (2017). Degradation diagnostics for lithium ion cells. *Journal of Power Sources*, 341, 373-386. https://doi.org/10.1016/j.jpowsour.2016.09.073

---

## Author

**Narain Karthikeyan**  
MS Materials Science & Engineering, Purdue University (GPA 3.7)  
nkarthik4004@gmail.com | [linkedin.com/in/narainkarthi21](https://linkedin.com/in/narainkarthi21)