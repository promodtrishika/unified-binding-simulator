# Streamlit App: Unified Binding + Quantum Visibility

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from fpdf import FPDF
import base64

# Constants
G = 6.67430e-11
c = 3e8
k_v = 0.1
k_D = 0.1
beta = 0.5
R_ref = 1.0
D_min = 1.0

STAGE_THRESHOLDS = {
    1: 0.5,
    2: 0.2,
    3: 0.01
}

def schwarzschild_radius(M):
    return 2 * G * M / c**2

def gravitational_time_dilation(r, r_s):
    return np.sqrt(1 - r_s / r) if r > r_s else 0.0

def relativistic_gamma(v):
    return 1.0 / np.sqrt(1 - (v / c) ** 2) if v < c else 1.0

def calculate_binding(R_eff, D, dR_dt=0, v=0, phi_size=0, phi_motion=0, phi_coupling=0, gamma_tau=1.0, eta_space=0.0):
    size_term = R_eff / R_ref
    distance_term = D_min / D
    dynamic_size_term = 1 + k_v * (dR_dt / c)
    doppler_term = 1 + k_D * (v / c)
    direction_term = abs(np.cos(np.radians(phi_size))) * abs(np.cos(np.radians(phi_motion))) * (1 + beta * np.cos(np.radians(phi_coupling)))
    space_revelation_term = 1 - eta_space
    return size_term * distance_term * dynamic_size_term * doppler_term * direction_term * space_revelation_term * gamma_tau

def determine_stage(alpha):
    for stage, threshold in STAGE_THRESHOLDS.items():
        if alpha >= threshold:
            return stage
    return 4

def quantum_visibility(alpha, decay_rate=0.3):
    return np.exp(-decay_rate * (1 - alpha))

st.title("Unified Binding + Quantum Visibility Simulator")

# Controls
M = st.sidebar.number_input("Mass (kg)", value=1e30)
v_obs = st.sidebar.slider("Observer velocity (fraction of c)", 0.0, 0.99, 0.8)
t_max = st.sidebar.slider("Simulation time (s)", 1, 20, 10)
phi_size = st.sidebar.slider("Size angle (°)", 0, 180, 90)
phi_motion = st.sidebar.slider("Motion angle (°)", 0, 180, 90)
phi_coupling = st.sidebar.slider("Coupling angle (°)", 0, 180, 90)
quantum_mode = st.sidebar.checkbox("Enable Quantum Visibility", value=True)
decay_rate = st.sidebar.slider("Quantum decay rate", 0.01, 1.0, 0.3)

# Simulation
time_steps = np.linspace(0, t_max, 50)
r_s = schwarzschild_radius(M)
data = []

for t in time_steps:
    Rx = max(1.0 - 0.03 * t, 0.01)
    Ry = max(1.0 - 0.01 * t, 0.01)
    Rz = max(1.0 - 0.02 * t, 0.01)
    R_eff = (Rx * Ry * Rz) ** (1/3)
    D = max(10 - 0.3 * t, 1.1 * r_s)
    gamma_rel = 1.0 / relativistic_gamma(v_obs * c)
    gamma_grav = gravitational_time_dilation(D, r_s)
    gamma_combined = gamma_grav * gamma_rel
    dR_dt = -0.03
    eta_space = 0.01 * t
    alpha = calculate_binding(R_eff, D, dR_dt, v_obs * c, phi_size, phi_motion, phi_coupling, gamma_combined, eta_space)
    stage = determine_stage(alpha)
    qv = quantum_visibility(alpha, decay_rate) if quantum_mode else None
    data.append((t, alpha, stage, R_eff, D, gamma_combined, qv))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Time", "Alpha", "Stage", "R_eff", "Distance", "Gamma", "QVisibility"])

# Plot
st.subheader("Binding Coefficient α(t)")
st.line_chart(df.set_index("Time")["Alpha"])
if quantum_mode:
    st.subheader("Quantum Visibility (probability)")
    st.line_chart(df.set_index("Time")["QVisibility"])

# Data Table
st.subheader("Simulation Data")
st.dataframe(df.round(5))

# CSV Export
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "binding_simulation.csv", "text/csv")

# PDF Export
def create_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Unified Binding Simulation Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    for i in range(min(len(dataframe), 30)):
        row = dataframe.iloc[i]
        qv_str = f" QV={row['QVisibility']:.3f}" if quantum_mode else ""
        line = f"t={row['Time']:.1f}s  α={row['Alpha']:.4f}  Stage={int(row['Stage'])}  R_eff={row['R_eff']:.3f}  D={row['Distance']:.3f}  γ={row['Gamma']:.4f}{qv_str}"
        pdf.cell(200, 8, txt=line, ln=1)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

if st.button("Generate PDF Report"):
    pdf_bytes = create_pdf(df)
    b64_pdf = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="binding_report.pdf">Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)
