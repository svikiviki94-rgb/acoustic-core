import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Acoustic Core Designer", layout="wide")

st.title("🔊 Acoustic Core Designer — Numerical (Geometry-Only)")
st.caption("Estimate thickness, porosity, or cell size to meet an attenuation target using your datasets. Material losses are neglected.")

# -----------------------------
# Embedded datasets
# -----------------------------

# PN attenuation (thickness sweep)
PN_TXT = """Frequency\t10mm\tFrequency\t15mm\tFrequency\t20mm
1000\t0.56332\t1000\t1.0348\t1000\t1.3171
1100\t0.59615\t1100\t1.1048\t1100\t1.3963
1200\t0.62977\t1200\t1.1755\t1200\t1.4739
1300\t0.66412\t1300\t1.2466\t1300\t1.5493
1400\t0.69914\t1400\t1.3177\t1400\t1.6217
1500\t0.73475\t1500\t1.3882\t1500\t1.6905
1600\t0.77088\t1600\t1.4579\t1600\t1.755
1700\t0.80745\t1700\t1.5263\t1700\t1.8148
1800\t0.84439\t1800\t1.5932\t1800\t1.8694
1900\t0.88159\t1900\t1.6581\t1900\t1.9183
2000\t0.91898\t2000\t1.7207\t2000\t1.9611
2100\t0.95646\t2100\t1.7808\t2100\t1.9975
2200\t0.99395\t2200\t1.838\t2200\t2.0273
2300\t1.0314\t2300\t1.8921\t2300\t2.0503
2400\t1.0686\t2400\t1.9428\t2400\t2.0663
2500\t1.1055\t2500\t1.99\t2500\t2.0753
2600\t1.1421\t2600\t2.0334\t2600\t2.0771
2700\t1.1783\t2700\t2.0729\t2700\t2.0719
2800\t1.2139\t2800\t2.1083\t2800\t2.0597
2900\t1.2489\t2900\t2.1394\t2900\t2.0408
3000\t1.2832\t3000\t2.1662\t3000\t2.0153
3100\t1.3167\t3100\t2.1885\t3100\t1.9837
3200\t1.3493\t3200\t2.2063\t3200\t1.9463
3300\t1.3811\t3300\t2.2194\t3300\t1.9037
3400\t1.4118\t3400\t2.2279\t3400\t1.8566
3500\t1.4414\t3500\t2.2317\t3500\t1.8058
3600\t1.4699\t3600\t2.2309\t3600\t1.7522
3700\t1.4971\t3700\t2.2253\t3700\t1.6972
3800\t1.5231\t3800\t2.215\t3800\t1.6421
3900\t1.5478\t3900\t2.2001\t3900\t1.5888
4000\t1.571\t4000\t2.1807\t4000\t1.5393
4100\t1.5928\t4100\t2.1568\t4100\t1.4967
4200\t1.6132\t4200\t2.1285\t4200\t1.4643
4300\t1.6319\t4300\t2.0961\t4300\t1.447
4400\t1.6492\t4400\t2.0595\t4400\t1.4516
4500\t1.6647\t4500\t2.0191\t4500\t1.4875
4600\t1.6787\t4600\t1.9751\t4600\t1.5696
4700\t1.6909\t4700\t1.9277\t4700\t1.7214
4800\t1.7014\t4800\t1.8772\t4800\t1.9827
4900\t1.7101\t4900\t1.8239\t4900\t2.4218
5000\t1.7171\t5000\t1.7682\t5000\t3.1478
5100\t1.7222\t5100\t1.7106\t5100\t4.2595
5200\t1.7256\t5200\t1.6514\t5200\t5.4123
5300\t1.7271\t5300\t1.5913\t5300\t5.2397
5400\t1.7267\t5400\t1.5308\t5400\t3.7994
5500\t1.7245\t5500\t1.4705\t5500\t2.5531
5600\t1.7204\t5600\t1.4113\t5600\t1.8119
5700\t1.7145\t5700\t1.3539\t5700\t1.4111
5800\t1.7067\t5800\t1.2994\t5800\t1.2046
5900\t1.697\t5900\t1.2489\t5900\t1.1094
6000\t1.6856\t6000\t1.2038\t6000\t1.0806
6100\t1.6722\t6100\t1.166\t6100\t1.0934
6200\t1.6571\t6200\t1.1381\t6200\t1.1331
"""

# Porosity sweep
POR_TXT = """Frequency\t90%\tFrequency\t80%\tFrequency\t70%
1000\t0.58926\t1000\t1.0348\t1000\t2.1246
1100\t0.62395\t1100\t1.1048\t1100\t2.2587
1200\t0.65873\t1200\t1.1755\t1200\t2.3923
1300\t0.69353\t1300\t1.2466\t1300\t2.5242
1400\t0.72826\t1400\t1.3177\t1400\t2.6534
1500\t0.76283\t1500\t1.3882\t1500\t2.7789
1600\t0.79713\t1600\t1.4579\t1600\t2.9001
1700\t0.83107\t1700\t1.5263\t1700\t3.0161
1800\t0.86453\t1800\t1.5932\t1800\t3.1262
1900\t0.89739\t1900\t1.6581\t1900\t3.23
2000\t0.92955\t2000\t1.7207\t2000\t3.327
2100\t0.96089\t2100\t1.7808\t2100\t3.4166
2200\t0.99131\t2200\t1.838\t2200\t3.4985
2300\t1.0207\t2300\t1.8921\t2300\t3.5724
2400\t1.0489\t2400\t1.9428\t2400\t3.638
2500\t1.0759\t2500\t1.99\t2500\t3.6951
2600\t1.1015\t2600\t2.0334\t2600\t3.7433
2700\t1.1258\t2700\t2.0729\t2700\t3.7827
2800\t1.1485\t2800\t2.1083\t2800\t3.8129
2900\t1.1696\t2900\t2.1394\t2900\t3.834
3000\t1.189\t3000\t2.1662\t3000\t3.8457
3100\t1.2067\t3100\t2.1885\t3100\t3.8481
3200\t1.2226\t3200\t2.2063\t3200\t3.8412
3300\t1.2367\t3300\t2.2194\t3300\t3.8248
3400\t1.2489\t3400\t2.2279\t3400\t3.799
3500\t1.2591\t3500\t2.2317\t3500\t3.764
3600\t1.2674\t3600\t2.2309\t3600\t3.7197
3700\t1.2738\t3700\t2.2253\t3700\t3.6664
3800\t1.2781\t3800\t2.215\t3800\t3.6042
3900\t1.2805\t3900\t2.2001\t3900\t3.5334
4000\t1.2809\t4000\t2.1807\t4000\t3.4543
4100\t1.2793\t4100\t2.1568\t4100\t3.3674
4200\t1.2758\t4200\t2.1285\t4200\t3.2732
4300\t1.2704\t4300\t2.0961\t4300\t3.1723
4400\t1.2631\t4400\t2.0595\t4400\t3.0656
4500\t1.254\t4500\t2.0191\t4500\t2.9542
4600\t1.2431\t4600\t1.9751\t4600\t2.8392
4700\t1.2305\t4700\t1.9277\t4700\t2.7222
4800\t1.2163\t4800\t1.8772\t4800\t2.6052
4900\t1.2006\t4900\t1.8239\t4900\t2.4906
5000\t1.1834\t5000\t1.7682\t5000\t2.3814
5100\t1.165\t5100\t1.7106\t5100\t2.2816
5200\t1.1453\t5200\t1.6514\t5200\t2.1962
5300\t1.1245\t5300\t1.5913\t5300\t2.1322
5400\t1.1028\t5400\t1.5308\t5400\t2.099
5500\t1.0803\t5500\t1.4705\t5500\t2.1104
5600\t1.0571\t5600\t1.4113\t5600\t2.1867
5700\t1.0335\t5700\t1.3539\t5700\t2.3588
5800\t1.0095\t5800\t1.2994\t5800\t2.671
5900\t0.98544\t5900\t1.2489\t5900\t3.172
6000\t0.96142\t6000\t1.2038\t6000\t3.8392
6100\t0.93767\t6100\t1.166\t6100\t4.3534
6200\t0.91438\t6200\t1.1381\t6200\t4.1405
"""

# Cell size sweep
CELL_TXT = """Frequency\t3mm\tFrequency\t5mm\tFrequency\t7.5mm
1000\t1.6676\t1000\t1.0348\t1000\t0.71282
1100\t1.735\t1100\t1.1048\t1100\t0.77806
1200\t1.8025\t1200\t1.1755\t1200\t0.84488
1300\t1.8697\t1300\t1.2466\t1300\t0.91287
1400\t1.9362\t1400\t1.3177\t1400\t0.98161
1500\t2.0017\t1500\t1.3882\t1500\t1.0507
1600\t2.0659\t1600\t1.4579\t1600\t1.1197
1700\t2.1283\t1700\t1.5263\t1700\t1.1882
1800\t2.1886\t1800\t1.5932\t1800\t1.2558
1900\t2.2467\t1900\t1.6581\t1900\t1.3221
2000\t2.3023\t2000\t1.7207\t2000\t1.3868
2100\t2.3551\t2100\t1.7808\t2100\t1.4494
2200\t2.4048\t2200\t1.838\t2200\t1.5098
2300\t2.4514\t2300\t1.8921\t2300\t1.5675
2400\t2.4946\t2400\t1.9428\t2400\t1.6222
2500\t2.5342\t2500\t1.99\t2500\t1.6737
2600\t2.5702\t2600\t2.0334\t2600\t1.7218
2700\t2.6024\t2700\t2.0729\t2700\t1.7661
2800\t2.6308\t2800\t2.1083\t2800\t1.8065
2900\t2.6552\t2900\t2.1394\t2900\t1.8428
3000\t2.6755\t3000\t2.1662\t3000\t1.8748
3100\t2.6918\t3100\t2.1885\t3100\t1.9023
3200\t2.704\t3200\t2.2063\t3200\t1.9253
3300\t2.712\t3300\t2.2194\t3300\t1.9435
3400\t2.716\t3400\t2.2279\t3400\t1.9568
3500\t2.7159\t3500\t2.2317\t3500\t1.9653
3600\t2.7117\t3600\t2.2309\t3600\t1.9687
3700\t2.7035\t3700\t2.2253\t3700\t1.9671
3800\t2.6914\t3800\t2.215\t3800\t1.9604
3900\t2.6754\t3900\t2.2001\t3900\t1.9486
4000\t2.6558\t4000\t2.1807\t4000\t1.9317
4100\t2.6326\t4100\t2.1568\t4100\t1.9098
4200\t2.6059\t4200\t2.1285\t4200\t1.8828
4300\t2.576\t4300\t2.0961\t4300\t1.8508
4400\t2.5431\t4400\t2.0595\t4400\t1.814
4500\t2.5073\t4500\t2.0191\t4500\t1.7724
4600\t2.4689\t4600\t1.9751\t4600\t1.7263
4700\t2.4283\t4700\t1.9277\t4700\t1.6758
4800\t2.3856\t4800\t1.8772\t4800\t1.6212
4900\t2.3413\t4900\t1.8239\t4900\t1.5627
5000\t2.2957\t5000\t1.7682\t5000\t1.5007
5100\t2.2492\t5100\t1.7106\t5100\t1.4356
5200\t2.2022\t5200\t1.6514\t5200\t1.3677
5300\t2.1551\t5300\t1.5913\t5300\t1.2977
5400\t2.1086\t5400\t1.5308\t5400\t1.226
5500\t2.063\t5500\t1.4705\t5500\t1.1534
5600\t2.019\t5600\t1.4113\t5600\t1.0806
5700\t1.9772\t5700\t1.3539\t5700\t1.0084
5800\t1.9383\t5800\t1.2994\t5800\t0.93782
5900\t1.903\t5900\t1.2489\t5900\t0.87002
6000\t1.8722\t6000\t1.2038\t6000\t0.8064
6100\t1.8468\t6100\t1.166\t6100\t0.74879
6200\t1.828\t6200\t1.1381\t6200\t0.69979
"""

# Read text blocks into DataFrames
df_t = pd.read_csv(StringIO(PN_TXT), sep='\t')
df_t.columns = ['f','A10','f2','A15','f3','A20']
df_p = pd.read_csv(StringIO(POR_TXT), sep='\t')
df_p.columns = ['f','A90','f2','A80','f3','A70']
df_c = pd.read_csv(StringIO(CELL_TXT), sep='\t')
df_c.columns = ['f','A3','f2','A5','f3','A7']

# Baseline configuration
BASE = {"t": 15.0, "phi": 0.80, "d": 5.0}

def interp_at_frequency(freq):
    A10 = np.interp(freq, df_t['f'], df_t['A10'])
    A15 = np.interp(freq, df_t['f'], df_t['A15'])
    A20 = np.interp(freq, df_t['f'], df_t['A20'])
    A90 = np.interp(freq, df_p['f'], df_p['A90'])
    A80 = np.interp(freq, df_p['f'], df_p['A80'])
    A70 = np.interp(freq, df_p['f'], df_p['A70'])
    A3  = np.interp(freq, df_c['f'], df_c['A3'])
    A5  = np.interp(freq, df_c['f'], df_c['A5'])
    A7  = np.interp(freq, df_c['f'], df_c['A7'])
    return (A10, A15, A20, A90, A80, A70, A3, A5, A7)

# Local linear model coefficients at a given frequency
def local_coeffs(freq):
    A10,A15,A20,A90,A80,A70,A3,A5,A7 = interp_at_frequency(freq)
    kt,_ = np.polyfit(np.array([10,15,20]), np.array([A10,A15,A20]), 1)
    kp,_ = np.polyfit(np.array([1-0.90,1-0.80,1-0.70]), np.array([A90,A80,A70]), 1)
    kc,_ = np.polyfit(np.array([1/3,1/5,1/7.5]), np.array([A3,A5,A7]), 1)
    A_base = A15
    return A_base, kt, kp, kc

def predict_A(freq, t, phi, d):
    A_base, kt, kp, kc = local_coeffs(freq)
    delta = kt*(t-BASE['t']) + kp*((1-phi)-(1-BASE['phi'])) + kc*((1/d)-(1/BASE['d']))
    return float(A_base + delta)

def solve_for(var, freq, targetA, t=None, phi=None, d=None):
    A_base, kt, kp, kc = local_coeffs(freq)
    if var == 't':
        if abs(kt) < 1e-6: return None
        delta = targetA - A_base - kp*((1-phi)-(1-BASE['phi'])) - kc*((1/d)-(1/BASE['d']))
        return float(BASE['t'] + delta/kt)
    if var == 'phi':
        if abs(kp) < 1e-6: return None
        delta = targetA - A_base - kt*(t-BASE['t']) - kc*((1/d)-(1/BASE['d']))
        s = (1-BASE['phi']) + delta/kp
        return float(1 - s)
    if var == 'd':
        if abs(kc) < 1e-6: return None
        delta = targetA - A_base - kt*(t-BASE['t']) - kp*((1-phi)-(1-BASE['phi']))
        u = (1/BASE['d']) + delta/kc
        return float(1/u) if u>0 else None

# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("🎯 Target & Band")
    A_target = st.slider("Target attenuation (dB)", 0.0, 15.0, 5.0, 0.1)
    f_min, f_max = st.select_slider("Design frequency band (Hz)",
                                    options=list(range(1000, 6201, 100)),
                                    value=(1000, 6000))
    design_freq = st.slider("Anchor frequency for solving (Hz)", 1000, 6200, 1000, step=100)
    st.caption("Parameters are solved at the anchor frequency using local linear fits.")

    st.header("🔧 Fix / Free Parameters")
    fix_t   = st.checkbox("Fix thickness (mm)", True)
    t_val   = st.number_input("Thickness t (mm)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)

    fix_phi = st.checkbox("Fix porosity (%)", True)
    phi_pct = st.number_input("Porosity φ (%)", min_value=10.0, max_value=95.0, value=70.0, step=1.0)

    fix_d   = st.checkbox("Fix cell size (mm)", True)
    d_val   = st.number_input("Cell size d (mm)", min_value=0.2, max_value=20.0, value=3.0, step=0.1)

phi_val = phi_pct/100.0

# Decide which variable to solve for
free_vars = [v for v,fix in zip(['t','phi','d'], [fix_t, fix_phi, fix_d]) if not fix]
if len(free_vars) == 0:
    st.warning("Uncheck one parameter in the sidebar to let the tool solve for it.")
elif len(free_vars) > 1:
    st.warning("Please leave exactly one parameter free (solve for one variable at a time).")
else:
    var = free_vars[0]
    sol = None
    if var == 't':
        sol = solve_for('t', design_freq, A_target, phi=phi_val, d=d_val)
        t_sol, phi_sol, d_sol = sol, phi_val, d_val
    elif var == 'phi':
        sol = solve_for('phi', design_freq, A_target, t=t_val, d=d_val)
        t_sol, phi_sol, d_sol = t_val, sol, d_val
    elif var == 'd':
        sol = solve_for('d', design_freq, A_target, t=t_val, phi=phi_val)
        t_sol, phi_sol, d_sol = t_val, phi_val, sol

    if sol is None:
        st.error("Could not find a physical solution at the selected anchor frequency.")
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Solved Parameters")
            st.write(f"**Solved variable:** `{var}` at **{design_freq} Hz**")
            st.metric("Thickness t (mm)", f"{t_sol:0.2f}")
            st.metric("Porosity φ (%)", f"{phi_sol*100:0.2f}")
            st.metric("Cell size d (mm)", f"{d_sol:0.3f}")

            if var=='phi' and (phi_sol<=0 or phi_sol>=1):
                st.warning("Porosity out of [0,1] range — target may be unattainable with current constraints.")
            if var=='d' and (d_sol is None or d_sol<=0):
                st.warning("Computed cell size is non-physical — adjust target or fixed values.")

        freqs = np.arange(f_min, f_max+1, 100)
        A_pred = [predict_A(f, t_sol, phi_sol, d_sol) for f in freqs]

        with col2:
            st.subheader("Predicted Attenuation vs Frequency")
            fig, ax = plt.subplots(figsize=(6.5,4.2))
            ax.plot(freqs, A_pred, linewidth=2)
            ax.axhline(A_target, linestyle='--', linewidth=1)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Attenuation (dB)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Local Coefficients at Anchor Frequency")
        A_base, kt, kp, kc = local_coeffs(design_freq)
        st.write(f"**A_base(15mm, 80%, 5mm)** = {A_base:0.3f} dB")
        st.write(f"**k_t (dB/mm)** = {kt:0.4f}")
        st.write(f"**k_p (dB per unit solids fraction)** = {kp:0.2f}")
        st.write(f"**k_c (dB per 1/mm)** = {kc:0.2f}")
        st.caption("Model: A ≈ A_base + k_t·(t−15) + k_p·((1−φ)−0.2) + k_c·((1/d)−0.2). Coefficients are fitted from your datasets at the chosen frequency.")
