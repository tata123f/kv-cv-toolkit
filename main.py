# main.py
# Streamlit Web App (Android/Windows/macOS friendly)
# - Flow unit conversion
# - Pressure unit conversion
# - Kv/Cv from points (>=1)
# - Fit ΔP = a·Q^n (>=2)
# - Plot points + fitted curve (Plotly, no matplotlib)

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ================================
# Units
# ================================
FLOW_UNITS = ["m3/h", "m3/s", "m3/min", "L/h", "L/min", "L/s", "gpm (US)", "gpm (Imp)", "cfm"]
DP_UNITS = ["Pa", "kPa", "bar", "MPa", "psi"]

def flow_to_m3s(flow, unit):
    if unit == "m3/s":   return flow
    if unit == "m3/min": return flow / 60.0
    if unit == "m3/h":   return flow / 3600.0
    if unit == "L/s":    return flow / 1000.0
    if unit == "L/min":  return flow / 1000.0 / 60.0
    if unit == "L/h":    return flow / 1000.0 / 3600.0
    if unit == "gpm (US)":  return flow * 3.785411784e-3 / 60.0
    if unit == "gpm (Imp)": return flow * 4.54609e-3 / 60.0
    if unit == "cfm":    return flow * 0.028316846592 / 60.0
    raise ValueError(f"Unsupported flow unit: {unit}")

def m3s_to_flow(val_m3s, unit):
    if unit == "m3/s":   return val_m3s
    if unit == "m3/min": return val_m3s * 60.0
    if unit == "m3/h":   return val_m3s * 3600.0
    if unit == "L/s":    return val_m3s * 1000.0
    if unit == "L/min":  return val_m3s * 1000.0 * 60.0
    if unit == "L/h":    return val_m3s * 1000.0 * 3600.0
    if unit == "gpm (US)":  return val_m3s * 60.0 / 3.785411784e-3
    if unit == "gpm (Imp)": return val_m3s * 60.0 / 4.54609e-3
    if unit == "cfm":    return val_m3s * 60.0 / 0.028316846592
    raise ValueError(f"Unsupported flow unit: {unit}")

def flow_to_m3h(flow, unit):
    return flow_to_m3s(flow, unit) * 3600.0

def m3h_to_flow(val_m3h, unit):
    return m3s_to_flow(val_m3h / 3600.0, unit)

def dp_to_pa(dp, unit):
    if unit == "Pa":  return dp
    if unit == "kPa": return dp * 1_000.0
    if unit == "bar": return dp * 100_000.0
    if unit == "MPa": return dp * 1_000_000.0
    if unit == "psi": return dp * 6894.757293168
    raise ValueError(f"Unsupported pressure unit: {unit}")

def pa_to_dp(pa, unit):
    if unit == "Pa":  return pa
    if unit == "kPa": return pa / 1_000.0
    if unit == "bar": return pa / 100_000.0
    if unit == "MPa": return pa / 1_000_000.0
    if unit == "psi": return pa / 6894.757293168
    raise ValueError(f"Unsupported pressure unit: {unit}")

def dp_to_bar(dp, unit):
    return pa_to_dp(dp_to_pa(dp, unit), "bar")

def bar_to_dp(bar, unit):
    return pa_to_dp(dp_to_pa(bar, "bar"), unit)

# ================================
# Kv/Cv + fitting
# ================================
def parse_points(text):
    pts = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = s.replace(",", " ")
        parts = [p for p in s.split() if p]
        if len(parts) < 2:
            raise ValueError(f"Bad line: '{ln}' (need Q and ΔP)")
        q = float(parts[0])
        dp = float(parts[1])
        pts.append((q, dp))
    if len(pts) < 1:
        raise ValueError("Need at least 1 point.")
    return pts

def kv_from_point(Q_m3h, dp_bar, sg):
    if dp_bar <= 0:
        raise ValueError("ΔP must be > 0")
    if sg <= 0:
        raise ValueError("SG must be > 0")
    return Q_m3h * math.sqrt(sg / dp_bar)

def fit_power_law(Qs_m3h, dPs_bar):
    # dp = a * Q^n  => ln(dp)=ln(a)+n ln(Q)
    if len(Qs_m3h) < 2:
        raise ValueError("Need at least 2 points to fit.")
    x = []
    y = []
    for q, dp in zip(Qs_m3h, dPs_bar):
        if q <= 0 or dp <= 0:
            raise ValueError("All Q and ΔP must be > 0 for fitting.")
        x.append(math.log(q))
        y.append(math.log(dp))
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = np.polyfit(x, y, 1)[0]
    ln_a = np.polyfit(x, y, 1)[1]
    a = math.exp(ln_a)
    return a, n

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Kv / Cv Toolkit", layout="wide")

st.title("Kv / Cv Toolkit (Web)")
st.caption("Converters + Kv/Cv + Fit + PQ Plot — works on Windows/macOS/Android browser. No matplotlib needed.")

tab1, tab2 = st.tabs(["Converters", "Kv/Cv Tool"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Flow Unit Converter")
        fv = st.number_input("Value", value=100.0, step=1.0, key="flow_val")
        f_from = st.selectbox("From", FLOW_UNITS, index=4, key="flow_from")  # L/min
        f_to = st.selectbox("To", FLOW_UNITS, index=0, key="flow_to")        # m3/h
        if st.button("Convert Flow"):
            try:
                out = m3s_to_flow(flow_to_m3s(fv, f_from), f_to)
                st.success(f"{out:.6g} {f_to}")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.subheader("Pressure Unit Converter")
        pv = st.number_input("Value ", value=35.0, step=1.0, key="p_val")
        p_from = st.selectbox("From ", DP_UNITS, index=1, key="p_from")  # kPa
        p_to = st.selectbox("To ", DP_UNITS, index=2, key="p_to")        # bar
        if st.button("Convert Pressure"):
            try:
                out = pa_to_dp(dp_to_pa(pv, p_from), p_to)
                st.success(f"{out:.6g} {p_to}")
            except Exception as e:
                st.error(str(e))

with tab2:
    st.subheader("Kv/Cv from points + Fit + PQ Plot")

    sg = st.number_input("Specific Gravity (SG)", value=1.0, min_value=0.0001, step=0.01)
    c3, c4 = st.columns(2)
    with c3:
        flow_unit = st.selectbox("Flow unit for input points (Q)", FLOW_UNITS, index=4)
    with c4:
        dp_unit = st.selectbox("ΔP unit for input points", DP_UNITS, index=1)

    pts_text = st.text_area(
        "Paste points (Q, ΔP), one per line (comma or space). Example:",
        value="120, 35\n150, 45\n180, 62\n",
        height=140
    )

    if "calc_result" not in st.session_state:
        st.session_state.calc_result = None
    if "fit_result" not in st.session_state:
        st.session_state.fit_result = None
    if "last_data" not in st.session_state:
        st.session_state.last_data = None

    colA, colB = st.columns([1, 1])
    with colA:
        do_calc = st.button("Calculate / Fit")
    with colB:
        do_plot = st.button("Plot PQ")

    if do_calc:
        try:
            pts = parse_points(pts_text)
            Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pts]
            dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pts]

            kvs = [kv_from_point(Q, dp, sg) for Q, dp in zip(Q_m3h, dp_bar)]
            cvs = [kv / 0.865 for kv in kvs]

            lines = []
            lines.append(f"SG = {sg:.4f}")
            lines.append(f"Input units: Q in {flow_unit}, ΔP in {dp_unit}")
            lines.append("Internal units: Q in m³/h, ΔP in bar\n")
            lines.append("Point-wise Kv/Cv:")
            for i, ((q_in, dp_in), Qint, dpint, kv, cv) in enumerate(zip(pts, Q_m3h, dp_bar, kvs, cvs), 1):
                lines.append(
                    f"{i}. Q={q_in:g} {flow_unit} (= {Qint:.6g} m³/h)   "
                    f"ΔP={dp_in:g} {dp_unit} (= {dpint:.6g} bar)  ->  Kv={kv:.4f}, Cv={cv:.4f}"
                )

            fit = None
            if len(pts) >= 2:
                a, n = fit_power_law(Q_m3h, dp_bar)
                K0 = math.sqrt(sg / a)
                m = 1 - n / 2
                fit = {"a": a, "n": n, "K0": K0, "m": m}
                lines.append("\nFit model (Q in m³/h, ΔP in bar):")
                lines.append(f"ΔP = {a:.6g} · Q^{n:.6g}")
                lines.append("Derived functions:")
                lines.append(f"Kv(Q) = {K0:.6g} · Q^{m:.6g}")
                lines.append("Cv(Q) = Kv(Q) / 0.865")
            else:
                lines.append("\nFit not available (need ≥2 points). Single-point Kv/Cv computed.")

            st.session_state.calc_result = "\n".join(lines)
            st.session_state.fit_result = fit
            st.session_state.last_data = {
                "pts": pts,
                "Q_m3h": Q_m3h,
                "dp_bar": dp_bar,
                "flow_unit": flow_unit,
                "dp_unit": dp_unit,
            }

            st.success("Calculated.")
        except Exception as e:
            st.error(str(e))

    if st.session_state.calc_result:
        st.text_area("Results", value=st.session_state.calc_result, height=260)

    if do_plot:
        try:
            if not st.session_state.last_data:
                raise ValueError("No data. Click Calculate / Fit first.")

            data = st.session_state.last_data
            Q_m3h = data["Q_m3h"]
            dp_bar = data["dp_bar"]
            flow_unit = data["flow_unit"]
            dp_unit = data["dp_unit"]

            Q_axis = [m3h_to_flow(q, flow_unit) for q in Q_m3h]
            dp_axis = [bar_to_dp(dpb, dp_unit) for dpb in dp_bar]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=Q_axis, y=dp_axis,
                mode="markers",
                name="Measured points"
            ))

            fit = st.session_state.fit_result
            if fit and len(Q_m3h) >= 2:
                a = fit["a"]
                n = fit["n"]
                qmin, qmax = min(Q_m3h), max(Q_m3h)
                if qmin <= 0 or qmax <= 0 or qmin == qmax:
                    raise ValueError("Cannot plot fit curve: invalid Q range.")

                steps = 160
                lmin = math.log10(qmin)
                lmax = math.log10(qmax)
                curve_Q_m3h = [10 ** (lmin + (lmax - lmin) * i / (steps - 1)) for i in range(steps)]
                curve_dp_bar = [a * (q ** n) for q in curve_Q_m3h]

                curve_Q = [m3h_to_flow(q, flow_unit) for q in curve_Q_m3h]
                curve_dp = [bar_to_dp(dpb, dp_unit) for dpb in curve_dp_bar]

                fig.add_trace(go.Scatter(
                    x=curve_Q, y=curve_dp,
                    mode="lines",
                    name="Fitted curve"
                ))

            fig.update_layout(
                title="PQ Curve (Measured + Fit)",
                xaxis_title=f"Flow [{flow_unit}]",
                yaxis_title=f"ΔP [{dp_unit}]",
                template="plotly_white",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))
