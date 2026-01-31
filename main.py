# main.py
# Streamlit Web App: Flow/Pressure Converter + Kv/Cv (1 point OK) + Fit (>=2) + PQ Plot
# Deployable on Streamlit Community Cloud (public URL)

import math
import numpy as np
import pandas as pd
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
# Kv/Cv logic
# ================================
def parse_points(text):
    """
    Each line: Q, dP  (comma or space separated)
    """
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
    # dp = a * Q^n, with Q in m3/h and dp in bar
    if len(Qs_m3h) < 2:
        raise ValueError("Need at least 2 points to fit.")
    xs, ys = [], []
    for q, dp in zip(Qs_m3h, dPs_bar):
        if q <= 0 or dp <= 0:
            raise ValueError("All Q and ΔP must be > 0 for fitting.")
        xs.append(math.log(q))
        ys.append(math.log(dp))

    npts = len(xs)
    xbar = sum(xs) / npts
    ybar = sum(ys) / npts
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("All Q values identical; cannot fit exponent.")
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    n_exp = sxy / sxx
    a = math.exp(ybar - n_exp * xbar)
    return a, n_exp

def build_fit_curve(a, n_exp, qmin, qmax, steps=160):
    # log-spaced curve in internal units (m3/h)
    if qmin <= 0 or qmax <= 0 or qmin == qmax:
        return []
    lmin = math.log10(qmin)
    lmax = math.log10(qmax)
    qs = [10 ** (lmin + (lmax - lmin) * i / (steps - 1)) for i in range(steps)]
    dps = [a * (q ** n_exp) for q in qs]
    return list(zip(qs, dps))

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Kv/Cv Toolkit", layout="wide")

st.title("Kv / Cv Toolkit")
st.caption("Flow & Pressure converters + Kv/Cv from points (1 point OK) + Fit (>=2 points) + PQ plot")

tab1, tab2 = st.tabs(["Converters", "Kv/Cv Tool"])

# ---------- Converters ----------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Flow Unit Converter")
        fv = st.text_input("Value", "100", key="flow_val")
        f_from = st.selectbox("From unit", FLOW_UNITS, index=4, key="flow_from")
        f_to = st.selectbox("To unit", FLOW_UNITS, index=0, key="flow_to")

        if st.button("Convert Flow"):
            try:
                v = float(fv)
                out = m3s_to_flow(flow_to_m3s(v, f_from), f_to)
                st.success(f"{v:g} {f_from}  =  **{out:.6g} {f_to}**")
            except Exception as e:
                st.error(str(e))

    with c2:
        st.subheader("Pressure Unit Converter")
        pv = st.text_input("Value ", "35", key="p_val")
        p_from = st.selectbox("From unit ", DP_UNITS, index=1, key="p_from")
        p_to = st.selectbox("To unit ", DP_UNITS, index=2, key="p_to")

        if st.button("Convert Pressure"):
            try:
                v = float(pv)
                out = pa_to_dp(dp_to_pa(v, p_from), p_to)
                st.success(f"{v:g} {p_from}  =  **{out:.6g} {p_to}**")
            except Exception as e:
                st.error(str(e))

# ---------- Kv/Cv Tool ----------
with tab2:
    left, right = st.columns([1.05, 1.0])

    with left:
        st.subheader("Inputs")
        sg = st.number_input("Specific Gravity (SG)", min_value=0.0001, value=1.0, step=0.01)

        flow_unit = st.selectbox("Flow unit for input points", FLOW_UNITS, index=4)
        dp_unit = st.selectbox("ΔP unit for input points", DP_UNITS, index=1)

        pts_text = st.text_area(
            "Paste points (Q, ΔP), one per line",
            value="120, 35\n150, 45\n180, 62\n",
            height=180
        )

        do_calc = st.button("Calculate / Fit", type="primary")

        if do_calc:
            try:
                pts = parse_points(pts_text)

                Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pts]
                dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pts]

                kvs = [kv_from_point(Q, dp, sg) for Q, dp in zip(Q_m3h, dp_bar)]
                cvs = [kv / 0.865 for kv in kvs]

                # store in session
                st.session_state["last_pts"] = pts
                st.session_state["last_internal"] = {"Q_m3h": Q_m3h, "dp_bar": dp_bar, "sg": sg}
                st.session_state["last_units"] = {"flow_unit": flow_unit, "dp_unit": dp_unit}

                fit = None
                if len(pts) >= 2:
                    a, n_exp = fit_power_law(Q_m3h, dp_bar)
                    K0 = math.sqrt(sg / a)
                    m_exp = 1 - n_exp / 2
                    fit = {"a": a, "n": n_exp, "K0": K0, "m": m_exp}
                    st.session_state["last_fit"] = fit
                else:
                    st.session_state["last_fit"] = None

                # table
                df = pd.DataFrame({
                    f"Q [{flow_unit}]": [p[0] for p in pts],
                    f"ΔP [{dp_unit}]": [p[1] for p in pts],
                    "Q [m³/h]": Q_m3h,
                    "ΔP [bar]": dp_bar,
                    "Kv": kvs,
                    "Cv": cvs
                })

                st.session_state["last_df"] = df
                st.success("Calculated successfully.")

            except Exception as e:
                st.error(str(e))

        st.divider()
        st.subheader("Results")

        df = st.session_state.get("last_df", None)
        if df is not None:
            st.dataframe(df, use_container_width=True)

            fit = st.session_state.get("last_fit", None)
            if fit:
                st.markdown("**Fitted model (internal units):**")
                st.code(
                    f"ΔP = a · Q^n   (Q in m³/h, ΔP in bar)\n"
                    f"a = {fit['a']:.6g}\n"
                    f"n = {fit['n']:.6g}\n\n"
                    f"Kv(Q) = K0 · Q^m\n"
                    f"K0 = {fit['K0']:.6g}\n"
                    f"m  = {fit['m']:.6g}\n\n"
                    f"Cv(Q) = Kv(Q) / 0.865"
                )
            else:
                st.info("Fit not available (need ≥2 points). Single-point Kv/Cv is already computed.")

    with right:
        st.subheader("PQ Plot (Points + Fit)")

        internal = st.session_state.get("last_internal", None)
        units = st.session_state.get("last_units", None)
        fit = st.session_state.get("last_fit", None)

        if internal is None or units is None:
            st.info("Click **Calculate / Fit** first to generate the plot.")
        else:
            Q_m3h = internal["Q_m3h"]
            dp_bar = internal["dp_bar"]

            # plot in user-selected units
            fu = units["flow_unit"]
            pu = units["dp_unit"]

            Q_axis = [m3h_to_flow(q, fu) for q in Q_m3h]
            dp_axis = [bar_to_dp(dpb, pu) for dpb in dp_bar]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=Q_axis, y=dp_axis,
                mode="markers", name="Measured points"
            ))

            if fit and len(Q_m3h) >= 2:
                curve_internal = build_fit_curve(fit["a"], fit["n"], min(Q_m3h), max(Q_m3h), steps=180)
                if curve_internal:
                    curve_Q = [m3h_to_flow(q, fu) for q, _ in curve_internal]
                    curve_dp = [bar_to_dp(dpb, pu) for _, dpb in curve_internal]
                    fig.add_trace(go.Scatter(
                        x=curve_Q, y=curve_dp,
                        mode="lines", name="Fitted curve"
                    ))

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                height=520,
                xaxis_title=f"Flow [{fu}]",
                yaxis_title=f"ΔP [{pu}]",
                title="PQ Curve"
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            st.plotly_chart(fig, use_container_width=True)
