import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


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

def flow_to_m3h(flow, unit):
    return flow_to_m3s(flow, unit) * 3600.0

def m3h_to_flow(val_m3h, unit):
    return m3s_to_flow(val_m3h / 3600.0, unit)

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
    # dp = a * Q^n
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


# ================================
# UI
# ================================
st.set_page_config(page_title="Kv/Cv Toolkit", layout="centered")
st.title("Kv / Cv Toolkit (Web)")

tab1, tab2 = st.tabs(["Converters", "Kv/Cv Tool"])

# ---------- Converters ----------
with tab1:
    st.subheader("Flow Unit Converter")
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        flow_val = st.number_input("Value", value=100.0, key="flow_val")
    with c2:
        flow_from = st.selectbox("From", FLOW_UNITS, index=4, key="flow_from")
    with c3:
        flow_to = st.selectbox("To", FLOW_UNITS, index=0, key="flow_to")

    if st.button("Convert Flow"):
        try:
            out = m3s_to_flow(flow_to_m3s(flow_val, flow_from), flow_to)
            st.success(f"{flow_val:g} {flow_from} = {out:.6g} {flow_to}")
        except Exception as e:
            st.error(str(e))

    st.divider()

    st.subheader("Pressure Unit Converter")
    p1, p2, p3 = st.columns([2, 2, 2])

    with p1:
        p_val = st.number_input("Value ", value=35.0, key="p_val")
    with p2:
        p_from = st.selectbox("From ", DP_UNITS, index=1, key="p_from")
    with p3:
        p_to = st.selectbox("To ", DP_UNITS, index=2, key="p_to")

    if st.button("Convert Pressure"):
        try:
            out = pa_to_dp(dp_to_pa(p_val, p_from), p_to)
            st.success(f"{p_val:g} {p_from} = {out:.6g} {p_to}")
        except Exception as e:
            st.error(str(e))


# ---------- Kv/Cv Tool ----------
with tab2:
    st.subheader("Kv/Cv from points + Fit (ΔP = a·Q^n)")

    sg = st.number_input("Specific Gravity (SG)", value=1.0, min_value=0.0001, format="%.4f")

    colu1, colu2 = st.columns(2)
    with colu1:
        flow_unit = st.selectbox("Flow unit (input)", FLOW_UNITS, index=4)
    with colu2:
        dp_unit = st.selectbox("ΔP unit (input)", DP_UNITS, index=1)

    default_text = "120, 35\n150, 45\n180, 62"
    pts_text = st.text_area("Paste points (Q, ΔP), one per line", value=default_text, height=140)

    if st.button("Calculate / Fit"):
        try:
            pts = parse_points(pts_text)
            Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pts]
            dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pts]

            kvs = [kv_from_point(Q, dp, sg) for Q, dp in zip(Q_m3h, dp_bar)]
            cvs = [kv / 0.865 for kv in kvs]

            st.session_state["last_data"] = {"Q_m3h": Q_m3h, "dp_bar": dp_bar, "pts": pts, "flow_unit": flow_unit, "dp_unit": dp_unit, "sg": sg}
            st.session_state["last_fit"] = None

            st.markdown("### Point-wise results")
            for i, ((q_in, dp_in), kv, cv) in enumerate(zip(pts, kvs, cvs), 1):
                st.write(f"{i}. Q={q_in:g} {flow_unit}, ΔP={dp_in:g} {dp_unit} → Kv={kv:.4f}, Cv={cv:.4f}")

            if len(pts) >= 2:
                a, n_exp = fit_power_law(Q_m3h, dp_bar)
                K0 = math.sqrt(sg / a)
                m_exp = 1 - n_exp / 2
                st.session_state["last_fit"] = {"a": a, "n": n_exp, "K0": K0, "m": m_exp}

                st.markdown("### Fit model")
                st.write(f"ΔP = {a:.6g} · Q^{n_exp:.6g}   (Q in m³/h, ΔP in bar)")
                st.write(f"Kv(Q) = {K0:.6g} · Q^{m_exp:.6g}")
                st.write("Cv(Q) = Kv(Q) / 0.865")
            else:
                st.info("Fit not available (need ≥2 points). Single-point Kv/Cv computed.")

        except Exception as e:
            st.error(str(e))

    if st.button("Plot PQ (points + fit)"):
        try:
            if "last_data" not in st.session_state:
                st.error("No data yet. Click Calculate / Fit first.")
            else:
                data = st.session_state["last_data"]
                fit = st.session_state.get("last_fit", None)

                Q_m3h = data["Q_m3h"]
                dp_bar = data["dp_bar"]
                flow_unit = data["flow_unit"]
                dp_unit = data["dp_unit"]

                # plot axes in selected units
                Q_axis = [m3h_to_flow(q, flow_unit) for q in Q_m3h]
                dp_axis = [bar_to_dp(dpb, dp_unit) for dpb in dp_bar]

                fig, ax = plt.subplots()
                ax.plot(Q_axis, dp_axis, "o", label="Measured points")

                if fit and len(Q_m3h) >= 2:
                    a = fit["a"]; n = fit["n"]
                    qmin, qmax = min(Q_m3h), max(Q_m3h)
                    xs = np.logspace(math.log10(qmin), math.log10(qmax), 180)
                    ys = a * (xs ** n)
                    ax.plot([m3h_to_flow(x, flow_unit) for x in xs],
                            [bar_to_dp(y, dp_unit) for y in ys],
                            "-", label="Fitted curve")

                ax.set_title("PQ Curve")
                ax.set_xlabel(f"Flow [{flow_unit}]")
                ax.set_ylabel(f"ΔP [{dp_unit}]")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(str(e))