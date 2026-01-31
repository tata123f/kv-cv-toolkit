import math
import streamlit as st

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
    if dp_bar <= 0: raise ValueError("ΔP must be > 0")
    if sg <= 0: raise ValueError("SG must be > 0")
    return Q_m3h * math.sqrt(sg / dp_bar)

def fit_power_law(Qs_m3h, dPs_bar):
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
# Pure SVG Plot (no plotly/matplotlib)
# ================================
def svg_plot(points, curve=None, width=760, height=420, title="PQ Curve", xlabel="Flow", ylabel="ΔP"):
    if not points:
        return "<div>No data</div>"

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if curve:
        xs += [p[0] for p in curve]
        ys += [p[1] for p in curve]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmin *= 0.9; xmax *= 1.1 if xmax != 0 else 1
    if ymin == ymax: ymin *= 0.9; ymax *= 1.1 if ymax != 0 else 1

    # padding
    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.10
    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad

    ml, mr, mt, mb = 70, 20, 40, 60
    px0, py0 = ml, mt
    px1, py1 = width - mr, height - mb

    def xmap(x): return px0 + (x - xmin) / (xmax - xmin) * (px1 - px0)
    def ymap(y): return py1 - (y - ymin) / (ymax - ymin) * (py1 - py0)

    # build svg
    svg = []
    svg.append(f'<svg width="{width}" height="{height}" style="background:white;border:1px solid #ddd;">')
    svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#111">{title}</text>')
    svg.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="white" stroke="#ccc"/>')
    svg.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="12" fill="#222">{xlabel}</text>')
    svg.append(f'<text x="18" y="{height/2}" text-anchor="middle" font-size="12" fill="#222" transform="rotate(-90 18 {height/2})">{ylabel}</text>')

    # grid lines (simple 5 ticks)
    for i in range(6):
        tx = px0 + (px1 - px0) * i / 5
        ty = py0 + (py1 - py0) * i / 5
        svg.append(f'<line x1="{tx}" y1="{py0}" x2="{tx}" y2="{py1}" stroke="#f2f2f2"/>')
        svg.append(f'<line x1="{px0}" y1="{ty}" x2="{px1}" y2="{ty}" stroke="#f2f2f2"/>')

    # curve
    if curve and len(curve) >= 2:
        pts = " ".join([f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in curve])
        svg.append(f'<polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="2.5"/>')

    # points
    for x, y in points:
        cx, cy = xmap(x), ymap(y)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.5" fill="#111"/>')

    svg.append('</svg>')
    return "".join(svg)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Kv/Cv Toolkit", layout="wide")

st.title("Kv / Cv Toolkit (Web)")
st.caption("Converters + Kv/Cv from points (1 point OK) + Fit (>=2 points) + PQ plot (no external plotting libs)")

tab1, tab2 = st.tabs(["Converters", "Kv/Cv Tool"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Flow Unit Converter")
        fv = st.number_input("Value", value=100.0, key="fv")
        ffrom = st.selectbox("From unit", FLOW_UNITS, index=4, key="ffrom")
        fto = st.selectbox("To unit", FLOW_UNITS, index=0, key="fto")
        if st.button("Convert Flow"):
            out = m3s_to_flow(flow_to_m3s(fv, ffrom), fto)
            st.success(f"{fv:g} {ffrom} = {out:.6g} {fto}")

    with c2:
        st.subheader("Pressure Unit Converter")
        pv = st.number_input("Value ", value=35.0, key="pv")
        pfrom = st.selectbox("From unit ", DP_UNITS, index=1, key="pfrom")
        pto = st.selectbox("To unit ", DP_UNITS, index=2, key="pto")
        if st.button("Convert Pressure"):
            out = pa_to_dp(dp_to_pa(pv, pfrom), pto)
            st.success(f"{pv:g} {pfrom} = {out:.6g} {pto}")

with tab2:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Inputs")
        sg = st.number_input("Specific Gravity (SG)", value=1.0, min_value=0.0001)
        flow_unit = st.selectbox("Flow unit (input)", FLOW_UNITS, index=4)
        dp_unit = st.selectbox("ΔP unit (input)", DP_UNITS, index=1)

        pts_text = st.text_area(
            "Paste points (Q, ΔP) — one per line",
            value="120, 35\n150, 45\n180, 62\n",
            height=160,
        )

        calc = st.button("Calculate / Fit")
        plot = st.button("Plot PQ")

    # session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "fit" not in st.session_state:
        st.session_state.fit = None

    if calc:
        try:
            pts = parse_points(pts_text)
            Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pts]
            dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pts]

            kvs = [kv_from_point(Q, dp, sg) for Q, dp in zip(Q_m3h, dp_bar)]
            cvs = [kv / 0.865 for kv in kvs]

            fit = None
            if len(pts) >= 2:
                a, n = fit_power_law(Q_m3h, dp_bar)
                K0 = math.sqrt(sg / a)
                m = 1 - n / 2
                fit = {"a": a, "n": n, "K0": K0, "m": m}

            st.session_state.data = {
                "pts_raw": pts,
                "Q_m3h": Q_m3h,
                "dp_bar": dp_bar,
                "flow_unit": flow_unit,
                "dp_unit": dp_unit,
                "sg": sg,
                "kvs": kvs,
                "cvs": cvs,
            }
            st.session_state.fit = fit

        except Exception as e:
            st.error(str(e))

    with right:
        st.subheader("Results")
        data = st.session_state.data
        fit = st.session_state.fit

        if not data:
            st.info("Click **Calculate / Fit** first.")
        else:
            lines = []
            lines.append(f"SG = {data['sg']:.4f}")
            lines.append(f"Input units: Q in {data['flow_unit']}, ΔP in {data['dp_unit']}")
            lines.append("Internal units: Q in m³/h, ΔP in bar\n")

            for i, ((q_in, dp_in), Qint, dpint, kv, cv) in enumerate(
                zip(data["pts_raw"], data["Q_m3h"], data["dp_bar"], data["kvs"], data["cvs"]), 1
            ):
                lines.append(
                    f"{i}. Q={q_in:g} {data['flow_unit']} (= {Qint:.6g} m³/h)   "
                    f"ΔP={dp_in:g} {data['dp_unit']} (= {dpint:.6g} bar)  ->  Kv={kv:.4f}, Cv={cv:.4f}"
                )

            if fit:
                lines.append("\nFit model (Q in m³/h, ΔP in bar):")
                lines.append(f"ΔP = {fit['a']:.6g} · Q^{fit['n']:.6g}")
                lines.append("Derived:")
                lines.append(f"Kv(Q) = {fit['K0']:.6g} · Q^{fit['m']:.6g}")
                lines.append("Cv(Q) = Kv(Q) / 0.865")
            else:
                lines.append("\nFit not available (need ≥2 points).")

            st.code("\n".join(lines), language="text")

        st.subheader("PQ Plot")
        if plot:
            if not data:
                st.warning("No data. Click Calculate / Fit first.")
            else:
                # plot in displayed units
                Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in data["Q_m3h"]]
                dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in data["dp_bar"]]
                points = list(zip(Q_axis, dp_axis))

                curve = None
                if fit and len(data["Q_m3h"]) >= 2:
                    a = fit["a"]; n = fit["n"]
                    qmin, qmax = min(data["Q_m3h"]), max(data["Q_m3h"])
                    if qmin > 0 and qmax > 0 and qmin != qmax:
                        steps = 160
                        lmin, lmax = math.log10(qmin), math.log10(qmax)
                        curve_Q_m3h = [10 ** (lmin + (lmax - lmin) * i / (steps - 1)) for i in range(steps)]
                        curve_dp_bar = [a * (q ** n) for q in curve_Q_m3h]
                        curve_Q = [m3h_to_flow(q, data["flow_unit"]) for q in curve_Q_m3h]
                        curve_dp = [bar_to_dp(dpb, data["dp_unit"]) for dpb in curve_dp_bar]
                        curve = list(zip(curve_Q, curve_dp))

                svg = svg_plot(
                    points=points,
                    curve=curve,
                    title="PQ Curve (Measured + Fit)",
                    xlabel=f"Flow [{data['flow_unit']}]",
                    ylabel=f"ΔP [{data['dp_unit']}]",
                )
                st.components.v1.html(svg, height=450, scrolling=False)
        else:
            st.caption("Click **Plot PQ** to render the curve.")
