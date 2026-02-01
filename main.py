# main.py
# Streamlit Web App (Public URL friendly)
# - Flow unit converter (persistent results)
# - Pressure unit converter (Pa/kPa/bar/MPa/psi/inH2O/mmAq) (persistent results)
# - Pressure Head calculator (same tab)
# - Kv/Cv from points (>=1 point)
# - Fit ΔP = a·Q^n (>=2 points)
# - PQ plot (Measured + Fit) using pure SVG (NO matplotlib, NO plotly)
# - Pump curve vs System curve intersection (Kv/Cv tab) + overlay plot

import math
import streamlit as st

# ================================
# Units
# ================================
FLOW_UNITS = ["m3/h", "m3/s", "m3/min", "L/h", "L/min", "L/s", "gpm (US)", "gpm (Imp)", "cfm"]
DP_UNITS = ["Pa", "kPa", "bar", "MPa", "psi", "inH2O", "mmAq"]

G0 = 9.80665  # m/s²


def flow_to_m3s(flow, unit):
    if unit == "m3/s": return flow
    if unit == "m3/min": return flow / 60.0
    if unit == "m3/h": return flow / 3600.0
    if unit == "L/s": return flow / 1000.0
    if unit == "L/min": return flow / 1000.0 / 60.0
    if unit == "L/h": return flow / 1000.0 / 3600.0
    if unit == "gpm (US)": return flow * 3.785411784e-3 / 60.0
    if unit == "gpm (Imp)": return flow * 4.54609e-3 / 60.0
    if unit == "cfm": return flow * 0.028316846592 / 60.0
    raise ValueError(f"Unsupported flow unit: {unit}")


def m3s_to_flow(val_m3s, unit):
    if unit == "m3/s": return val_m3s
    if unit == "m3/min": return val_m3s * 60.0
    if unit == "m3/h": return val_m3s * 3600.0
    if unit == "L/s": return val_m3s * 1000.0
    if unit == "L/min": return val_m3s * 1000.0 * 60.0
    if unit == "L/h": return val_m3s * 1000.0 * 3600.0
    if unit == "gpm (US)": return val_m3s * 60.0 / 3.785411784e-3
    if unit == "gpm (Imp)": return val_m3s * 60.0 / 4.54609e-3
    if unit == "cfm": return val_m3s * 60.0 / 0.028316846592
    raise ValueError(f"Unsupported flow unit: {unit}")


def flow_to_m3h(flow, unit):
    return flow_to_m3s(flow, unit) * 3600.0


def m3h_to_flow(val_m3h, unit):
    return m3s_to_flow(val_m3h / 3600.0, unit)


def dp_to_pa(dp, unit):
    if unit == "Pa": return dp
    if unit == "kPa": return dp * 1_000.0
    if unit == "bar": return dp * 100_000.0
    if unit == "MPa": return dp * 1_000_000.0
    if unit == "psi": return dp * 6894.757293168
    if unit == "inH2O": return dp * 249.08891
    if unit == "mmAq": return dp * 9.80665
    raise ValueError(f"Unsupported pressure unit: {unit}")


def pa_to_dp(pa, unit):
    if unit == "Pa": return pa
    if unit == "kPa": return pa / 1_000.0
    if unit == "bar": return pa / 100_000.0
    if unit == "MPa": return pa / 1_000_000.0
    if unit == "psi": return pa / 6894.757293168
    if unit == "inH2O": return pa / 249.08891
    if unit == "mmAq": return pa / 9.80665
    raise ValueError(f"Unsupported pressure unit: {unit}")


def dp_to_bar(dp, unit):
    return pa_to_dp(dp_to_pa(dp, unit), "bar")


def bar_to_dp(bar, unit):
    return pa_to_dp(dp_to_pa(bar, "bar"), unit)


def pressure_to_head(dp_value, unit, sg):
    """Pressure -> head (m), using rho = 1000*SG"""
    rho = 1000.0 * sg
    pa = dp_to_pa(dp_value, unit)
    return pa / (rho * G0)


def head_to_pressure(head_m, unit, sg):
    """Head (m) -> pressure in target unit"""
    rho = 1000.0 * sg
    pa = head_m * rho * G0
    return pa_to_dp(pa, unit)


# ================================
# Kv/Cv + fitting
# ================================
def parse_points(text):
    """
    Accept lines like:
      120, 35
      150  45
    returns list[(Q, dP)] in input units
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
    """
    Fit dp = a * Q^n
    Q in m3/h, dp in bar
    """
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
# Pump vs System curve intersection
# ================================
def pump_dp_piecewise_linear(Q, pump_Qs, pump_dps):
    """Piecewise linear dp(Q) from sorted points. Returns None outside range."""
    if len(pump_Qs) < 2:
        return None
    if Q < pump_Qs[0] or Q > pump_Qs[-1]:
        return None
    for i in range(len(pump_Qs) - 1):
        q0, q1 = pump_Qs[i], pump_Qs[i + 1]
        if q0 <= Q <= q1:
            dp0, dp1 = pump_dps[i], pump_dps[i + 1]
            if q1 == q0:
                return dp0
            t = (Q - q0) / (q1 - q0)
            return dp0 * (1 - t) + dp1 * t
    return None


def system_dp(Q, dp0_bar, k_bar_per_q2):
    """ΔP_sys = ΔP0 + k Q^2   (Q in m3/h, ΔP in bar)"""
    return dp0_bar + k_bar_per_q2 * (Q ** 2)


def find_intersection_pump_vs_system(pump_Qs, pump_dps, dp0_bar, k_bar_per_q2):
    """
    Find Q where dp_pump(Q) = dp_sys(Q) within pump range.
    Inputs:
      pump_Qs: m3/h (sorted or unsorted ok)
      pump_dps: bar
      dp0_bar: bar
      k_bar_per_q2: bar/(m3/h)^2
    Returns (Q*, dp*) in internal units or None.
    """
    if len(pump_Qs) < 2:
        return None

    pairs = sorted(zip(pump_Qs, pump_dps), key=lambda x: x[0])
    pump_Qs = [p[0] for p in pairs]
    pump_dps = [p[1] for p in pairs]

    def f(q):
        dp_p = pump_dp_piecewise_linear(q, pump_Qs, pump_dps)
        if dp_p is None:
            return None
        return dp_p - system_dp(q, dp0_bar, k_bar_per_q2)

    for i in range(len(pump_Qs) - 1):
        a = pump_Qs[i]
        b = pump_Qs[i + 1]
        fa = f(a)
        fb = f(b)
        if fa is None or fb is None:
            continue

        if fa == 0:
            return a, pump_dps[i]
        if fb == 0:
            return b, pump_dps[i + 1]

        if fa * fb < 0:
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = f(mid)
                if fmid is None:
                    break
                if abs(fmid) < 1e-12:
                    lo = hi = mid
                    break
                if flo * fmid < 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid

            q_star = 0.5 * (lo + hi)
            dp_star = pump_dp_piecewise_linear(q_star, pump_Qs, pump_dps)
            if dp_star is None:
                return None
            return q_star, dp_star

    return None


def make_curve_samples(qmin, qmax, steps=200):
    """Log-spaced samples when possible; else linear."""
    if qmin <= 0 or qmax <= 0 or qmin == qmax:
        return [qmin + (qmax - qmin) * i / (steps - 1) for i in range(steps)]
    lmin = math.log10(qmin)
    lmax = math.log10(qmax)
    return [10 ** (lmin + (lmax - lmin) * i / (steps - 1)) for i in range(steps)]


# ================================
# Pure SVG plotting (with ticks)
# ================================
def nice_ticks(vmin, vmax, nticks=6):
    """Generate 'nice' ticks using 1-2-5 rule."""
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw_step = span / max(1, (nticks - 1))

    p = 10 ** math.floor(math.log10(raw_step))
    r = raw_step / p
    if r <= 1:
        step = 1 * p
    elif r <= 2:
        step = 2 * p
    elif r <= 5:
        step = 5 * p
    else:
        step = 10 * p

    start = math.floor(vmin / step) * step
    ticks = []
    t = start
    while t <= vmax + 0.5 * step:
        if t >= vmin - 0.5 * step:
            ticks.append(t)
        t += step

    if len(ticks) > 12:
        ticks = ticks[::2]
    return ticks


def fmt_tick(x):
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    if ax >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


def svg_plot(points, curve=None, width=980, height=520,
             title="PQ Curve (Measured + Fit)", xlabel="Flow", ylabel="ΔP",
             extra_curves=None, markers=None):
    """
    points: list[(x,y)] in display units (measured points)
    curve:  list[(x,y)] optional (fitted curve)
    extra_curves: list of dicts: {"name": str, "pts": list[(x,y)], "stroke": "#RRGGBB", "width": int}
    markers: list of dicts: {"name": str, "x": float, "y": float, "color": "#RRGGBB"}
    """
    if not points:
        return "<div>No data</div>"

    extra_curves = extra_curves or []
    markers = markers or []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if curve:
        xs += [p[0] for p in curve]
        ys += [p[1] for p in curve]
    for ec in extra_curves:
        xs += [p[0] for p in ec["pts"]]
        ys += [p[1] for p in ec["pts"]]
    for mk in markers:
        xs.append(mk["x"])
        ys.append(mk["y"])

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.10
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    ml, mr, mt, mb = 88, 26, 56, 78
    px0, py0 = ml, mt
    px1, py1 = width - mr, height - mb

    def xmap(x):
        return px0 + (x - xmin) / (xmax - xmin) * (px1 - px0)

    def ymap(y):
        return py1 - (y - ymin) / (ymax - ymin) * (py1 - py0)

    xticks = nice_ticks(xmin, xmax, nticks=6)
    yticks = nice_ticks(ymin, ymax, nticks=6)

    svg = []
    svg.append(
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="background:white;border-radius:14px;border:1px solid #ddd;">'
    )

    # Title
    svg.append(
        f'<text x="{width/2}" y="36" text-anchor="middle" '
        f'font-size="20" font-weight="700" fill="#111">{title}</text>'
    )

    # Plot area
    svg.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="white" stroke="#cfcfcf"/>')

    # Grid + ticks
    for t in xticks:
        x = xmap(t)
        svg.append(f'<line x1="{x:.2f}" y1="{py0}" x2="{x:.2f}" y2="{py1}" stroke="#f0f0f0"/>')
        svg.append(f'<line x1="{x:.2f}" y1="{py1}" x2="{x:.2f}" y2="{py1+7}" stroke="#888"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{py1+28}" text-anchor="middle" font-size="12" fill="#222">{fmt_tick(t)}</text>'
        )

    for t in yticks:
        y = ymap(t)
        svg.append(f'<line x1="{px0}" y1="{y:.2f}" x2="{px1}" y2="{y:.2f}" stroke="#f0f0f0"/>')
        svg.append(f'<line x1="{px0-7}" y1="{y:.2f}" x2="{px0}" y2="{y:.2f}" stroke="#888"/>')
        svg.append(
            f'<text x="{px0-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#222">{fmt_tick(t)}</text>'
        )

    # Axis labels
    svg.append(
        f'<text x="{(px0+px1)/2}" y="{height-26}" text-anchor="middle" font-size="14" fill="#111">{xlabel}</text>'
    )
    svg.append(
        f'<text x="24" y="{(py0+py1)/2}" text-anchor="middle" font-size="14" fill="#111" '
        f'transform="rotate(-90 24 {(py0+py1)/2})">{ylabel}</text>'
    )

    # extra curves (pump/system)
    for ec in extra_curves:
        pts = ec["pts"]
        if len(pts) >= 2:
            poly = " ".join([f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in pts])
            stroke = ec.get("stroke", "#999999")
            w = ec.get("width", 3)
            svg.append(f'<polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="{w}"/>')

    # fitted curve
    if curve and len(curve) >= 2:
        pts = " ".join([f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in curve])
        svg.append(f'<polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="3"/>')

    # measured points
    for x, y in points:
        cx, cy = xmap(x), ymap(y)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5.2" fill="#111"/>')

    # markers
    for mk in markers:
        cx, cy = xmap(mk["x"]), ymap(mk["y"])
        color = mk.get("color", "#ff7a00")
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7.5" fill="{color}" stroke="#111" stroke-width="1.5"/>')

    # Legend
    lx, ly = px0 + 14, py0 + 18
    legend_items = [("Measured points", "#111111", "dot")]
    if curve and len(curve) >= 2:
        legend_items.append(("Fitted curve", "#2b6cb0", "line"))
    for ec in extra_curves:
        legend_items.append((ec.get("name", "Curve"), ec.get("stroke", "#999"), "line"))
    for mk in markers:
        legend_items.append((mk.get("name", "Marker"), mk.get("color", "#ff7a00"), "dot"))

    for name, color, kind in legend_items[:6]:
        if kind == "line":
            svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text x="{lx+38}" y="{ly+4}" font-size="12" fill="#333">{name}</text>')
        else:
            svg.append(f'<circle cx="{lx+14}" cy="{ly}" r="5" fill="{color}"/>')
            svg.append(f'<text x="{lx+38}" y="{ly+4}" font-size="12" fill="#333">{name}</text>')
        ly += 18

    svg.append("</svg>")
    return "".join(svg)


# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Kv/Cv Toolkit", layout="wide")

st.title("Kv / Cv Toolkit (Web)")
st.caption("Flow + Pressure converters • Pressure head • Kv/Cv (1 point OK) • Fit ΔP=a·Q^n (≥2 points) • PQ plot (SVG) • Pump/System intersection")

tabs = st.tabs(["Converters", "Kv/Cv Tool"])

# ---------- Converters ----------
with tabs[0]:
    # init persistent outputs
    st.session_state.setdefault("flow_conv_result", None)
    st.session_state.setdefault("press_conv_result", None)
    st.session_state.setdefault("head_result", None)

    st.subheader("Flow unit converter")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        flow_val = st.number_input("Value", value=100.0, key="flow_val")
    with c2:
        flow_from = st.selectbox("From unit", FLOW_UNITS, index=4, key="flow_from")  # L/min
    with c3:
        flow_to = st.selectbox("To unit", FLOW_UNITS, index=0, key="flow_to")  # m3/h
    with c4:
        st.write("")
        st.write("")
        do_flow = st.button("Convert Flow", use_container_width=True, key="btn_flow")

    if do_flow:
        try:
            out = m3s_to_flow(flow_to_m3s(flow_val, flow_from), flow_to)
            st.session_state.flow_conv_result = (flow_val, flow_from, out, flow_to)
        except Exception as e:
            st.session_state.flow_conv_result = ("__ERROR__", str(e))

    # Always show last flow result until next flow convert
    if st.session_state.flow_conv_result is not None:
        if st.session_state.flow_conv_result[0] == "__ERROR__":
            st.error(st.session_state.flow_conv_result[1])
        else:
            v_in, u_in, v_out, u_out = st.session_state.flow_conv_result
            st.success(f"{v_in:g} {u_in}  →  **{v_out:.6g} {u_out}**")

    st.divider()

    st.subheader("Pressure unit converter")
    p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with p1:
        p_val = st.number_input("Value ", value=35.0, key="p_val")
    with p2:
        p_from = st.selectbox("From unit ", DP_UNITS, index=1, key="p_from")  # kPa
    with p3:
        p_to = st.selectbox("To unit ", DP_UNITS, index=2, key="p_to")  # bar
    with p4:
        st.write("")
        st.write("")
        do_p = st.button("Convert Pressure", use_container_width=True, key="btn_press")

    if do_p:
        try:
            out = pa_to_dp(dp_to_pa(p_val, p_from), p_to)
            st.session_state.press_conv_result = (p_val, p_from, out, p_to)
        except Exception as e:
            st.session_state.press_conv_result = ("__ERROR__", str(e))

    # Always show last pressure result until next pressure convert
    if st.session_state.press_conv_result is not None:
        if st.session_state.press_conv_result[0] == "__ERROR__":
            st.error(st.session_state.press_conv_result[1])
        else:
            v_in, u_in, v_out, u_out = st.session_state.press_conv_result
            st.success(f"{v_in:g} {u_in}  →  **{v_out:.6g} {u_out}**")

    st.divider()

    st.subheader("Pressure Head Calculator (same tab)")

    hc1, hc2, hc3 = st.columns([1.3, 1.3, 1.1])
    with hc1:
        head_val = st.number_input("Input Value", value=10.0, key="head_val")
    with hc2:
        head_unit = st.selectbox("Input Unit", DP_UNITS + ["m"], index=0, key="head_unit")
    with hc3:
        sg_val = st.number_input("Specific Gravity (SG)", value=1.0, step=0.01, key="head_sg")

    convert_type = st.radio(
        "Conversion Type",
        ["Pressure → Head (m)", "Head (m) → Pressure"],
        horizontal=True,
        key="head_mode"
    )

    if st.button("Calculate Head / Pressure", use_container_width=True, key="btn_head_calc"):
        try:
            if convert_type == "Pressure → Head (m)":
                if head_unit == "m":
                    result = float(head_val)
                else:
                    result = pressure_to_head(head_val, head_unit, sg_val)
                st.session_state.head_result = ("Head", result, "m")
            else:
                # Head input must be m; if user selected a pressure unit here, treat input as that pressure -> head first is confusing
                # We'll interpret: head_val is head in meters ALWAYS when converting Head->Pressure.
                head_m = float(head_val) if head_unit == "m" else float(head_val)
                # If user didn't choose m, we still treat input as meters (simple, avoids ambiguity)
                out_p = head_to_pressure(head_m, "kPa", sg_val)  # default output kPa? better let user choose output
                st.session_state.head_result = ("Pressure", out_p, "kPa")
        except Exception as e:
            st.session_state.head_result = ("__ERROR__", str(e), "")

    # show last head calc
    if st.session_state.head_result is not None:
        if st.session_state.head_result[0] == "__ERROR__":
            st.error(st.session_state.head_result[1])
        else:
            kind, val, u = st.session_state.head_result
            if kind == "Head":
                st.success(f"Head = **{val:.4f} {u}**")
            else:
                st.success(f"Pressure = **{val:.6g} {u}**")

    st.caption("Note: For Head→Pressure, output is shown in kPa (simple default). If you want selectable output units, tell me and I’ll add it cleanly.")


# ---------- Kv/Cv Tool ----------
with tabs[1]:
    left, right = st.columns([1.1, 1.2])

    # session state storage
    st.session_state.setdefault("last_data", None)
    st.session_state.setdefault("last_fit", None)
    st.session_state.setdefault("results_text", "")
    st.session_state.setdefault("pump_system_data", None)
    st.session_state.setdefault("pump_system_text", "")

    with left:
        st.subheader("Inputs")

        sg = st.number_input("Specific Gravity (SG)", value=1.0, min_value=0.000001, step=0.01, key="kv_sg")
        flow_unit = st.selectbox("Flow unit (input points)", FLOW_UNITS, index=4, key="kv_flow_unit")  # L/min
        dp_unit = st.selectbox("ΔP unit (input points)", DP_UNITS, index=1, key="kv_dp_unit")          # kPa

        st.markdown("**Paste points (Q, ΔP) — one per line**")
        points_text = st.text_area(
            "Example format: `120, 35`",
            value="120, 35\n150, 45\n180, 62\n",
            height=180,
            key="kv_points_text"
        )

        colb1, colb2 = st.columns(2)
        with colb1:
            btn_calc = st.button("Calculate / Fit", use_container_width=True, key="btn_calc")
        with colb2:
            btn_plot = st.button("Plot PQ", use_container_width=True, key="btn_plot")

        # ---- Pump/System Intersection UI ----
        st.divider()
        st.subheader("Pump vs System Curve Intersection")

        enable_intersection = st.checkbox("Enable pump/system intersection", value=False, key="enable_intersection")
        st.caption("Pump points use SAME units as selected above. System curve: ΔP_sys = ΔP0 + k·Q².")

        pump_text = st.text_area(
            "Pump curve points (Q, ΔP) — one per line",
            value="0, 80\n50, 70\n100, 55\n150, 35\n200, 10\n",
            height=140,
            key="pump_points_text"
        )

        c_int1, c_int2 = st.columns(2)
        with c_int1:
            sys_dp0 = st.number_input("System ΔP0 (same ΔP unit)", value=5.0, key="sys_dp0")
        with c_int2:
            sys_k = st.number_input("System k (ΔP / Q²)", value=0.001, format="%.8f", key="sys_k")

        btn_intersect = st.button("Compute Intersection", use_container_width=True, key="btn_intersect")

    # ---- Calculate/Fit ----
    if btn_calc:
        try:
            pts = parse_points(points_text)
            Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pts]
            dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pts]

            kvs = [kv_from_point(Q, dpb, sg) for Q, dpb in zip(Q_m3h, dp_bar)]
            cvs = [kv / 0.865 for kv in kvs]

            lines = []
            lines.append(f"SG = {sg:.4f}")
            lines.append(f"Input units: Q in {flow_unit}, ΔP in {dp_unit}")
            lines.append("Internal units: Q in m³/h, ΔP in bar\n")
            lines.append("Point-wise results:")
            for i, ((q_in, dp_in), Qint, dpint, kv, cv) in enumerate(zip(pts, Q_m3h, dp_bar, kvs, cvs), 1):
                lines.append(
                    f"  {i:>2}. Q={q_in:g} {flow_unit:<8} (= {Qint:.6g} m³/h)   "
                    f"ΔP={dp_in:g} {dp_unit:<6} (= {dpint:.6g} bar)   "
                    f"Kv={kv:.4f}   Cv={cv:.4f}"
                )

            last_fit = None
            if len(pts) >= 2:
                a, n_exp = fit_power_law(Q_m3h, dp_bar)
                K0 = math.sqrt(sg / a)
                m_exp = 1 - n_exp / 2
                last_fit = {"a": a, "n": n_exp, "K0": K0, "m": m_exp}

                lines.append("\nFitted model (Q in m³/h, ΔP in bar):")
                lines.append(f"  ΔP = a · Q^n")
                lines.append(f"  a = {a:.6g}")
                lines.append(f"  n = {n_exp:.6g}")
                lines.append("\nDerived functions:")
                lines.append(f"  Kv(Q) = K0 · Q^m")
                lines.append(f"  K0 = sqrt(SG/a) = {K0:.6g}")
                lines.append(f"  m  = 1 - n/2    = {m_exp:.6g}")
                lines.append("  Cv(Q) = Kv(Q) / 0.865")
            else:
                lines.append("\nFit not available (need ≥2 points). Single-point Kv/Cv computed.")

            st.session_state.last_data = {
                "pts_raw": pts,
                "Q_m3h": Q_m3h,
                "dp_bar": dp_bar,
                "flow_unit": flow_unit,
                "dp_unit": dp_unit,
                "sg": sg
            }
            st.session_state.last_fit = last_fit
            st.session_state.results_text = "\n".join(lines)
            st.success("Calculated / fitted.")
        except Exception as e:
            st.error(str(e))

    # ---- Intersection compute ----
    if enable_intersection and btn_intersect:
        try:
            pump_pts = parse_points(pump_text)

            pump_Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pump_pts]
            pump_dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pump_pts]

            dp0_bar = dp_to_bar(sys_dp0, dp_unit)

            # sys_k is entered as (dp_unit)/(flow_unit^2).
            # Convert to bar/(m3/h)^2 by converting numerator to bar and denominator to (m3/h)^2.
            q_test_in = 1.0
            q_test_m3h = flow_to_m3h(q_test_in, flow_unit)
            k_bar_per_m3h2 = dp_to_bar(sys_k, dp_unit) / (q_test_m3h ** 2)

            inter = find_intersection_pump_vs_system(pump_Q_m3h, pump_dp_bar, dp0_bar, k_bar_per_m3h2)

            if inter is None:
                st.session_state.pump_system_data = {
                    "pump_Q_m3h": pump_Q_m3h,
                    "pump_dp_bar": pump_dp_bar,
                    "dp0_bar": dp0_bar,
                    "k_bar_per_m3h2": k_bar_per_m3h2,
                    "intersection": None,
                    "flow_unit": flow_unit,
                    "dp_unit": dp_unit
                }
                st.session_state.pump_system_text = "No intersection found within pump curve Q range."
            else:
                q_star_m3h, dp_star_bar = inter
                q_star_disp = m3h_to_flow(q_star_m3h, flow_unit)
                dp_star_disp = bar_to_dp(dp_star_bar, dp_unit)

                st.session_state.pump_system_data = {
                    "pump_Q_m3h": pump_Q_m3h,
                    "pump_dp_bar": pump_dp_bar,
                    "dp0_bar": dp0_bar,
                    "k_bar_per_m3h2": k_bar_per_m3h2,
                    "intersection": (q_star_m3h, dp_star_bar),
                    "flow_unit": flow_unit,
                    "dp_unit": dp_unit
                }
                st.session_state.pump_system_text = (
                    f"Intersection: Q = {q_star_disp:.6g} {flow_unit},  ΔP = {dp_star_disp:.6g} {dp_unit}"
                )
        except Exception as e:
            st.session_state.pump_system_data = None
            st.session_state.pump_system_text = f"Error: {e}"

    # ---- Right side outputs + plot ----
    with right:
        st.subheader("Results")
        st.text_area(
            "Output",
            value=st.session_state.get("results_text", "Click 'Calculate / Fit' to see results."),
            height=260
        )

        st.subheader("Pump/System Intersection")
        txt = st.session_state.get("pump_system_text", "Compute intersection to see the operating point.")
        st.info(txt)

        st.subheader("PQ Plot")
        if btn_plot:
            try:
                if not st.session_state.last_data:
                    raise ValueError("No data. Click Calculate / Fit first.")

                data = st.session_state.last_data
                fit = st.session_state.last_fit

                # measured points in display units
                Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in data["Q_m3h"]]
                dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in data["dp_bar"]]
                points = list(zip(Q_axis, dp_axis))

                # fitted curve in display units
                curve = None
                if fit and len(data["Q_m3h"]) >= 2:
                    a = fit["a"]
                    n = fit["n"]
                    qmin, qmax = min(data["Q_m3h"]), max(data["Q_m3h"])
                    if qmin > 0 and qmax > 0 and qmin != qmax:
                        qs_m3h = make_curve_samples(qmin, qmax, steps=180)
                        dps_bar = [a * (q ** n) for q in qs_m3h]
                        curve_Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in qs_m3h]
                        curve_dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in dps_bar]
                        curve = list(zip(curve_Q_axis, curve_dp_axis))

                # pump/system overlays
                extra_curves = []
                markers = []
                ps = st.session_state.get("pump_system_data", None)
                if ps and ps.get("flow_unit") == data["flow_unit"] and ps.get("dp_unit") == data["dp_unit"]:
                    pump_Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in ps["pump_Q_m3h"]]
                    pump_dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in ps["pump_dp_bar"]]
                    pump_curve = list(zip(pump_Q_axis, pump_dp_axis))

                    qmin_p = min(ps["pump_Q_m3h"])
                    qmax_p = max(ps["pump_Q_m3h"])
                    qs_m3h = make_curve_samples(qmin_p, qmax_p, steps=220)
                    sys_dp_bar = [system_dp(q, ps["dp0_bar"], ps["k_bar_per_m3h2"]) for q in qs_m3h]
                    sys_Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in qs_m3h]
                    sys_dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in sys_dp_bar]
                    sys_curve = list(zip(sys_Q_axis, sys_dp_axis))

                    extra_curves.append({"name": "Pump curve", "pts": pump_curve, "stroke": "#d64545", "width": 3})
                    extra_curves.append({"name": "System curve", "pts": sys_curve, "stroke": "#2f855a", "width": 3})

                    if ps["intersection"] is not None:
                        q_star_m3h, dp_star_bar = ps["intersection"]
                        mkx = m3h_to_flow(q_star_m3h, data["flow_unit"])
                        mky = bar_to_dp(dp_star_bar, data["dp_unit"])
                        markers.append({"name": "Operating point", "x": mkx, "y": mky, "color": "#ff7a00"})

                svg = svg_plot(
                    points=points,
                    curve=curve,
                    title="PQ Curve (Measured + Fit)",
                    xlabel=f"Flow [{data['flow_unit']}]",
                    ylabel=f"ΔP [{data['dp_unit']}]",
                    extra_curves=extra_curves,
                    markers=markers
                )

                st.components.v1.html(svg, height=560, scrolling=False)
                st.success("Plot updated.")
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Click **Calculate / Fit** first, then click **Plot PQ**.")
