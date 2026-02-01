# main.py
# Streamlit Web App (Public URL friendly)
# - Flow unit converter (persistent results)
# - Pressure unit converter (Pa/kPa/bar/MPa/psi/inH2O/mmAq) (persistent results)
# - Pressure Head calculator (same tab)
# - NEW: Flow rate + Area -> Velocity calculator (same tab, persistent)
# - Kv/Cv from points (>=1 point)
# - Fit ΔP = a·Q^n (>=2 points)
# - Cv PQ plot (Measured + Fit) using pure SVG (NO matplotlib, NO plotly)  [ALONE]
# - Pump curve vs System curve (separate plot) + Intersection optional + Clear buttons
# - Plot mode selection + legend placement

import math
import streamlit as st


# ================================
# Units
# ================================
FLOW_UNITS = ["m3/h", "m3/s", "m3/min", "L/h", "L/min", "L/s", "gpm (US)", "gpm (Imp)", "cfm"]
DP_UNITS = ["Pa", "kPa", "bar", "MPa", "psi", "inH2O", "mmAq"]

AREA_UNITS = ["m2", "cm2", "mm2", "in2"]

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
    rho = 1000.0 * sg
    pa = dp_to_pa(dp_value, unit)
    return pa / (rho * G0)


def head_to_pressure(head_m, unit, sg):
    rho = 1000.0 * sg
    pa = head_m * rho * G0
    return pa_to_dp(pa, unit)


def area_to_m2(area, unit):
    if unit == "m2": return area
    if unit == "cm2": return area * 1e-4
    if unit == "mm2": return area * 1e-6
    if unit == "in2": return area * 0.00064516
    raise ValueError(f"Unsupported area unit: {unit}")

VEL_UNITS = ["m/s", "m/min", "m/h", "ft/s", "ft/min"]

def ms_to_vel(v_ms, unit):
    if unit == "m/s":   return v_ms
    if unit == "m/min": return v_ms * 60.0
    if unit == "m/h":   return v_ms * 3600.0
    if unit == "ft/s":  return v_ms * 3.280839895
    if unit == "ft/min":return v_ms * 3.280839895 * 60.0
    raise ValueError(f"Unsupported velocity unit: {unit}")

AREA_UNITS = ["m2", "cm2", "mm2", "in2", "ft2"]

VEL_UNITS = ["m/s", "m/min", "m/h", "ft/s", "ft/min"]

def area_to_m2(a, unit):
    if unit == "m2":  return a
    if unit == "cm2": return a * 1e-4
    if unit == "mm2": return a * 1e-6
    if unit == "in2": return a * (0.0254 ** 2)
    if unit == "ft2": return a * (0.3048 ** 2)
    raise ValueError(f"Unsupported area unit: {unit}")

def m2_to_area(a_m2, unit):
    if unit == "m2":  return a_m2
    if unit == "cm2": return a_m2 / 1e-4
    if unit == "mm2": return a_m2 / 1e-6
    if unit == "in2": return a_m2 / (0.0254 ** 2)
    if unit == "ft2": return a_m2 / (0.3048 ** 2)
    raise ValueError(f"Unsupported area unit: {unit}")

def vel_to_ms(v, unit):
    if unit == "m/s":   return v
    if unit == "m/min": return v / 60.0
    if unit == "m/h":   return v / 3600.0
    if unit == "ft/s":  return v * 0.3048
    if unit == "ft/min":return v * 0.3048 / 60.0
    raise ValueError(f"Unsupported velocity unit: {unit}")

def ms_to_vel(v_ms, unit):
    if unit == "m/s":   return v_ms
    if unit == "m/min": return v_ms * 60.0
    if unit == "m/h":   return v_ms * 3600.0
    if unit == "ft/s":  return v_ms / 0.3048
    if unit == "ft/min":return v_ms * 60.0 / 0.3048
    raise ValueError(f"Unsupported velocity unit: {unit}")


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
# Pump/System helpers
# ================================
def pump_dp_piecewise_linear(Q, pump_Qs, pump_dps):
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
    return dp0_bar + k_bar_per_q2 * (Q ** 2)


def find_intersection_pump_vs_system(pump_Qs, pump_dps, dp0_bar, k_bar_per_q2):
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
    if qmax <= qmin:
        return [qmin]
    if qmin <= 0:
        return [qmin + (qmax - qmin) * i / (steps - 1) for i in range(steps)]
    lmin = math.log10(qmin)
    lmax = math.log10(qmax)
    return [10 ** (lmin + (lmax - lmin) * i / (steps - 1)) for i in range(steps)]


# ================================
# SVG plotting + legend positioning
# ================================
def nice_ticks(vmin, vmax, nticks=6):
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


def _choose_legend_pos(legend_loc, px0, py0, px1, py1):
    pad = 14
    if legend_loc == "Top-left":
        return px0 + pad, py0 + pad, "start"
    if legend_loc == "Top-right":
        return px1 - pad, py0 + pad, "end"
    if legend_loc == "Bottom-left":
        return px0 + pad, py1 - 60, "start"
    if legend_loc == "Bottom-right":
        return px1 - pad, py1 - 60, "end"
    if legend_loc == "Outside-right":
        return px1 + 18, py0 + pad, "start"
    return px1 - pad, py0 + pad, "end"  # Auto


def svg_plot(
    width=980, height=520,
    title="Plot", xlabel="X", ylabel="Y",
    curves=None, points=None, markers=None,
    legend_loc="Auto"
):
    curves = curves or []
    points = points or []
    markers = markers or []

    has_any = bool(points) or bool(markers) or any(c.get("pts") for c in curves)
    if not has_any:
        return "<div>No data</div>"

    xs, ys = [], []
    for c in curves:
        for x, y in c["pts"]:
            xs.append(x); ys.append(y)
    for x, y in points:
        xs.append(x); ys.append(y)
    for m in markers:
        xs.append(m["x"]); ys.append(m["y"])

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    if xmin == xmax:
        xmin -= 1; xmax += 1
    if ymin == ymax:
        ymin -= 1; ymax += 1

    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.10
    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad

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

    svg.append(
        f'<text x="{width/2}" y="36" text-anchor="middle" '
        f'font-size="20" font-weight="700" fill="#111">{title}</text>'
    )

    svg.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="white" stroke="#cfcfcf"/>')

    for t in xticks:
        x = xmap(t)
        svg.append(f'<line x1="{x:.2f}" y1="{py0}" x2="{x:.2f}" y2="{py1}" stroke="#f0f0f0"/>')
        svg.append(f'<line x1="{x:.2f}" y1="{py1}" x2="{x:.2f}" y2="{py1+7}" stroke="#888"/>')
        svg.append(f'<text x="{x:.2f}" y="{py1+28}" text-anchor="middle" font-size="12" fill="#222">{fmt_tick(t)}</text>')

    for t in yticks:
        y = ymap(t)
        svg.append(f'<line x1="{px0}" y1="{y:.2f}" x2="{px1}" y2="{y:.2f}" stroke="#f0f0f0"/>')
        svg.append(f'<line x1="{px0-7}" y1="{y:.2f}" x2="{px0}" y2="{y:.2f}" stroke="#888"/>')
        svg.append(f'<text x="{px0-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#222">{fmt_tick(t)}</text>')

    svg.append(f'<text x="{(px0+px1)/2}" y="{height-26}" text-anchor="middle" font-size="14" fill="#111">{xlabel}</text>')
    svg.append(
        f'<text x="24" y="{(py0+py1)/2}" text-anchor="middle" font-size="14" fill="#111" '
        f'transform="rotate(-90 24 {(py0+py1)/2})">{ylabel}</text>'
    )

    for c in curves:
        pts = c.get("pts") or []
        if len(pts) >= 2:
            poly = " ".join([f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in pts])
            stroke = c.get("stroke", "#999999")
            w = c.get("width", 3)
            svg.append(f'<polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="{w}"/>')

    for x, y in points:
        cx, cy = xmap(x), ymap(y)
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5.2" fill="#111"/>')

    for m in markers:
        cx, cy = xmap(m["x"]), ymap(m["y"])
        color = m.get("color", "#ff7a00")
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7.5" fill="{color}" stroke="#111" stroke-width="1.5"/>')

    legend_items = []
    for c in curves:
        legend_items.append(("line", c.get("name", "Curve"), c.get("stroke", "#999999")))
    for m in markers:
        legend_items.append(("dot", m.get("name", "Marker"), m.get("color", "#ff7a00")))
    if points:
        legend_items.append(("dot", "Measured points", "#111111"))

    if legend_items:
        lx, ly, anchor = _choose_legend_pos(legend_loc, px0, py0, px1, py1)

        box_w = 210
        box_h = 18 * min(len(legend_items), 8) + 10
        if legend_loc == "Outside-right":
            bx = lx - 6
        else:
            bx = lx - (box_w if anchor == "end" else 6)
        by = ly - 8

        svg.append(f'<rect x="{bx}" y="{by}" width="{box_w}" height="{box_h}" fill="white" stroke="#e5e5e5" opacity="0.95" rx="8"/>')

        y = ly + 8
        for kind, name, color in legend_items[:8]:
            if kind == "line":
                x1 = lx - 28 if anchor == "end" else lx
                x2 = lx if anchor == "end" else lx + 28
                svg.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            else:
                cx = lx - 14 if anchor == "end" else lx + 14
                svg.append(f'<circle cx="{cx}" cy="{y}" r="5" fill="{color}"/>')

            tx = lx - 38 if anchor == "end" else lx + 38
            svg.append(f'<text x="{tx}" y="{y+4}" text-anchor="{anchor}" font-size="12" fill="#333">{name}</text>')
            y += 18

    svg.append("</svg>")
    return "".join(svg)


def fmt_num(x):
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.6g}"
    if ax >= 1:
        return f"{x:.6g}"
    return f"{x:.6g}"


# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Kv/Cv Toolkit", layout="wide")

st.title("Kv / Cv Toolkit (Web)")
st.caption("Converters • Pressure head • Velocity calc • Kv/Cv • Fit • Cv plot • Pump/System plot")

tabs = st.tabs(["Converters", "Kv/Cv Tool"])


# ---------- Converters ----------
with tabs[0]:
    st.session_state.setdefault("flow_conv_result", None)
    st.session_state.setdefault("press_conv_result", None)
    st.session_state.setdefault("head_result", None)
    st.session_state.setdefault("vel_result", None)

    st.subheader("Flow unit converter")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        flow_val = st.number_input("Value", value=100.0, key="flow_val")
    with c2:
        flow_from = st.selectbox("From unit", FLOW_UNITS, index=4, key="flow_from")
    with c3:
        flow_to = st.selectbox("To unit", FLOW_UNITS, index=0, key="flow_to")
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
        p_from = st.selectbox("From unit ", DP_UNITS, index=1, key="p_from")
    with p3:
        p_to = st.selectbox("To unit ", DP_UNITS, index=2, key="p_to")
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

    if st.session_state.press_conv_result is not None:
        if st.session_state.press_conv_result[0] == "__ERROR__":
            st.error(st.session_state.press_conv_result[1])
        else:
            v_in, u_in, v_out, u_out = st.session_state.press_conv_result
            st.success(f"{v_in:g} {u_in}  →  **{v_out:.6g} {u_out}**")

    st.divider()

    st.subheader("Pressure Head Calculator")
    hc1, hc2, hc3 = st.columns([1.3, 1.3, 1.1])
    with hc1:
        head_val = st.number_input("Input Value", value=10.0, key="head_val")
    with hc2:
        head_unit = st.selectbox("Input Unit", DP_UNITS + ["m"], index=0, key="head_unit")
    with hc3:
        sg_val = st.number_input("Specific Gravity (SG)", value=1.0, step=0.01, key="head_sg")

    convert_type = st.radio(
        "Conversion Type",
        ["Pressure → Head (m)", "Head (m) → Pressure (kPa)"],
        horizontal=True,
        key="head_mode"
    )

    if st.button("Calculate Head / Pressure", use_container_width=True, key="btn_head_calc"):
        try:
            if convert_type.startswith("Pressure"):
                if head_unit == "m":
                    result = float(head_val)
                else:
                    result = pressure_to_head(head_val, head_unit, sg_val)
                st.session_state.head_result = ("Head", result, "m")
            else:
                head_m = float(head_val)
                out_p = head_to_pressure(head_m, "kPa", sg_val)
                st.session_state.head_result = ("Pressure", out_p, "kPa")
        except Exception as e:
            st.session_state.head_result = ("__ERROR__", str(e), "")

    if st.session_state.head_result is not None:
        if st.session_state.head_result[0] == "__ERROR__":
            st.error(st.session_state.head_result[1])
        else:
            kind, val, u = st.session_state.head_result
            if kind == "Head":
                st.success(f"Head = **{val:.4f} {u}**")
            else:
                st.success(f"Pressure = **{val:.6g} {u}**")

    st.divider()

st.subheader("Flow / Velocity / Area Calculator (bi-directional)")

# --- session state init ---
if "va_mode" not in st.session_state:
    st.session_state.va_mode = "Flow rate + Area → Velocity"
if "va_vel_ms" not in st.session_state:
    st.session_state.va_vel_ms = None          # store velocity in m/s
if "va_flow_m3s" not in st.session_state:
    st.session_state.va_flow_m3s = None        # store flow in m3/s
if "va_inputs" not in st.session_state:
    st.session_state.va_inputs = None          # store last inputs
if "va_vel_out_unit" not in st.session_state:
    st.session_state.va_vel_out_unit = "m/s"
if "va_flow_out_unit" not in st.session_state:
    st.session_state.va_flow_out_unit = "L/min"

# --- mode selection ---
st.session_state.va_mode = st.radio(
    "Mode",
    ["Flow rate + Area → Velocity", "Velocity + Area → Flow rate"],
    horizontal=True
)

# --- input rows (change depending on mode) ---
if st.session_state.va_mode == "Flow rate + Area → Velocity":
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        q_val = st.number_input("Flow rate value", value=100.0, key="va_q_val")
    with c2:
        q_unit = st.selectbox("Flow rate unit", FLOW_UNITS, index=4, key="va_q_unit")  # L/min
    with c3:
        a_val = st.number_input("Area value", value=100.0, key="va_a_val")
    with c4:
        a_unit = st.selectbox("Area unit", AREA_UNITS, index=2, key="va_a_unit")  # mm2

else:
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        v_val = st.number_input("Velocity value", value=1.0, key="va_v_val")
    with c2:
        v_unit = st.selectbox("Velocity unit", VEL_UNITS, index=0, key="va_v_unit")  # m/s
    with c3:
        a_val = st.number_input("Area value ", value=100.0, key="va_a_val2")
    with c4:
        a_unit = st.selectbox("Area unit ", AREA_UNITS, index=2, key="va_a_unit2")  # mm2

# --- action buttons ---
b1, b2 = st.columns([1, 1])
with b1:
    do_calc = st.button("Calculate", use_container_width=True, key="va_calc_btn")
with b2:
    do_clear = st.button("Clear Result", use_container_width=True, key="va_clear_btn")

if do_clear:
    st.session_state.va_vel_ms = None
    st.session_state.va_flow_m3s = None
    st.session_state.va_inputs = None

# --- calculate and store base units ---
if do_calc:
    try:
        # common: area must be > 0
        A_m2 = area_to_m2(a_val, a_unit)
        if A_m2 <= 0:
            raise ValueError("Area must be > 0")

        if st.session_state.va_mode == "Flow rate + Area → Velocity":
            Q_m3s = flow_to_m3s(q_val, q_unit)
            v_ms = Q_m3s / A_m2

            st.session_state.va_flow_m3s = Q_m3s
            st.session_state.va_vel_ms = v_ms
            st.session_state.va_inputs = ("Q+A→v", q_val, q_unit, a_val, a_unit)

        else:
            v_ms = vel_to_ms(v_val, v_unit)
            Q_m3s = v_ms * A_m2

            st.session_state.va_flow_m3s = Q_m3s
            st.session_state.va_vel_ms = v_ms
            st.session_state.va_inputs = ("v+A→Q", v_val, v_unit, a_val, a_unit)

    except Exception as e:
        st.session_state.va_flow_m3s = "__ERROR__"
        st.session_state.va_vel_ms = "__ERROR__"
        st.session_state.va_inputs = str(e)

# --- output unit selectors (kept near result) ---
out1, out2 = st.columns([2.5, 1])

with out2:
    if st.session_state.va_mode == "Flow rate + Area → Velocity":
        st.session_state.va_vel_out_unit = st.selectbox(
            "Output velocity unit",
            VEL_UNITS,
            index=VEL_UNITS.index(st.session_state.va_vel_out_unit),
            key="va_vel_out_unit_sel",
        )
    else:
        st.session_state.va_flow_out_unit = st.selectbox(
            "Output flow unit",
            FLOW_UNITS,
            index=FLOW_UNITS.index(st.session_state.va_flow_out_unit),
            key="va_flow_out_unit_sel",
        )

# --- show result (always show last result until cleared or recalculated) ---
if st.session_state.va_vel_ms is not None and st.session_state.va_vel_ms != "__ERROR__":
    mode_tag = st.session_state.va_inputs[0] if isinstance(st.session_state.va_inputs, tuple) else ""
    if st.session_state.va_mode == "Flow rate + Area → Velocity":
        v_out = ms_to_vel(st.session_state.va_vel_ms, st.session_state.va_vel_out_unit)
        _, qv, qu, av, au = st.session_state.va_inputs
        st.success(
            f"{mode_tag}: Q={qv:g} {qu}, A={av:g} {au}  →  "
            f"**v={v_out:.6g} {st.session_state.va_vel_out_unit}** "
            f"(stored: {st.session_state.va_vel_ms:.6g} m/s)"
        )
    else:
        q_out = m3s_to_flow(st.session_state.va_flow_m3s, st.session_state.va_flow_out_unit)
        _, vv, vu, av, au = st.session_state.va_inputs
        st.success(
            f"{mode_tag}: v={vv:g} {vu}, A={av:g} {au}  →  "
            f"**Q={q_out:.6g} {st.session_state.va_flow_out_unit}** "
            f"(stored: {st.session_state.va_flow_m3s:.6g} m³/s)"
        )

elif st.session_state.va_vel_ms == "__ERROR__":
    st.error(st.session_state.va_inputs)
else:
    st.info("Enter values and click **Calculate**. You can change the output unit after calculation.")

# ---------- Kv/Cv Tool ----------
with tabs[1]:
    st.session_state.setdefault("last_data", None)
    st.session_state.setdefault("last_fit", None)
    st.session_state.setdefault("results_text", "")
    st.session_state.setdefault("cv_svg", None)
    st.session_state.setdefault("ps_svg", None)
    st.session_state.setdefault("pump_system_text", "")

    left, right = st.columns([1.1, 1.2])

    with left:
        st.subheader("Inputs")

        sg = st.number_input("Specific Gravity (SG)", value=1.0, min_value=0.000001, step=0.01, key="kv_sg")
        flow_unit = st.selectbox("Flow unit (input points)", FLOW_UNITS, index=4, key="kv_flow_unit")
        dp_unit = st.selectbox("ΔP unit (input points)", DP_UNITS, index=1, key="kv_dp_unit")

        st.markdown("**Paste points (Q, ΔP) — one per line**")
        points_text = st.text_area(
            "Example format: `120, 35`",
            value="120, 35\n150, 45\n180, 62\n",
            height=180,
            key="kv_points_text"
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            btn_calc = st.button("Calculate / Fit", use_container_width=True, key="btn_calc")
        with b2:
            btn_plot_cv = st.button("Plot Cv PQ", use_container_width=True, key="btn_plot_cv")
        with b3:
            btn_clear_cv = st.button("Clear Cv Plot", use_container_width=True, key="btn_clear_cv")

        st.divider()
        st.subheader("Pump vs System Curve (Separate Diagram)")
        st.caption("Pump points use SAME units as above. System curve: ΔP_sys = ΔP0 + k·Q²")

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

        st.markdown("**Plot options**")
        plot_mode = st.selectbox(
            "What to plot",
            [
                "Pump curve only",
                "System curve only",
                "Operating point only",
                "Pump + System",
                "Pump + System + Operating point",
            ],
            index=4,
            key="ps_plot_mode"
        )

        legend_loc = st.selectbox(
            "Legend location",
            ["Auto", "Top-left", "Top-right", "Bottom-left", "Bottom-right", "Outside-right"],
            index=0,
            key="ps_legend_loc"
        )

        bb1, bb2, bb3 = st.columns(3)
        with bb1:
            btn_intersect = st.button("Compute Intersection", use_container_width=True, key="btn_intersect")
        with bb2:
            btn_plot_ps = st.button("Plot Pump/System", use_container_width=True, key="btn_plot_ps")
        with bb3:
            btn_clear_ps = st.button("Clear Pump/System Plot", use_container_width=True, key="btn_clear_ps")

    if btn_clear_cv:
        st.session_state.cv_svg = None

    if btn_clear_ps:
        st.session_state.ps_svg = None
        st.session_state.pump_system_text = ""

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
                last_fit = {"a": a, "n": n_exp}

                lines.append("\nFitted model (Q in m³/h, ΔP in bar):")
                lines.append(f"  ΔP = a · Q^n")
                lines.append(f"  a = {a:.6g}")
                lines.append(f"  n = {n_exp:.6g}")

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

    if btn_plot_cv:
        try:
            if not st.session_state.last_data:
                raise ValueError("No data. Click Calculate / Fit first.")
            data = st.session_state.last_data
            fit = st.session_state.last_fit

            Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in data["Q_m3h"]]
            dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in data["dp_bar"]]
            pts = list(zip(Q_axis, dp_axis))

            curves = []
            if fit and len(data["Q_m3h"]) >= 2:
                a = fit["a"]
                n = fit["n"]
                qmin, qmax = min(data["Q_m3h"]), max(data["Q_m3h"])
                if qmin > 0 and qmax > 0 and qmin != qmax:
                    qs_m3h = make_curve_samples(qmin, qmax, steps=180)
                    dps_bar = [a * (q ** n) for q in qs_m3h]
                    curve_Q_axis = [m3h_to_flow(q, data["flow_unit"]) for q in qs_m3h]
                    curve_dp_axis = [bar_to_dp(dpb, data["dp_unit"]) for dpb in dps_bar]
                    curves.append({"name": "Fitted curve", "pts": list(zip(curve_Q_axis, curve_dp_axis)), "stroke": "#2b6cb0", "width": 3})

            st.session_state.cv_svg = svg_plot(
                title="Cv PQ Curve (Measured + Fit)",
                xlabel=f"Flow [{data['flow_unit']}]",
                ylabel=f"ΔP [{data['dp_unit']}]",
                curves=curves,
                points=pts,
                markers=[],
                legend_loc="Auto"
            )
        except Exception as e:
            st.error(str(e))

    if btn_intersect:
        try:
            pump_pts = parse_points(pump_text)
            pump_Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pump_pts]
            pump_dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pump_pts]

            dp0_bar = dp_to_bar(sys_dp0, dp_unit)
            q_test_m3h = flow_to_m3h(1.0, flow_unit)
            k_bar_per_m3h2 = dp_to_bar(sys_k, dp_unit) / (q_test_m3h ** 2)

            eq = f"ΔP_sys = {fmt_num(sys_dp0)} {dp_unit} + ({fmt_num(sys_k)}) {dp_unit}/({flow_unit})² · Q²"

            inter = find_intersection_pump_vs_system(pump_Q_m3h, pump_dp_bar, dp0_bar, k_bar_per_m3h2)
            if inter is None:
                st.session_state.pump_system_text = f"System curve: {eq}\nIntersection: (none found in pump Q range)"
            else:
                q_star_m3h, dp_star_bar = inter
                q_star_disp = m3h_to_flow(q_star_m3h, flow_unit)
                dp_star_disp = bar_to_dp(dp_star_bar, dp_unit)
                st.session_state.pump_system_text = (
                    f"System curve: {eq}\n"
                    f"Intersection: Q = {q_star_disp:.6g} {flow_unit},  ΔP = {dp_star_disp:.6g} {dp_unit}"
                )
        except Exception as e:
            st.session_state.pump_system_text = f"Error: {e}"

    if btn_plot_ps:
        try:
            pump_pts = parse_points(pump_text)
            if len(pump_pts) < 2 and plot_mode != "System curve only":
                raise ValueError("Pump curve needs at least 2 points (unless plotting System curve only).")

            eq = f"ΔP_sys = {fmt_num(sys_dp0)} {dp_unit} + ({fmt_num(sys_k)}) {dp_unit}/({flow_unit})² · Q²"
            st.session_state.pump_system_text = f"System curve: {eq}"

            dp0_bar = dp_to_bar(sys_dp0, dp_unit)
            q_test_m3h = flow_to_m3h(1.0, flow_unit)
            k_bar_per_m3h2 = dp_to_bar(sys_k, dp_unit) / (q_test_m3h ** 2)

            curves = []
            markers = []

            pump_Q_m3h = []
            pump_dp_bar = []

            if len(pump_pts) >= 2:
                pump_Q_m3h = [flow_to_m3h(q, flow_unit) for q, _ in pump_pts]
                pump_dp_bar = [dp_to_bar(dp, dp_unit) for _, dp in pump_pts]
                pairs = sorted(zip(pump_Q_m3h, pump_dp_bar), key=lambda x: x[0])
                pump_Q_m3h = [p[0] for p in pairs]
                pump_dp_bar = [p[1] for p in pairs]

            if plot_mode in ["Pump curve only", "Pump + System", "Pump + System + Operating point"]:
                pump_Q_axis = [m3h_to_flow(q, flow_unit) for q in pump_Q_m3h]
                pump_dp_axis = [bar_to_dp(dpb, dp_unit) for dpb in pump_dp_bar]
                curves.append({"name": "Pump curve", "pts": list(zip(pump_Q_axis, pump_dp_axis)), "stroke": "#d64545", "width": 3})

            if plot_mode in ["System curve only", "Pump + System", "Pump + System + Operating point"]:
                if len(pump_Q_m3h) >= 2:
                    qmin_p, qmax_p = min(pump_Q_m3h), max(pump_Q_m3h)
                    qs_m3h = make_curve_samples(max(qmin_p, 1e-9), qmax_p, steps=240) if qmax_p > 0 else [0.0]
                else:
                    qmax_guess = flow_to_m3h(100.0, flow_unit)
                    qs_m3h = make_curve_samples(1e-9, max(qmax_guess, 1e-6), steps=240)

                sys_dp_bar = [system_dp(q, dp0_bar, k_bar_per_m3h2) for q in qs_m3h]
                sys_Q_axis = [m3h_to_flow(q, flow_unit) for q in qs_m3h]
                sys_dp_axis = [bar_to_dp(dpb, dp_unit) for dpb in sys_dp_bar]
                curves.append({"name": "System curve", "pts": list(zip(sys_Q_axis, sys_dp_axis)), "stroke": "#2f855a", "width": 3})

            if plot_mode in ["Operating point only", "Pump + System + Operating point"]:
                if len(pump_Q_m3h) < 2:
                    raise ValueError("Need pump curve points to compute operating point.")
                inter = find_intersection_pump_vs_system(pump_Q_m3h, pump_dp_bar, dp0_bar, k_bar_per_m3h2)
                if inter is None:
                    st.session_state.pump_system_text += "\nIntersection: (none found in pump Q range)"
                else:
                    q_star_m3h, dp_star_bar = inter
                    mkx = m3h_to_flow(q_star_m3h, flow_unit)
                    mky = bar_to_dp(dp_star_bar, dp_unit)
                    markers.append({"name": "Operating point", "x": mkx, "y": mky, "color": "#ff7a00"})
                    st.session_state.pump_system_text += f"\nIntersection: Q = {mkx:.6g} {flow_unit},  ΔP = {mky:.6g} {dp_unit}"

            if plot_mode == "Operating point only":
                curves = []

            st.session_state.ps_svg = svg_plot(
                title="Pump PQ vs System Curve",
                xlabel=f"Flow [{flow_unit}]",
                ylabel=f"ΔP [{dp_unit}]",
                curves=curves,
                points=[],
                markers=markers,
                legend_loc=legend_loc
            )

        except Exception as e:
            st.error(str(e))

    with right:
        st.subheader("Results")
        st.text_area(
            "Output",
            value=st.session_state.get("results_text", "Click 'Calculate / Fit' to see results."),
            height=260
        )

        st.subheader("Cv PQ Plot (Measured + Fit)")
        if st.session_state.cv_svg:
            st.components.v1.html(st.session_state.cv_svg, height=560, scrolling=False)
        else:
            st.info("No Cv plot yet. Click **Plot Cv PQ**.")

        st.subheader("Pump/System Plot (Separate Diagram)")
        if st.session_state.pump_system_text:
            st.info(st.session_state.pump_system_text)
        else:
            st.info("System curve equation will show here after **Plot Pump/System** (or **Compute Intersection**).")

        if st.session_state.ps_svg:
            st.components.v1.html(st.session_state.ps_svg, height=560, scrolling=False)
        else:
            st.info("No pump/system plot yet. Click **Plot Pump/System**.")
