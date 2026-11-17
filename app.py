import csv
import io
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.integrate import solve_ivp
from dynamics import x, r, parse_polynomial, equilibria, stability, critical_parameters

st.set_page_config(page_title='Bifurcation Lab', page_icon='〰️', layout='wide')
COLORS = {'Stable': '#08916f', 'Unstable': '#e45c45', 'Semistable': '#ab75d1', 'Nonhyperbolic': '#d59c21'}
MODELS = {
    'Saddle-node': ('r - x^2', 'Two equilibria meet at r = 0, then disappear.'),
    'Transcritical': ('r*x - x^2', 'Two equilibrium branches exchange stability at r = 0.'),
    'Supercritical pitchfork': ('r*x - x^3', 'At r = 0, a stable equilibrium loses stability and two stable branches emerge.'),
    'Subcritical pitchfork': ('r*x + x^3', 'Two unstable branches meet the central stable branch at r = 0.'),
    'Custom polynomial': ('r*x - x^3 + 0.2', 'Explore an autonomous scalar equation dx/dt = f(x, r).'),
}

st.title('Bifurcation Lab')
st.write('Explore how equilibria and trajectories change as the parameter r varies.')
with st.sidebar:
    st.header('Equation & controls')
    model = st.selectbox('Model', list(MODELS))
    source, description = MODELS[model]
    if model == 'Custom polynomial':
        source = st.text_area('f(x, r)', source, help='Example: (r-1)*x - x^3. Use * for multiplication; ^ or ** for powers. No t dependence.')
    st.caption(description)
    a, b = st.columns(2)
    lower = a.number_input('r minimum', value=-2.0, step=0.5)
    upper = b.number_input('r maximum', value=2.0, step=0.5)
    ymin = a.number_input('x minimum', value=-3.0, step=0.5)
    ymax = b.number_input('x maximum', value=3.0, step=0.5)
    if not all(np.isfinite([lower, upper, ymin, ymax])) or lower >= upper or ymin >= ymax:
        st.error('Use finite, increasing r and x bounds.')
        st.stop()
    parameter = st.slider('Current r', float(lower), float(upper), float(np.clip(0.0, lower, upper)), format='%.4f')
    resolution = st.select_slider('Diagram resolution', options=[201, 401, 801], value=401)
    horizon = st.slider('Final time t', 1, 50, 12)
    initial = st.slider('Highlighted initial value x(0)', float(ymin), float(ymax), float(np.clip(0.5, ymin, ymax)))

try:
    expression = parse_polynomial(source)
except (ValueError, TypeError, sp.PolynomialError, OverflowError) as exc:
    st.error(str(exc))
    st.stop()
st.latex(r'\frac{dx}{dt} = ' + sp.latex(expression))

@st.cache_data(max_entries=16, show_spinner='Computing equilibria…')
def diagram_data(source, lower, upper, ymin, ymax, resolution):
    expr = parse_polynomial(source)
    candidates = critical_parameters(expr, lower, upper)
    parameters = np.unique(np.r_[np.linspace(lower, upper, resolution), candidates, 0.0 if lower <= 0 <= upper else lower])
    # Compile coefficient functions once for the entire parameter sweep.
    polynomial = sp.Poly(expr, x)
    functions = [sp.lambdify(r, c, 'numpy') for c in polynomial.all_coeffs()]
    derivative = sp.lambdify((x, r), sp.diff(expr, x), 'numpy')
    rows, continua = [], []
    for rv in parameters:
        coeff = np.trim_zeros(np.array([float(f(rv)) for f in functions]), 'f')
        if not len(coeff):
            continua.append(float(rv))
            continue
        roots = np.polynomial.polynomial.polyroots(coeff[::-1]) if len(coeff) > 1 else []
        seen = []
        for z in sorted(roots, key=lambda z: z.real):
            if abs(z.imag) > 1e-6 * (1 + abs(z.real)):
                continue
            xv = float(z.real)
            if not ymin <= xv <= ymax or any(abs(xv-v) < 1e-5 * (1+abs(xv)) for v in seen):
                continue
            seen.append(xv)
            slope = float(derivative(xv, rv))
            label = ('Stable' if slope < 0 else 'Unstable') if abs(slope) > 1e-7 else stability(expr, rv, xv)
            rows.append({'r': float(rv), 'x': xv, 'stability': label})
    return rows, continua, candidates

try:
    rows, continua, candidates = diagram_data(source, lower, upper, ymin, ymax, resolution)
    current = equilibria(expression, parameter)
except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
    st.error(f'Numerical calculation failed: {exc}. Try smaller coefficients or a narrower range.')
    st.stop()

left, right = st.columns([1.4, 1])
with left:
    st.subheader('Bifurcation diagram')
    fig = go.Figure()
    for label, color in COLORS.items():
        points = [row for row in rows if row['stability'] == label]
        if points:
            fig.add_trace(go.Scatter(x=[p['r'] for p in points], y=[p['x'] for p in points], mode='markers', name=label, marker=dict(color=color, size=4), hovertemplate='r=%{x:.5g}<br>x=%{y:.5g}<extra>'+label+'</extra>'))
    for rv in continua:
        fig.add_shape(type='line', x0=rv, x1=rv, y0=ymin, y1=ymax, line=dict(color='#d59c21', dash='dot'))
    fig.add_vline(x=parameter, line_dash='dash', line_color='#8290a4')
    fig.update_layout(xaxis_title='r', yaxis_title='Equilibrium x', xaxis_range=[lower, upper], yaxis_range=[ymin, ymax], height=470, margin=dict(t=20), legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Equilibria satisfy f(x, r) = 0. The dashed line marks the current r. Dotted gold lines indicate an entire line of equilibria.')
with right:
    st.subheader('Vector field & phase line')
    f = sp.lambdify((x, r), expression, 'numpy')
    xx = np.linspace(ymin, ymax, 500)
    with np.errstate(over='ignore', invalid='ignore'):
        yy = np.broadcast_to(np.asarray(f(xx, parameter), dtype=float), xx.shape)
    phase = go.Figure(go.Scatter(x=xx, y=yy, mode='lines', name='f(x, r)', line_color='#527cda'))
    phase.add_hline(y=0, line_color='#8290a4')
    for root in current or []:
        if ymin <= root <= ymax:
            label = stability(expression, parameter, root)
            phase.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name=label, marker=dict(color=COLORS[label], size=11), showlegend=False))
    for position in np.linspace(ymin, ymax, 19)[1:-1]:
        value = float(f(position, parameter))
        if np.isfinite(value) and abs(value) > 1e-8:
            direction = np.sign(value)
            phase.add_annotation(x=position+direction*(ymax-ymin)*0.018, y=0, ax=position-direction*(ymax-ymin)*0.018, ay=0, xref='x', yref='y', axref='x', ayref='y', text='', showarrow=True, arrowhead=2, arrowcolor='#8a94a5')
    phase.update_layout(xaxis_title='x', yaxis_title='dx/dt', xaxis_range=[ymin, ymax], height=470, margin=dict(t=20), showlegend=False)
    st.plotly_chart(phase, use_container_width=True)

st.subheader(f'Solutions x(t) at r = {parameter:.4g}')
trajectories = go.Figure()
failures = 0
for x0 in np.unique(np.r_[np.linspace(ymin, ymax, 9)[1:-1], initial]):
    def rhs(t, state):
        return [float(f(state[0], parameter))]
    def exit_window(t, state):
        return min(state[0]-ymin, ymax-state[0])
    exit_window.terminal = True
    exit_window.direction = -1
    result = solve_ivp(rhs, (0, horizon), [x0], events=exit_window, rtol=1e-7, atol=1e-9, max_step=horizon/150)
    failures += not result.success
    highlight = bool(abs(x0-initial) < 1e-10)
    trajectories.add_trace(go.Scatter(x=result.t, y=result.y[0], mode='lines', name=f'x(0) = {x0:.3g}', line=dict(color='#527cda' if highlight else '#9eabc0', width=3 if highlight else 1), showlegend=highlight))
trajectories.update_layout(xaxis_title='t', yaxis_title='x(t)', xaxis_range=[0, horizon], yaxis_range=[ymin, ymax], height=340, margin=dict(t=15))
st.plotly_chart(trajectories, use_container_width=True)
st.caption('Trajectories stop when they leave the displayed x interval. A highlighted curve uses your chosen initial value.')
if failures:
    st.warning('Some integrations could not finish. Try a shorter time interval or smaller coefficients.')

with st.expander('Equilibria at the current r', expanded=True):
    if current is None:
        st.info('f(x, r) is identically zero at this r: every x is an equilibrium (neutral).')
    elif not current:
        st.info('No real equilibria at this r.')
    else:
        st.dataframe([{'x': root, 'stability': stability(expression, parameter, root), 'in displayed interval': ymin <= root <= ymax} for root in current], hide_index=True, use_container_width=True)

with st.expander('How to read the plots & numerical limits'):
    st.markdown('''**Stable** equilibria attract nearby trajectories; **unstable** equilibria repel them. **Semistable** equilibria attract from one side and repel from the other. Stability at a multiple root uses the first nonzero term in the local polynomial expansion.

The custom mode accepts autonomous polynomials in **x** and **r**, with total degree at most 20. **t** appears only as the independent variable in x(t). Examples: `r - x^2`, `r*x - x^3`, `(r-1)*x + x^3 - x^5`.

The diagram samples r and computes real polynomial roots numerically. Very close roots, large coefficients, or narrow branches can be missed; increase resolution or zoom the bounds. For total degree at most 8, additional parameter samples come from the discriminant and leading coefficient. These are candidate changes, not an automatic classification of bifurcations. Higher-degree diagrams use the regular grid. No finite scan guarantees every feature.''')

buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=['r', 'x', 'stability'])
writer.writeheader()
writer.writerows(rows)
st.download_button('Download equilibrium samples (CSV)', buffer.getvalue(), 'equilibria.csv', 'text/csv')
st.caption('Inspired by Bifurcations in First-Order ODEs, Wolfram Demonstrations Project: https://demonstrations.wolfram.com/BifurcationsInFirstOrderODEs/')
