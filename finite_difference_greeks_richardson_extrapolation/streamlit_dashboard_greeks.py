import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from finite_difference_greeks import FiniteDifferenceGreeks
from scipy.stats import norm

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def pricing_function(params: dict) -> float:
    return black_scholes_call(
        S=params.get('S', 100),
        K=params.get('K', 100),
        T=params.get('T', 1.0),
        r=params.get('r', 0.05),
        sigma=params.get('sigma', 0.2)
    )

st.set_page_config(
    page_title="Finite-Difference Greeks Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Robust Finite-Difference Greeks with Richardson Extrapolation")

with st.sidebar:
    st.header("Option Parameters")
    S = st.number_input("Spot Price (S)", 50.0, 200.0, 100.0, 1.0)
    K = st.number_input("Strike Price (K)", 50.0, 200.0, 100.0, 1.0)
    T = st.slider("Time to Maturity (years)", 0.01, 2.0, 1.0, 0.01)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
    sigma = st.slider("Volatility (%)", 5.0, 50.0, 20.0, 1.0) / 100
    
    st.header("Greek Computation Settings")
    use_richardson = st.checkbox("Use Richardson Extrapolation", value=True)
    tolerance = st.slider("Convergence Tolerance", 1e-6, 1e-2, 1e-4, format="%.0e")
    min_step = st.number_input("Min Step Size", 1e-8, 1e-3, 1e-6, format="%.0e")
    max_step = st.number_input("Max Step Size", 1e-4, 0.5, 0.1, format="%.3f")
    
    st.header("Visualisation Settings")
    S_range_min = st.number_input("S Range Min", 50.0, 150.0, 50.0, 1.0)
    S_range_max = st.number_input("S Range Max", 50.0, 200.0, 150.0, 1.0)
    n_points = st.slider("Number of Points", 50, 500, 200, 10)

base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
greek_calculator = FiniteDifferenceGreeks(
    pricing_function=pricing_function,
    base_params=base_params,
    tolerance=tolerance,
    min_step=min_step,
    max_step=max_step
)

if st.button("Compute Greeks", type="primary"):
    with st.spinner("Computing Greeks..."):
        greeks = greek_calculator.compute_all_greeks(
            S=S, t=T, sigma=sigma, r=r, use_richardson=use_richardson
        )
        
        option_price = pricing_function(base_params)
        
        st.subheader("Greek Values")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta", f"{greeks['Delta'][0]:.6f}", 
                     delta=f"Error: {greeks['Delta'][1]:.2e}")
        with col2:
            st.metric("Gamma", f"{greeks['Gamma'][0]:.6f}",
                     delta=f"Error: {greeks['Gamma'][1]:.2e}")
        with col3:
            st.metric("Theta", f"{greeks['Theta'][0]:.6f}",
                     delta=f"Error: {greeks['Theta'][1]:.2e}")
        with col4:
            st.metric("Vega", f"{greeks['Vega'][0]:.6f}",
                     delta=f"Error: {greeks['Vega'][1]:.2e}")
        with col5:
            st.metric("Rho", f"{greeks['Rho'][0]:.6f}",
                     delta=f"Error: {greeks['Rho'][1]:.2e}")
        
        st.metric("Option Price", f"${option_price:.4f}")
        
        S_range = np.linspace(S_range_min, S_range_max, n_points)
        prices = [pricing_function({**base_params, 'S': s}) for s in S_range]
        
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        
        for s in S_range:
            d, _ = greek_calculator.compute_delta(s, use_richardson=use_richardson)
            g, _ = greek_calculator.compute_gamma(s, use_richardson=use_richardson)
            t, _ = greek_calculator.compute_theta(T, use_richardson=use_richardson)
            v, _ = greek_calculator.compute_vega(sigma, use_richardson=use_richardson)
            r_val, _ = greek_calculator.compute_rho(r, use_richardson=use_richardson)
            deltas.append(d)
            gammas.append(g)
            thetas.append(t)
            vegas.append(v)
            rhos.append(r_val)
        
        st.subheader("Price and Greeks Profiles")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Option Price", "Delta", "Gamma", "Theta", "Vega", "Rho"),
            vertical_spacing=0.08
        )
        
        fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines',
                                name='Price', line=dict(color='#6865F2', width=2)),
                     row=1, col=1)
        fig.add_vline(x=K, line_dash="dash", line_color="gray",
                     annotation_text="Strike", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines',
                                name='Delta', line=dict(color='#5DFFBC', width=2)),
                     row=1, col=2)
        
        fig.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines',
                                name='Gamma', line=dict(color='#FF6B6B', width=2)),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=S_range, y=thetas, mode='lines',
                                name='Theta', line=dict(color='#FFD93D', width=2)),
                     row=2, col=2)
        
        fig.add_trace(go.Scatter(x=S_range, y=vegas, mode='lines',
                                name='Vega', line=dict(color='#6BCF7F', width=2)),
                     row=3, col=1)
        
        fig.add_trace(go.Scatter(x=S_range, y=rhos, mode='lines',
                                name='Rho', line=dict(color='#4D96FF', width=2)),
                     row=3, col=2)
        
        fig.update_xaxes(title_text="Spot Price", row=3, col=1)
        fig.update_xaxes(title_text="Spot Price", row=3, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Delta", row=1, col=2)
        fig.update_yaxes(title_text="Gamma", row=2, col=1)
        fig.update_yaxes(title_text="Theta", row=2, col=2)
        fig.update_yaxes(title_text="Vega", row=3, col=1)
        fig.update_yaxes(title_text="Rho", row=3, col=2)
        
        fig.update_layout(height=900, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Convergence Analysis")
        step_sizes = np.logspace(-4, -1, 20)
        delta_standard = []
        delta_richardson = []
        
        for h in step_sizes:
            d_std, _ = greek_calculator.compute_delta(S, h, use_richardson=False)
            d_rich, _ = greek_calculator.compute_delta(S, h, use_richardson=True)
            delta_standard.append(d_std)
            delta_richardson.append(d_rich)
        
        analytical_delta = norm.cdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=step_sizes, y=np.abs(np.array(delta_standard) - analytical_delta),
                                      mode='lines+markers', name='Standard FD',
                                      line=dict(color='#FF6B6B', width=2)))
        fig_conv.add_trace(go.Scatter(x=step_sizes, y=np.abs(np.array(delta_richardson) - analytical_delta),
                                      mode='lines+markers', name='Richardson Extrapolation',
                                      line=dict(color='#6865F2', width=2)))
        fig_conv.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_conv.update_xaxes(type="log", title_text="Step Size (h)")
        fig_conv.update_yaxes(type="log", title_text="Absolute Error")
        fig_conv.update_layout(title="Delta Convergence: Standard vs Richardson",
                              height=400)
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.subheader("Error Analysis")
        error_col1, error_col2 = st.columns(2)
        
        with error_col1:
            errors = [greeks['Delta'][1], greeks['Gamma'][1], 
                     greeks['Theta'][1], greeks['Vega'][1], greeks['Rho'][1]]
            greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Bar(x=greek_names, y=errors,
                                        marker_color='#6865F2', opacity=0.7))
            fig_errors.update_layout(title="Estimated Errors by Greek",
                                    xaxis_title="Greek",
                                    yaxis_title="Estimated Error",
                                    height=400)
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with error_col2:
            analytical_greeks = {
                'Delta': analytical_delta,
                'Gamma': norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) / (S * sigma * np.sqrt(T)),
                'Theta': -(S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))),
                'Vega': S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T),
                'Rho': K * T * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
            }
            
            computed_values = [greeks['Delta'][0], greeks['Gamma'][0],
                             greeks['Theta'][0], greeks['Vega'][0], greeks['Rho'][0]]
            analytical_values = [analytical_greeks['Delta'], analytical_greeks['Gamma'],
                               analytical_greeks['Theta'], analytical_greeks['Vega'], analytical_greeks['Rho']]
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(x=greek_names, y=computed_values,
                                           name='Computed', marker_color='#6865F2', opacity=0.7))
            fig_comparison.add_trace(go.Bar(x=greek_names, y=analytical_values,
                                           name='Analytical', marker_color='#5DFFBC', opacity=0.7))
            fig_comparison.update_layout(title="Computed vs Analytical Greeks",
                                       xaxis_title="Greek",
                                       yaxis_title="Value",
                                       height=400,
                                       barmode='group')
            st.plotly_chart(fig_comparison, use_container_width=True)
