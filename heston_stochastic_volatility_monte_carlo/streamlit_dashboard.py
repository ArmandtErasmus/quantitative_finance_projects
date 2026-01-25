import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from heston_model import HestonModel, HestonOptionPricer

st.set_page_config(
    page_title="Heston Stochastic Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Heston Stochastic Volatility Model")

with st.sidebar:
    st.header("Model Parameters")
    kappa = st.slider("Mean Reversion (κ)", 0.5, 10.0, 2.0, 0.1)
    theta = st.slider("Long-term Variance (θ)", 0.01, 0.25, 0.04, 0.01)
    sigma_v = st.slider("Vol of Vol (σᵥ)", 0.1, 1.0, 0.3, 0.05)
    rho = st.slider("Correlation (ρ)", -0.9, 0.9, -0.7, 0.1)
    v0 = st.slider("Initial Variance (v₀)", 0.01, 0.25, 0.04, 0.01)
    
    st.header("Market Parameters")
    S0 = st.number_input("Initial Stock Price", 50, 200, 100)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    
    st.header("Simulation Settings")
    num_paths = st.slider("Monte Carlo Paths", 1000, 50000, 10000)
    n_steps = st.slider("Time Steps", 50, 500, 252)
    T = st.slider("Time Horizon (years)", 0.1, 2.0, 1.0, 0.1)
    
    st.header("Option Parameters")
    K = st.number_input("Strike Price", 50, 200, 100)
    option_type = st.selectbox("Option Type", 
                               ["European Call", "Asian Call", "Barrier Up-and-Out Call"])
    if option_type == "Barrier Up-and-Out Call":
        B = st.number_input("Barrier Level", S0, S0 * 2, int(S0 * 1.5))

model = HestonModel(kappa, theta, sigma_v, rho, v0, r)
pricer = HestonOptionPricer(model)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mean Reversion Speed", f"{kappa:.2f}")
with col2:
    st.metric("Long-term Volatility", f"{np.sqrt(theta)*100:.2f}%")
with col3:
    st.metric("Vol of Vol", f"{sigma_v:.2f}")
with col4:
    st.metric("Correlation", f"{rho:.2f}")

if st.button("Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        S, v = model.simulate_path_exact(S0, T, n_steps, num_paths)
        
        if option_type == "European Call":
            price, std_err = pricer.model.price_european_call_mc(S0, K, T, num_paths, n_steps)
        elif option_type == "Asian Call":
            price, std_err = pricer.price_asian_call(S0, K, T, num_paths, n_steps)
        else:
            price, std_err = pricer.price_barrier_up_and_out_call(S0, K, B, T, num_paths, n_steps)
        
        st.subheader("Pricing Results")
        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("Option Price", f"${price:.4f}")
        with result_col2:
            st.metric("Standard Error", f"${std_err:.4f}")
        with result_col3:
            st.metric("95% Confidence Interval", 
                     f"[${price-1.96*std_err:.4f}, ${price+1.96*std_err:.4f}]")
        
        time_grid = np.linspace(0, T, n_steps + 1)
        
        st.subheader("Simulated Price Paths")
        fig_paths = go.Figure()
        
        sample_paths = min(100, num_paths)
        for i in range(sample_paths):
            fig_paths.add_trace(go.Scatter(
                x=time_grid,
                y=S[i, :],
                mode='lines',
                line=dict(width=0.5, color='rgba(104, 101, 242, 0.1)'),
                showlegend=False
            ))
        
        mean_path = np.mean(S, axis=0)
        std_path = np.std(S, axis=0)
        
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(width=2, color='#6865F2')
        ))
        
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path + 1.96 * std_path,
            mode='lines',
            name='95% Upper Bound',
            line=dict(width=1, color='#5DFFBC', dash='dash'),
            showlegend=True
        ))
        
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path - 1.96 * std_path,
            mode='lines',
            name='95% Lower Bound',
            line=dict(width=1, color='#5DFFBC', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(93, 255, 188, 0.1)'
        ))
        
        fig_paths.update_layout(
            title="Simulated Asset Price Paths",
            xaxis_title="Time (years)",
            yaxis_title="Asset Price",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_paths, use_container_width=True)
        
        st.subheader("Simulated Variance Paths")
        fig_var = go.Figure()
        
        for i in range(sample_paths):
            fig_var.add_trace(go.Scatter(
                x=time_grid,
                y=v[i, :],
                mode='lines',
                line=dict(width=0.5, color='rgba(255, 107, 107, 0.1)'),
                showlegend=False
            ))
        
        mean_var = np.mean(v, axis=0)
        
        fig_var.add_trace(go.Scatter(
            x=time_grid,
            y=mean_var,
            mode='lines',
            name='Mean Variance',
            line=dict(width=2, color='#FF6B6B')
        ))
        
        fig_var.add_hline(y=theta, line_dash="dash", line_color="gray",
                         annotation_text=f"Long-term Variance (θ={theta:.4f})")
        
        fig_var.update_layout(
            title="Simulated Variance Paths",
            xaxis_title="Time (years)",
            yaxis_title="Variance",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        st.subheader("Distribution Analysis")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            fig_hist_price = go.Figure()
            fig_hist_price.add_trace(go.Histogram(
                x=S[:, -1],
                nbinsx=50,
                marker_color='#6865F2',
                opacity=0.7,
                name='Final Prices'
            ))
            fig_hist_price.add_vline(x=K, line_dash="dash", line_color="red",
                                    annotation_text=f"Strike (K={K})")
            fig_hist_price.update_layout(
                title="Distribution of Final Asset Prices",
                xaxis_title="Asset Price",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_hist_price, use_container_width=True)
        
        with dist_col2:
            fig_hist_var = go.Figure()
            fig_hist_var.add_trace(go.Histogram(
                x=v[:, -1],
                nbinsx=50,
                marker_color='#FF6B6B',
                opacity=0.7,
                name='Final Variances'
            ))
            fig_hist_var.add_vline(x=theta, line_dash="dash", line_color="gray",
                                  annotation_text=f"Long-term (θ={theta:.4f})")
            fig_hist_var.update_layout(
                title="Distribution of Final Variances",
                xaxis_title="Variance",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_hist_var, use_container_width=True)
        
        st.subheader("Convergence Analysis")
        convergence_paths = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 50000]
        convergence_paths = [p for p in convergence_paths if p <= num_paths]
        
        prices_conv = []
        std_errs_conv = []
        
        for n_paths in convergence_paths:
            if option_type == "European Call":
                p, se = pricer.model.price_european_call_mc(S0, K, T, n_paths, n_steps)
            elif option_type == "Asian Call":
                p, se = pricer.price_asian_call(S0, K, T, n_paths, n_steps)
            else:
                p, se = pricer.price_barrier_up_and_out_call(S0, K, B, T, n_paths, n_steps)
            prices_conv.append(p)
            std_errs_conv.append(se)
        
        fig_conv = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price Convergence", "Standard Error Convergence"),
            vertical_spacing=0.1
        )
        
        fig_conv.add_trace(
            go.Scatter(x=convergence_paths, y=prices_conv, mode='lines+markers',
                      name='Option Price', line=dict(color='#6865F2', width=2)),
            row=1, col=1
        )
        fig_conv.add_hline(y=price, line_dash="dash", line_color="gray",
                          annotation_text="Final Price", row=1, col=1)
        
        fig_conv.add_trace(
            go.Scatter(x=convergence_paths, y=std_errs_conv, mode='lines+markers',
                      name='Standard Error', line=dict(color='#FF6B6B', width=2)),
            row=2, col=1
        )
        
        fig_conv.update_xaxes(title_text="Number of Paths", row=2, col=1)
        fig_conv.update_yaxes(title_text="Price", row=1, col=1)
        fig_conv.update_yaxes(title_text="Standard Error", row=2, col=1)
        fig_conv.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig_conv, use_container_width=True)
