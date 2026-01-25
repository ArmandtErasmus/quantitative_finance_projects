import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from american_barrier_option import AmericanBarrierOption

st.set_page_config(
    page_title="American Barrier Option Pricing Engine",
    page_icon="📊",
    layout="wide"
)

st.title("American Barrier Option Pricing: Monte-Carlo Regression")

with st.sidebar:
    st.header("Market Parameters")
    S0 = st.number_input("Initial Asset Price", 50.0, 200.0, 100.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
    sigma = st.slider("Volatility (%)", 5.0, 50.0, 20.0, 1.0) / 100
    dividend = st.slider("Dividend Yield (%)", 0.0, 5.0, 0.0, 0.1) / 100
    
    st.header("Option Parameters")
    K = st.number_input("Strike Price", 50.0, 200.0, 100.0, 1.0)
    T = st.slider("Time to Maturity (years)", 0.1, 2.0, 1.0, 0.1)
    option_type = st.selectbox("Option Type", ["call", "put"])
    
    st.header("Barrier Parameters")
    barrier_type = st.selectbox("Barrier Type", 
                               ["up-and-out", "down-and-out", "up-and-in", "down-and-in"])
    if barrier_type in ['up-and-out', 'up-and-in']:
        barrier = st.number_input("Barrier Level", S0, S0 * 2.0, S0 * 1.3, 1.0)
    else:
        barrier = st.number_input("Barrier Level", S0 * 0.5, S0, S0 * 0.7, 1.0)
    
    st.header("Simulation Settings")
    num_paths = st.slider("Monte Carlo Paths", 1000, 50000, 10000, 1000)
    num_steps = st.slider("Time Steps", 50, 500, 252, 10)
    basis_degree = st.slider("Basis Function Degree", 1, 5, 3, 1)
    seed = st.number_input("Random Seed (0 for random)", 0, 1000000, 42, 1)

option = AmericanBarrierOption(
    S0=S0, K=K, T=T, r=r, sigma=sigma, barrier=barrier,
    barrier_type=barrier_type, option_type=option_type, dividend=dividend
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    moneyness = S0 / K
    st.metric("Moneyness", f"{moneyness:.3f}")
with col2:
    barrier_ratio = barrier / S0
    st.metric("Barrier Ratio", f"{barrier_ratio:.3f}")
with col3:
    time_value_pct = (T * 365) / 365
    st.metric("Time to Maturity", f"{T:.2f} years")
with col4:
    vol_annualised = sigma * np.sqrt(T)
    st.metric("Total Volatility", f"{vol_annualised*100:.2f}%")

if st.button("Run Pricing Simulation", type="primary"):
    with st.spinner("Running Monte Carlo regression..."):
        seed_val = None if seed == 0 else int(seed)
        american_price, american_se = option.price_lsm(
            num_paths=num_paths, num_steps=num_steps,
            basis_degree=basis_degree, seed=seed_val
        )
        
        european_price, european_se = option.price_european_barrier(
            num_paths=num_paths, seed=seed_val
        )
        
        st.subheader("Pricing Results")
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        with result_col1:
            st.metric("American Option Price", f"${american_price:.4f}",
                     delta=f"±${1.96*american_se:.4f}")
        with result_col2:
            st.metric("European Option Price", f"${european_price:.4f}",
                     delta=f"±${1.96*european_se:.4f}")
        with result_col3:
            early_exercise_premium = american_price - european_price
            st.metric("Early Exercise Premium", f"${early_exercise_premium:.4f}",
                     delta=f"{(early_exercise_premium/european_price*100):.2f}%")
        with result_col4:
            st.metric("95% Confidence Interval", 
                     f"[${american_price-1.96*american_se:.4f}, ${american_price+1.96*american_se:.4f}]")
        
        S, M = option.simulate_paths(num_paths, num_steps, seed_val)
        time_grid = np.linspace(0, T, num_steps + 1)
        
        barrier_hit = np.zeros(num_paths, dtype=bool)
        if barrier_type == 'up-and-out' or barrier_type == 'up-and-in':
            barrier_hit = np.any(M >= barrier, axis=1)
        else:
            barrier_hit = np.any(M <= barrier, axis=1)
        
        barrier_hit_rate = np.mean(barrier_hit) * 100
        
        st.subheader("Path Analysis")
        path_col1, path_col2 = st.columns(2)
        
        with path_col1:
            st.metric("Barrier Hit Rate", f"{barrier_hit_rate:.2f}%")
        with path_col2:
            final_prices = S[:, -1]
            itm_rate = np.mean((final_prices > K) if option_type == 'call' 
                              else (final_prices < K)) * 100
            st.metric("Final ITM Rate", f"{itm_rate:.2f}%")
        
        st.subheader("Simulated Asset Price Paths")
        fig_paths = go.Figure()
        
        sample_paths = min(200, num_paths)
        alive_paths = np.where(~barrier_hit)[0][:sample_paths]
        hit_paths = np.where(barrier_hit)[0][:min(sample_paths, np.sum(barrier_hit))]
        
        for idx in alive_paths:
            fig_paths.add_trace(go.Scatter(
                x=time_grid,
                y=S[idx, :],
                mode='lines',
                line=dict(width=0.5, color='rgba(104, 101, 242, 0.15)'),
                showlegend=False
            ))
        
        for idx in hit_paths:
            hit_time_idx = np.argmax(M[idx, :] >= barrier) if barrier_type in ['up-and-out', 'up-and-in'] else np.argmax(M[idx, :] <= barrier)
            fig_paths.add_trace(go.Scatter(
                x=time_grid[:hit_time_idx+1],
                y=S[idx, :hit_time_idx+1],
                mode='lines',
                line=dict(width=0.5, color='rgba(255, 107, 107, 0.3)'),
                showlegend=False
            ))
        
        mean_path = np.mean(S[~barrier_hit, :], axis=0) if np.sum(~barrier_hit) > 0 else np.mean(S, axis=0)
        std_path = np.std(S[~barrier_hit, :], axis=0) if np.sum(~barrier_hit) > 0 else np.std(S, axis=0)
        
        fig_paths.add_trace(go.Scatter(
            x=time_grid,
            y=mean_path,
            mode='lines',
            name='Mean Path (Alive)',
            line=dict(width=2, color='#6865F2')
        ))
        
        fig_paths.add_hline(y=barrier, line_dash="dash", line_color="red",
                           annotation_text=f"Barrier (B={barrier:.2f})")
        fig_paths.add_hline(y=K, line_dash="dot", line_color="orange",
                           annotation_text=f"Strike (K={K:.2f})")
        
        fig_paths.update_layout(
            title="Simulated Asset Price Paths with Barrier",
            xaxis_title="Time (years)",
            yaxis_title="Asset Price",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_paths, use_container_width=True)
        
        st.subheader("Distribution Analysis")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            fig_hist_final = go.Figure()
            fig_hist_final.add_trace(go.Histogram(
                x=S[~barrier_hit, -1] if np.sum(~barrier_hit) > 0 else S[:, -1],
                nbinsx=50,
                marker_color='#6865F2',
                opacity=0.7,
                name='Final Prices (Alive)'
            ))
            if np.sum(barrier_hit) > 0:
                fig_hist_final.add_trace(go.Histogram(
                    x=S[barrier_hit, -1],
                    nbinsx=50,
                    marker_color='#FF6B6B',
                    opacity=0.7,
                    name='Final Prices (Knocked Out)'
                ))
            fig_hist_final.add_vline(x=K, line_dash="dash", line_color="orange",
                                    annotation_text=f"Strike (K={K:.2f})")
            fig_hist_final.update_layout(
                title="Distribution of Final Asset Prices",
                xaxis_title="Asset Price",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_hist_final, use_container_width=True)
        
        with dist_col2:
            max_prices = M[:, -1] if barrier_type in ['up-and-out', 'up-and-in'] else np.minimum.accumulate(S, axis=1)[:, -1]
            fig_hist_max = go.Figure()
            fig_hist_max.add_trace(go.Histogram(
                x=max_prices,
                nbinsx=50,
                marker_color='#5DFFBC',
                opacity=0.7,
                name='Maximum Prices'
            ))
            fig_hist_max.add_vline(x=barrier, line_dash="dash", line_color="red",
                                   annotation_text=f"Barrier (B={barrier:.2f})")
            fig_hist_max.update_layout(
                title="Distribution of Maximum Asset Prices",
                xaxis_title="Maximum Price",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_hist_max, use_container_width=True)
        
        st.subheader("Sensitivity Analysis")
        sens_col1, sens_col2 = st.columns(2)
        
        with sens_col1:
            strike_range = np.linspace(K * 0.7, K * 1.3, 20)
            prices_strike = []
            for k in strike_range:
                opt = AmericanBarrierOption(S0, k, T, r, sigma, barrier, 
                                           barrier_type, option_type, dividend)
                p, _ = opt.price_lsm(num_paths=5000, num_steps=num_steps,
                                     basis_degree=basis_degree, seed=seed_val)
                prices_strike.append(p)
            
            fig_strike = go.Figure()
            fig_strike.add_trace(go.Scatter(
                x=strike_range,
                y=prices_strike,
                mode='lines+markers',
                line=dict(color='#6865F2', width=2),
                name='Option Price'
            ))
            fig_strike.add_vline(x=K, line_dash="dash", line_color="gray",
                                annotation_text=f"Current Strike")
            fig_strike.update_layout(
                title="Price Sensitivity to Strike",
                xaxis_title="Strike Price",
                yaxis_title="Option Price",
                height=400
            )
            st.plotly_chart(fig_strike, use_container_width=True)
        
        with sens_col2:
            barrier_range = np.linspace(S0 * 0.8, S0 * 1.5, 20) if barrier_type in ['up-and-out', 'up-and-in'] else np.linspace(S0 * 0.5, S0 * 0.95, 20)
            prices_barrier = []
            for b in barrier_range:
                opt = AmericanBarrierOption(S0, K, T, r, sigma, b, 
                                           barrier_type, option_type, dividend)
                p, _ = opt.price_lsm(num_paths=5000, num_steps=num_steps,
                                    basis_degree=basis_degree, seed=seed_val)
                prices_barrier.append(p)
            
            fig_barrier = go.Figure()
            fig_barrier.add_trace(go.Scatter(
                x=barrier_range,
                y=prices_barrier,
                mode='lines+markers',
                line=dict(color='#5DFFBC', width=2),
                name='Option Price'
            ))
            fig_barrier.add_vline(x=barrier, line_dash="dash", line_color="gray",
                                 annotation_text=f"Current Barrier")
            fig_barrier.update_layout(
                title="Price Sensitivity to Barrier",
                xaxis_title="Barrier Level",
                yaxis_title="Option Price",
                height=400
            )
            st.plotly_chart(fig_barrier, use_container_width=True)
        
        st.subheader("Convergence Analysis")
        convergence_paths = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 50000]
        convergence_paths = [p for p in convergence_paths if p <= num_paths]
        
        prices_conv = []
        std_errs_conv = []
        
        for n_paths in convergence_paths:
            p, se = option.price_lsm(num_paths=n_paths, num_steps=num_steps,
                                    basis_degree=basis_degree, seed=seed_val)
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
        fig_conv.add_hline(y=american_price, line_dash="dash", line_color="gray",
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
