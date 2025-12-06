import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from one_period_binomial_model import OnePeriodBinomialPricer, BinomialVisualizer
import plotly.graph_objects as go
import plotly 

class BinomialDashboard:

    def __init__(self):
        st.set_page_config(page_title="One-Period Binomial Model", layout="wide")
        self._sidebar()

    def _sidebar(self):
        st.sidebar.title("Model Parameters")

        self.S0 = st.sidebar.number_input("S0 (Spot Price)", min_value=0.01, value=100.0)
        self.K = st.sidebar.number_input("K (Strike Price)", min_value=0.01, value=100.0)
        self.u = st.sidebar.number_input("u (Up Factor)", min_value=0.01, value=1.1)
        self.d = st.sidebar.number_input("d (Down Factor)", min_value=0.01, value=0.9)
        self.r = st.sidebar.number_input("r (Risk-free rate)", min_value=-0.99, value=0.02)

        self.option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    def _compute_result(self):
        pricer = OnePeriodBinomialPricer(
            S0=self.S0,
            K=self.K,
            u=self.u,
            d=self.d,
            r=self.r,
            validate=False
        )

        if self.option_type == "Call":
            return pricer.price_call()
        else:
            return pricer.price_put()

    def run(self):
        st.title("One-Period Binomial Option Pricing Dashboard")

        result = self._compute_result()

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"{result.price:.4f}")
        col2.metric("Delta", f"{result.delta:.4f}")
        col3.metric("Risk-Neutral Prob", f"{result.risk_neutral_prob:.4f}")

        col4, col5 = st.columns(2)
        col4.metric("Bond Position", f"{result.bond_position:.4f}")
        col5.metric("Replication Error", f"{result.replication_error:.6f}")

        # Re-instantiate visualizer with latest inputs
        vis = BinomialVisualizer(self.S0, self.K, self.r)

        # Tabs for plots
        tab1, tab2, tab3 = st.tabs([
            "Price Surface",
            "Delta vs Strike",
            "Arbitrage Region"
        ])

        with tab1:
            fig1 = vis.plot_price_surface_interactive(u_range=(self.u, 1.4), d_range=(0.6, self.d))
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            fig2 = vis.plot_delta_vs_strike(K_min=self.S0 * 0.5, K_max=self.S0 * 1.5, u=self.u, d=self.d)
            with col_center:
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            fig3 = vis.plot_arbitrage_region(u_range=(0.6, 1.4), d_range=(0.6, 1.0))
            with col_center:
                st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    app = BinomialDashboard()
    app.run()
