# importing necessary libraries
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly 
from typing import Tuple, Dict
from dataclasses import dataclass
import json
from datetime import datetime 
from typing import Callable


# defining the binomial result dataclass
@dataclass
class BinomialResult:
    price: float
    delta: float 
    bond_position: float
    risk_neutral_prob: float 
    is_arbitrage: bool 
    replication_error: float 

# defining the one-period binomial model pricer and set up input validation
class OnePeriodBinomialPricer:

    def __init__(self, S0: float, K: float, u: float, d: float, r: float, validate=True):
        self.S0 = self._validate_positive(S0, "S0")
        self.K = self._validate_positive(K, "K")
        self.u = self._validate_multiplier(u, "u")
        self.d = self._validate_multiplier(d, "d")
        self.r = self._validate_rate(r)

        if validate:
            self._validate_no_arbitrage()

    def _validate_positive(self, value: float, name: str) -> float:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return float(value)
    
    def _validate_multiplier(self, value: float, name: str) -> float:
        if value <= 0:
            raise ValueError(f"Multiplier {name} must be positive, got {value}")
        return float(value)
    
    def _validate_rate(self, r: float) -> float:
        if r <= -1:
            raise ValueError(f"Rate r must be > -1, got {r}")
        return float(r)
    
    def _validate_no_arbitrage(self):
        eps = 1e-10
        if not (self.d < np.exp(self.r) - eps < self.u):
            import warnings
            warnings.warn(f"Arbitrage condition violated: d={self.d} < e^r={np.exp(self.r):.4f} < u={self.u} expected")

    def price_call(self) -> BinomialResult:

        # terminal payoffs
        Su = self.S0 * self.u 
        Sd = self.S0 * self.d 
        Cu = max(Su - self.K, 0.0)
        Cd = max(Sd - self.K, 0.0)

        # risk neutral prob
        q = (np.exp(self.r) - self.d) / (self.u - self.d)

        # price and hedge ratio
        price = np.exp(-self.r) * (q * Cu + (1-q) * Cd)
        delta = (Cu - Cd) / (Su -Sd)
        bond = (Su * Cd - Sd * Cu) / ((Su - Sd) * np.exp(self.r))

        # replication error analysis
        portfolio_up = delta * Su + bond * np.exp(self.r)
        portfolio_down = delta * Sd + bond * np.exp(self.r)
        replication_error = max(abs(portfolio_up - Cu), abs(portfolio_down - Cd))

        return BinomialResult(
            price = price,
            delta = delta,
            bond_position = bond,
            risk_neutral_prob = q,
            is_arbitrage = not (0 < q < 1),
            replication_error = replication_error
        )
    
    def price_put(self) -> BinomialResult:

        call_result = self.price_call()
        forward = self.S0 - self.K * np.exp(-self.r)
        put_price = call_result.price - forward

        # terminal payoffs
        Su = self.S0 * self.u 
        Sd = self.S0 * self.d 
        Pu = max(self.K - Su, 0.0)
        Pd = max(self.K - Sd, 0.0)
        delta = (Pu - Pd) / (Su - Sd)


        return BinomialResult(
            price = put_price,
            delta = delta,
            bond_position = call_result.bond_position + self.K * np.exp(-self.r),
            risk_neutral_prob = call_result.risk_neutral_prob,
            is_arbitrage = call_result.is_arbitrage,
            replication_error = call_result.replication_error
        )
    
    def to_json(self, result: BinomialResult) -> str:
        output = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "S0": self.S0, "K": self.K, "u": self.u, "d": self.d, "r": self.r
            },
            "results": {
                "price": result.price,
                "delta": result.delta,
                "bond_position": result.bond_position,
                "risk_neutral_prob": result.risk_neutral_prob,
                "is_arbitrage": result.is_arbitrage,
                "replication_error": result.replication_error
            }
        }
        return json.dumps(output, indent = 2)
    
class BinomialVisualizer:

    def __init__(self, S0: float, K: float, r: float):
        self.S0 = S0
        self.K = K
        self.r = r

    def plot_delta_vs_strike(self,
                             K_min=50,
                             K_max=150,
                             num=100,
                             u=1.1,
                             d=0.9):

        strikes = np.linspace(K_min, K_max, num)
        deltas = np.zeros_like(strikes)

        for i, K in enumerate(strikes):
            pricer = OnePeriodBinomialPricer(
                S0=self.S0, K=K, u=u, d=d, r=self.r, validate=False
            )
            deltas[i] = pricer.price_call().delta

        fig = go.Figure()
        # Delta line
        fig.add_trace(go.Scatter(
            x=strikes,
            y=deltas,
            mode="lines",
            line=dict(width=2),
            name="Delta"
        ))

        # ATM vertical line
        fig.add_trace(go.Scatter(
            x=[self.S0, self.S0],
            y=[min(deltas), max(deltas)],
            mode="lines",
            line=dict(dash="dash"),
            name="ATM (S0)"
        ))

        fig.update_layout(
            title="Delta Sensitivity to Strike K",
            xaxis_title="Strike K",
            yaxis_title="Delta",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_price_surface_interactive(self, u_range=(1.0, 1.4), d_range=(0.6, 1.0), resolution=40):

        u_vals = np.linspace(u_range[0], u_range[1], resolution)
        d_vals = np.linspace(d_range[0], d_range[1], resolution)
        U, D = np.meshgrid(u_vals, d_vals)
        Z = np.zeros_like(U)

        for i in range(resolution):
            for j in range(resolution):
                if D[i, j] < np.exp(self.r) < U[i, j]:
                    pricer = OnePeriodBinomialPricer(
                        self.S0, self.K, U[i, j], D[i, j], self.r
                    )
                    Z[i, j] = pricer.price_call().price
                else:
                    Z[i, j] = float('nan')

        fig = go.Figure(
            data=[
                go.Surface(
                    z=Z,
                    x=U,
                    y=D,
                    colorscale="Viridis"
                )
            ]
        )

        fig.update_layout(
            title="Call Price Surface",
            scene=dict(
                xaxis_title="u",
                yaxis_title="d",
                zaxis_title="Price",
            ),
            height=600,
        )

        return fig

    def plot_arbitrage_region(self,
                          u_range=(1.0, 1.4),
                          d_range=(0.6, 1.0),
                          resolution=300):

        u_vals = np.linspace(u_range[0], u_range[1], resolution)
        d_vals = np.linspace(d_range[0], d_range[1], resolution)
        U, D = np.meshgrid(u_vals, d_vals)

        er = np.exp(self.r)
        arbi_free = (D < er) & (er < U)

        # Create hover text
        hover_text = np.empty_like(U, dtype=object)
        for i in range(resolution):
            for j in range(resolution):
                hover_text[i, j] = f"u: {U[i,j]:.3f}<br>d: {D[i,j]:.3f}<br>" + \
                                ("Arbitrage Free ✅" if arbi_free[i,j] else "Arbitrage ❌")

        # Heatmap without legend entry
        fig = go.Figure(data=go.Heatmap(
            x=U[0, :],
            y=D[:, 0],
            z=arbi_free.astype(int),
            colorscale="tropic",
            colorbar=dict(title=" "),
            showscale=False,
            showlegend=False,
            hoverinfo="text",
            text=hover_text
        ))

        # Horizontal boundary line (d = e^r)
        fig.add_trace(go.Scatter(
            x=[u_vals[0], u_vals[-1]],
            y=[er, er],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        ))
        fig.add_annotation(
            x=u_vals[-1],
            y=er,
            text="d = e^r boundary",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=12, color="black")
        )

        # Vertical boundary line (u = e^r)
        fig.add_trace(go.Scatter(
            x=[er, er],
            y=[d_range[0], d_range[1]],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        ))
        fig.add_annotation(
            x=er,
            y=d_range[-1],
            text="u = e^r boundary",
            showarrow=False,
            textangle=-90,
            xanchor="left",
            yanchor="top",
            font=dict(size=12, color="black")
        )

        # Layout adjustments
        fig.update_layout(
            title="Arbitrage Region in (u, d) Space",
            xaxis_title="u (Up Factor)",
            yaxis_title="d (Down Factor)",
            template="plotly_white",
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig


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
