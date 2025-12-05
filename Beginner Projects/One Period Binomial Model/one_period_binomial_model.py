# importing necessary libraries
from typing import Tuple, Dict
import numpy as np
from dataclasses import dataclass
import json
from datetime import datetime 
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable
import plotly.graph_objects as go

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

    def plot_price_surface(self,
                           u_range=(1.0, 1.4),
                           d_range=(0.6, 1.0),
                           resolution=40):
        u_vals = np.linspace(u_range[0], u_range[1], resolution)
        d_vals = np.linspace(d_range[0], d_range[1], resolution)

        U, D = np.meshgrid(u_vals, d_vals)
        prices = np.zeros_like(U)

        for i in range(resolution):
            for j in range(resolution):
                if D[i, j] < U[i, j]:      # only valid region
                    pricer = OnePeriodBinomialPricer(
                        S0=self.S0, K=self.K,
                        u=U[i, j], d=D[i, j], r=self.r, validate=False
                    )
                    prices[i, j] = pricer.price_call().price
                else:
                    prices[i, j] = np.nan

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(U, D, prices,
                               cmap=cm.viridis,
                               linewidth=0,
                               antialiased=True)

        ax.set_title("Call Price Surface as Function of u and d")
        ax.set_xlabel("u (Up Factor)")
        ax.set_ylabel("d (Down Factor)")
        ax.set_zlabel("Call Price")
        fig.colorbar(surf)
        plt.show()

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
                S0=self.S0, K=K,
                u=u, d=d, r=self.r, validate=False
            )
            deltas[i] = pricer.price_call().delta

        plt.figure(figsize=(8, 5))
        plt.plot(strikes, deltas)
        plt.axvline(self.S0, color="grey", linestyle="--", label="ATM (S0)")
        plt.title("Delta Sensitivity to Strike K")
        plt.xlabel("Strike K")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_price_surface_interactive(self,
                                   u_range=(1.0, 1.4),
                                   d_range=(0.6, 1.0),
                                   resolution=40):

        u_vals = np.linspace(u_range[0], u_range[1], resolution)
        d_vals = np.linspace(d_range[0], d_range[1], resolution)

        U, D = np.meshgrid(u_vals, d_vals)
        Z = np.zeros_like(U)

        for i in range(resolution):
            for j in range(resolution):
                if D[i, j] < np.exp(self.r) < U[i, j]:
                    try:
                        pricer = OnePeriodBinomialPricer(self.S0, self.K, U[i,j], D[i,j], self.r)
                        Z[i,j] = pricer.price_call().price
                    except:
                        Z[i,j] = np.nan
                else:
                    Z[i,j] = np.nan

        fig = go.Figure(data=[go.Surface(z=Z, x=U, y=D, colorscale="Viridis")])
        fig.update_layout(
            title="Interactive Call Price Surface",
            scene=dict(
                xaxis_title="u (Up Factor)",
                yaxis_title="d (Down Factor)",
                zaxis_title="Call Price"
            ),
            height=600,
        )
        fig.show()

    def plot_arbitrage_region(self,
                              u_range=(1.0, 1.4),
                              d_range=(0.6, 1.0),
                              resolution=300):
        u_vals = np.linspace(u_range[0], u_range[1], resolution)
        d_vals = np.linspace(d_range[0], d_range[1], resolution)

        U, D = np.meshgrid(u_vals, d_vals)

        # Arbitrage-free iff:   d < e^r < u
        arbi_free = (D < np.exp(self.r)) & (np.exp(self.r) < U)

        plt.figure(figsize=(8, 6))
        plt.contourf(U, D, arbi_free, cmap="RdYlGn", alpha=0.8)
        plt.colorbar(label="Arbitrage Free (1 = Yes, 0 = No)")

        # highlight boundary lines
        plt.axhline(np.exp(self.r), color="black", linestyle="--",
                    label="d = e^r boundary")
        plt.axvline(np.exp(self.r), color="black", linestyle="--")

        plt.title("Arbitrage Region in (u, d) Space")
        plt.xlabel("u (Up Factor)")
        plt.ylabel("d (Down Factor)")
        plt.legend()
        plt.grid(True)
        plt.show()

# example usage:
if __name__ == "__main__":
    pricer = OnePeriodBinomialPricer(S0=100, K=100, u=1.12, d=0.92, r=0.02)
    call_result = pricer.price_call()
    print(pricer.to_json(call_result))
    vis = BinomialVisualizer(S0=100, K=100, r=0.02)

    vis.plot_price_surface_interactive()

    vis.plot_delta_vs_strike(u=1.12, d=0.92)
  
    vis.plot_arbitrage_region()
