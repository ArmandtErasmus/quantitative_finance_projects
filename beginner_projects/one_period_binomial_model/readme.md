# 📘 One-Period Binomial Option Pricing Model

**Interactive Dashboard**
[Interactive Dashboard](https://oneperiodbinomialpricing.streamlit.app/)

A Python implementation with pricing, Greeks, arbitrage checks, and visualisations.

This repository contains a complete and well-structured implementation of the One-Period Binomial Model for pricing European call and put options. It includes:
- Arbitrage validation
- Call and put pricing
- Risk-neutral probability calculation
- Delta and bond hedge ratios
- Replication error diagnostics
- 2D and 3D visualisation tools (Matplotlib + Plotly)
- JSON export of pricing results

The project is designed for anyone learning derivatives pricing, quantitative finance, actuarial modelling, or computational finance.

If you find this repository useful, please ⭐ star it on GitHub — it really helps!

# 🚀 Features
✔️ Core Pricer (OnePeriodBinomialPricer)
- Prices European calls and puts
- Computes:
- Option price
- Delta hedge ratio
- Bond (cash) position
- Risk-neutral probability
- Arbitrage indicator
- Replication error
- Export results as JSON

# ✔️ Visualisation Tools (BinomialVisualizer)
- 3D Call Price Surface
- Delta vs Strike Plot

# 📦 Installation
```bash
pip install numpy matplotlib plotly
```

# 📄 Example Usage

1. Price a call option:
```python
pricer = OnePeriodBinomialPricer(S0=100, K=100, u=1.12, d=0.92, r=0.02)
result = pricer.price_call()
print(pricer.to_json(result))
```

2. Visualisations:
```python
vis = BinomialVisualizer(S0=100, K=100, r=0.02)

vis.plot_price_surface_interactive()
vis.plot_delta_vs_strike(u=1.12, d=0.92)
vis.plot_arbitrage_region()
```
   
