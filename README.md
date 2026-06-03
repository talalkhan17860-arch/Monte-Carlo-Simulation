# Monte Carlo Portfolio Risk Simulator

Quantitative risk simulation tool built for portfolio analysis using Geometric Brownian Motion — the same stochastic framework used in institutional risk management.

## Risk Metrics
- Value at Risk (VaR) at 1% and 5%
- CVaR / Expected Shortfall
- Sharpe Ratio
- Probability of Loss
- Bull / Base / Bear scenario analysis
- Skewness & Kurtosis

## Methodology
Simulates 10,000 portfolio paths using Geometric Brownian Motion:

dS = μS dt + σS dW

Where μ = expected return, σ = volatility, dW = Wiener process

## How to Run
```bash
pip install numpy pandas matplotlib scipy seaborn
python montecarlo_simulation.py
```

## Output
4-panel dashboard saved as `portfolio_analysis.png`:
- Simulated price paths with median highlighted
- Final value distribution with VaR markers
- Scenario analysis (Bull/Base/Bear)
- Probability of hitting target returns

## Tech Stack
Python, NumPy, Matplotlib, SciPy, Seaborn        <img width="4066" height="3060" alt="portfolio_analysis" src="https://github.com/user-attachments/assets/ac052dcf-699e-49ef-8eee-bb84eee6b9bd" />
