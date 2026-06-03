# Monte Carlo Portfolio Risk Simulator

A Monte Carlo simulator that projects portfolio value under Geometric Brownian Motion
(GBM) and reports the standard risk metrics — Value-at-Risk, Conditional VaR (expected
shortfall), an annualised Sharpe ratio, probability of loss, and a percentile-based
bull/base/bear scenario view — with a 4-panel dashboard.

## Method

Each simulated path evolves by the GBM log-return process on a daily step (dt = 1/252):

```
S(t) = S(0) * exp( (mu - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z ),   Z ~ N(0, 1)
```

Thousands of paths are simulated, and the distribution of terminal values is used to
compute the risk metrics.

## Engine validation

The simulation is checked against closed-form GBM results, and the comparison is printed
in every run. For a $100k portfolio at 10% drift / 15% volatility over 5 years:

- Simulated mean ≈ `S0 * exp(mu * T)` (within ~0.1%)
- Simulated median ≈ `S0 * exp((mu − 0.5*sigma²) * T)` (within ~0.4%)

This confirms the engine reproduces the theoretical moments rather than just producing
plausible-looking numbers.

## Metrics produced

- **VaR (5% and 1%)** — the terminal value at the 5th and 1st percentiles.
- **CVaR / expected shortfall (5%)** — the average outcome in the worst 5% of cases.
- **Sharpe ratio** — annualised: (annualised return − risk-free rate) ÷ annual volatility.
- **Probability of loss** — share of paths finishing below the initial investment.
- **Skewness & kurtosis** — shape of the terminal distribution (right-skewed, as a lognormal should be).
- **Scenario view** — bull/base/bear at the 90th/50th/10th percentiles.
- **Probability of target** — chance of finishing at or above a chosen value.

## How to run

```bash
pip install numpy pandas matplotlib scipy
python montecarlo_simulation.py
python montecarlo_simulation.py --initial 100000 --return 0.10 --vol 0.15 --years 5
python montecarlo_simulation.py --initial 250000 --return 0.07 --vol 0.20 --years 10 --sims 25000
```

Flags: `--initial`, `--return` (annual drift), `--vol` (annual volatility), `--years`,
`--rf` (risk-free rate), `--sims` (number of paths). The script prints a full report and
saves a 4-panel dashboard as `portfolio_analysis.png`.

## Dashboard panels

1. Simulated price paths (sample of 200) with the median path
2. Terminal-value distribution with VaR markers
3. Cumulative distribution function
4. Bull / base / bear scenario bars

## Limitations

GBM is a standard but simplified model of how prices move. Worth knowing before reading
the output:

- **Constant volatility and drift.** Real markets have volatility that clusters and
  changes over time; GBM assumes it's fixed.
- **Normally distributed log-returns.** This understates the chance of extreme moves —
  real return distributions have fatter tails, so true tail risk (VaR/CVaR) is likely
  worse than GBM implies.
- **No correlations or regime changes.** A single asset/portfolio is modelled in
  isolation, with no jumps, crashes, or shifting market regimes.
- **Inputs are assumptions, not forecasts.** The drift and volatility are supplied by the
  user; the output is only as meaningful as those inputs.

These are the well-known caveats of GBM-based risk modelling — the simulator is a
transparent illustration of the method, not a production risk system.

## Tech stack

Python, NumPy, Pandas, Matplotlib, SciPy.

## Author

Talal Bin Waqas
