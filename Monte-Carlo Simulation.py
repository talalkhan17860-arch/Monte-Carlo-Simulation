"""
============================================================
 Monte Carlo Portfolio Risk Simulator
 Author : Talal Bin Waqas

 Simulates portfolio value paths under Geometric Brownian
 Motion and reports the standard risk metrics: Value-at-Risk,
 Conditional VaR (expected shortfall), an annualised Sharpe
 ratio, probability of loss, and a percentile-based scenario
 view. Includes a 4-panel dashboard.

 The engine is validated against closed-form GBM results:
 the simulated mean and median terminal values match
 S0*exp(mu*T) and S0*exp((mu - 0.5*sigma^2)*T) respectively.
============================================================

Usage:
    python montecarlo_simulation.py
    python montecarlo_simulation.py --initial 100000 --return 0.10 --vol 0.15 --years 5
    python montecarlo_simulation.py --initial 250000 --return 0.07 --vol 0.20 --years 10 --sims 50000
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")


class PortfolioMonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio risk analysis using Geometric
    Brownian Motion (GBM).

    Each path evolves by the GBM log-return process:
        S(t) = S(0) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)

    on a daily time-step (dt = 1/252). From the distribution of terminal
    values it computes VaR, CVaR/expected shortfall, an annualised Sharpe
    ratio, probability of loss, and bull/base/bear percentile scenarios.

    Parameters
    ----------
    initial_investment : float — starting portfolio value ($)
    annual_return      : float — expected annual drift (e.g. 0.10 = 10%)
    volatility         : float — annual volatility (e.g. 0.15 = 15%)
    years              : int   — horizon in years
    risk_free_rate     : float — risk-free rate for the Sharpe ratio
    simulations        : int   — number of paths (default 10,000)
    seed               : int   — RNG seed for reproducibility
    """

    def __init__(
        self,
        initial_investment: float,
        annual_return: float,
        volatility: float,
        years: int,
        risk_free_rate: float = 0.05,
        simulations: int = 10_000,
        seed: int = 42,
    ):
        self.initial = initial_investment
        self.mu      = annual_return
        self.sigma   = volatility
        self.years   = years
        self.rf      = risk_free_rate
        self.sims    = simulations
        self.seed    = seed
        self.dt      = 1 / 252
        self.days    = years * 252
        self.results = None
        self.paths   = None

    # ------------------------------------------------------------------
    def run(self) -> np.ndarray:
        """Run the vectorised GBM simulation; returns the terminal values."""
        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal(size=(self.days, self.sims))
        daily_log_returns = (
            (self.mu - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )
        self.paths = self.initial * np.exp(np.cumsum(daily_log_returns, axis=0))
        self.results = self.paths[-1]
        return self.results

    # ------------------------------------------------------------------
    def calculate_metrics(self) -> dict:
        """
        Risk metrics from the terminal-value distribution.

        Sharpe is annualised correctly: excess of the annualised return
        over the risk-free rate, divided by the annual volatility (the
        model's input sigma) — not by terminal-value dispersion.
        """
        if self.results is None:
            self.run()
        r = self.results

        # annualised return implied by the mean terminal value
        annualised_return = (np.mean(r) / self.initial) ** (1 / self.years) - 1
        # Sharpe uses the annual volatility directly (correct annualisation)
        sharpe = (annualised_return - self.rf) / self.sigma if self.sigma else np.nan

        var_5  = np.percentile(r, 5)
        var_1  = np.percentile(r, 1)
        cvar_5 = np.mean(r[r <= var_5])

        return {
            "Initial Investment":               self.initial,
            "Mean Final Value":                 np.mean(r),
            "Median Final Value":               np.median(r),
            "Std Deviation":                    np.std(r),
            "Min":                              np.min(r),
            "Max":                              np.max(r),
            "Annualised Return":                annualised_return,
            "VaR (5%)":                         var_5,
            "VaR (1%)":                         var_1,
            "CVaR / Expected Shortfall (5%)":   cvar_5,
            "Sharpe Ratio":                     sharpe,
            "Prob. of Loss":                    np.mean(r < self.initial),
            "Skewness":                         stats.skew(r),
            "Kurtosis":                         stats.kurtosis(r),
        }

    # ------------------------------------------------------------------
    def scenario_analysis(self) -> dict:
        """Bull / base / bear percentile breakdown of terminal values."""
        if self.results is None:
            self.run()
        return {
            "Bull Case  (90th percentile)": np.percentile(self.results, 90),
            "Base Case  (50th percentile)": np.percentile(self.results, 50),
            "Bear Case  (10th percentile)": np.percentile(self.results, 10),
        }

    # ------------------------------------------------------------------
    def probability_of_target(self, target_value: float) -> float:
        """Probability the portfolio finishes at or above target_value."""
        if self.results is None:
            self.run()
        return float(np.mean(self.results >= target_value))

    # ------------------------------------------------------------------
    def validate(self) -> dict:
        """
        Sanity check: compare simulated moments to closed-form GBM.
        Returns the theoretical vs simulated mean/median and the % error.
        """
        if self.results is None:
            self.run()
        theo_mean   = self.initial * np.exp(self.mu * self.years)
        theo_median = self.initial * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.years)
        sim_mean    = float(np.mean(self.results))
        sim_median  = float(np.median(self.results))
        return {
            "Theoretical Mean":   theo_mean,
            "Simulated Mean":     sim_mean,
            "Mean Error (%)":     (sim_mean / theo_mean - 1) * 100,
            "Theoretical Median": theo_median,
            "Simulated Median":   sim_median,
            "Median Error (%)":   (sim_median / theo_median - 1) * 100,
        }

    # ------------------------------------------------------------------
    def visualize(self, save: bool = True, fname: str = "portfolio_analysis.png") -> plt.Figure:
        """4-panel dashboard: paths, terminal distribution + VaR, CDF, scenarios."""
        if self.results is None:
            self.run()
        metrics   = self.calculate_metrics()
        scenarios = self.scenario_analysis()

        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(
            f"Monte Carlo Portfolio Risk Analysis  —  "
            f"${self.initial:,.0f} initial  |  {self.years}Y horizon  |  "
            f"{self.sims:,} simulations",
            fontsize=14, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

        # Panel 1: price paths (sample of 200)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.paths[:, :200], alpha=0.05, linewidth=0.5, color="steelblue")
        ax1.plot(np.median(self.paths, axis=1), color="orange", linewidth=2, label="Median path")
        ax1.axhline(self.initial, color="red", linestyle="--", linewidth=1, label="Initial")
        ax1.set_title("Simulated price paths (200 sample)", fontweight="bold")
        ax1.set_xlabel("Trading days"); ax1.set_ylabel("Portfolio value ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=8)

        # Panel 2: terminal distribution with VaR markers
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.results, bins=120, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.3)
        ax2.axvline(metrics["VaR (5%)"], color="orange", linestyle="--", linewidth=1.5,
                    label=f"VaR 5%: ${metrics['VaR (5%)']:,.0f}")
        ax2.axvline(metrics["VaR (1%)"], color="red", linestyle="--", linewidth=1.5,
                    label=f"VaR 1%: ${metrics['VaR (1%)']:,.0f}")
        ax2.axvline(metrics["Mean Final Value"], color="green", linestyle="-", linewidth=1.5,
                    label=f"Mean: ${metrics['Mean Final Value']:,.0f}")
        ax2.set_title("Final portfolio distribution", fontweight="bold")
        ax2.set_xlabel("Portfolio value ($)"); ax2.set_ylabel("Frequency")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax2.legend(fontsize=8)

        # Panel 3: CDF
        ax3 = fig.add_subplot(gs[1, 0])
        sr = np.sort(self.results); cdf = np.arange(1, len(sr) + 1) / len(sr)
        ax3.plot(sr, cdf, color="steelblue", linewidth=1.5)
        ax3.axvline(self.initial, color="red", linestyle="--", linewidth=1.2, label="Initial")
        ax3.axhline(0.05, color="orange", linestyle=":", linewidth=1, label="5% confidence")
        ax3.set_title("Cumulative distribution function", fontweight="bold")
        ax3.set_xlabel("Portfolio value ($)"); ax3.set_ylabel("Cumulative probability")
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax3.legend(fontsize=8)

        # Panel 4: scenarios
        ax4 = fig.add_subplot(gs[1, 1])
        labels = ["Bear\n(P10)", "Base\n(P50)", "Bull\n(P90)"]
        values = [scenarios["Bear Case  (10th percentile)"],
                  scenarios["Base Case  (50th percentile)"],
                  scenarios["Bull Case  (90th percentile)"]]
        colors = ["#d9534f", "#5bc0de", "#5cb85c"]
        bars = ax4.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
        ax4.axhline(self.initial, color="black", linestyle="--", linewidth=1.2, label="Initial")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                     f"${val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax4.set_title("Scenario analysis — bull / base / bear", fontweight="bold")
        ax4.set_ylabel("Portfolio value ($)")
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax4.legend(fontsize=8)

        if save:
            try:
                plt.savefig(fname, dpi=200, bbox_inches="tight")
                print(f"\nDashboard saved as '{fname}'")
            except Exception as e:
                print(f"\nCould not save chart: {e}")
        plt.close(fig)
        return fig


# ======================================================================
def print_report(sim: PortfolioMonteCarloSimulator):
    bar = "=" * 62
    m = sim.calculate_metrics()
    v = sim.validate()

    print(f"\n{bar}\n  MONTE CARLO PORTFOLIO RISK ANALYSIS\n  Geometric Brownian Motion\n{bar}")
    print(f"  {'Initial investment':<40} ${sim.initial:>14,.0f}")
    print(f"  {'Expected annual return':<40} {sim.mu:>14.1%}")
    print(f"  {'Annual volatility':<40} {sim.sigma:>14.1%}")
    print(f"  {'Horizon':<40} {sim.years:>13} yrs")
    print(f"  {'Simulations':<40} {sim.sims:>14,}")

    print(f"\n{bar}\n  RISK METRICS\n{bar}")
    for k in ["Mean Final Value", "Median Final Value", "VaR (5%)", "VaR (1%)",
              "CVaR / Expected Shortfall (5%)", "Min", "Max"]:
        print(f"  {k:<40} ${m[k]:>14,.0f}")
    print(f"  {'Annualised return':<40} {m['Annualised Return']:>14.2%}")
    print(f"  {'Sharpe ratio':<40} {m['Sharpe Ratio']:>14.2f}")
    print(f"  {'Probability of loss':<40} {m['Prob. of Loss']:>14.2%}")
    print(f"  {'Skewness':<40} {m['Skewness']:>14.2f}")
    print(f"  {'Kurtosis':<40} {m['Kurtosis']:>14.2f}")

    print(f"\n{bar}\n  SCENARIO ANALYSIS\n{bar}")
    for label, val in sim.scenario_analysis().items():
        ret = (val / sim.initial - 1) * 100
        print(f"  {label:<40} ${val:>14,.0f}  ({ret:+.1f}%)")

    print(f"\n{bar}\n  PROBABILITY OF REACHING TARGET\n{bar}")
    for t in (sim.initial * 1.5, sim.initial * 2.0, sim.initial * 2.5):
        print(f"  P(final >= ${t:>12,.0f}) {sim.probability_of_target(t):>22.2%}")

    print(f"\n{bar}\n  ENGINE VALIDATION (simulated vs closed-form GBM)\n{bar}")
    print(f"  {'Theoretical mean  S0*exp(mu*T)':<40} ${v['Theoretical Mean']:>14,.0f}")
    print(f"  {'Simulated mean':<40} ${v['Simulated Mean']:>14,.0f}   ({v['Mean Error (%)']:+.2f}%)")
    print(f"  {'Theoretical median':<40} ${v['Theoretical Median']:>14,.0f}")
    print(f"  {'Simulated median':<40} ${v['Simulated Median']:>14,.0f}   ({v['Median Error (%)']:+.2f}%)")
    print(f"{bar}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monte Carlo Portfolio Risk Simulator — Talal Bin Waqas")
    ap.add_argument("--initial", type=float, default=100_000.0)
    ap.add_argument("--return", dest="ret", type=float, default=0.10)
    ap.add_argument("--vol", type=float, default=0.15)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--rf", type=float, default=0.05)
    ap.add_argument("--sims", type=int, default=10_000)
    a = ap.parse_args()

    sim = PortfolioMonteCarloSimulator(
        initial_investment=a.initial, annual_return=a.ret, volatility=a.vol,
        years=a.years, risk_free_rate=a.rf, simulations=a.sims,
    )
    sim.run()
    print_report(sim)
    sim.visualize(save=True)