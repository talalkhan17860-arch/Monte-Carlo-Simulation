"""
============================================================
 Monte Carlo Portfolio Risk Simulator
 Author : Talal Waqas
 Method : Geometric Brownian Motion (GBM)
 Metrics : VaR, CVaR/ES, Sharpe Ratio, Scenario Analysis
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class PortfolioMonteCarloSimulator:
    """
    Enterprise-grade Monte Carlo simulator for portfolio risk analysis.

    Implements Geometric Brownian Motion (GBM) with advanced risk metrics
    including Value-at-Risk (VaR), Conditional VaR (Expected Shortfall),
    Sharpe Ratio, and full scenario analysis — the standard toolkit
    used in investment banking and private equity.

    Parameters
    ----------
    initial_investment : float  — Starting portfolio value ($)
    annual_return      : float  — Expected annual return (e.g. 0.10 = 10%)
    volatility         : float  — Annual volatility / std dev (e.g. 0.15 = 15%)
    years              : int    — Investment horizon in years
    risk_free_rate     : float  — Risk-free rate for Sharpe calculation (e.g. 0.05)
    simulations        : int    — Number of Monte Carlo paths (default 10,000)
    """

    def __init__(
        self,
        initial_investment: float,
        annual_return: float,
        volatility: float,
        years: int,
        risk_free_rate: float = 0.05,
        simulations: int = 10_000,
    ):
        self.initial   = initial_investment
        self.mu        = annual_return
        self.sigma     = volatility
        self.years     = years
        self.rf        = risk_free_rate
        self.sims      = simulations
        self.dt        = 1 / 252          # daily time-step
        self.days      = years * 252      # total trading days
        self.results   = None             # final portfolio values
        self.paths     = None             # full price paths (for charting)

    # ------------------------------------------------------------------
    # Core Simulation
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """
        Execute vectorised Monte Carlo simulation.

        Uses the GBM log-return formula:
            S(t) = S(0) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
        where Z ~ N(0,1).
        """
        np.random.seed(42)

        # Shape: (trading_days, simulations)
        Z = np.random.normal(0, 1, size=(self.days, self.sims))

        daily_log_returns = (
            (self.mu - 0.5 * self.sigma ** 2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )

        # Cumulative price paths
        self.paths  = self.initial * np.exp(np.cumsum(daily_log_returns, axis=0))
        self.results = self.paths[-1]          # final values only
        return self.results

    # ------------------------------------------------------------------
    # Risk Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self) -> dict:
        """
        Compute comprehensive risk metrics used in IB / PE analysis.

        Returns a dictionary with:
          - Descriptive stats (mean, median, std, min, max)
          - VaR at 1% and 5% confidence levels
          - CVaR / Expected Shortfall at 5%
          - Sharpe Ratio (annualised)
          - Probability of loss
          - Skewness & Kurtosis
        """
        if self.results is None:
            self.run()

        r = self.results

        # Annualised return and volatility for Sharpe
        port_return     = (np.mean(r) / self.initial) ** (1 / self.years) - 1
        port_vol        = (np.std(r)  / self.initial) / np.sqrt(self.years)
        sharpe          = (port_return - self.rf) / port_vol if port_vol != 0 else np.nan

        var_5  = np.percentile(r, 5)
        var_1  = np.percentile(r, 1)
        cvar_5 = np.mean(r[r <= var_5])

        return {
            "Initial Investment":       self.initial,
            "Mean Final Value":         np.mean(r),
            "Median Final Value":       np.median(r),
            "Std Deviation":            np.std(r),
            "Min":                      np.min(r),
            "Max":                      np.max(r),
            "VaR (5%)":                 var_5,
            "VaR (1%)":                 var_1,
            "CVaR / Expected Shortfall (5%)": cvar_5,
            "Sharpe Ratio":             sharpe,
            "Prob. of Loss":            np.mean(r < self.initial),
            "Skewness":                 stats.skew(r),
            "Kurtosis":                 stats.kurtosis(r),
        }

    # ------------------------------------------------------------------
    # Scenario Analysis
    # ------------------------------------------------------------------

    def scenario_analysis(self) -> dict:
        """
        Bull / Base / Bear case breakdown — standard PE/IB framing.
        """
        if self.results is None:
            self.run()

        scenarios = {
            "Bull Case  (90th percentile)": np.percentile(self.results, 90),
            "Base Case  (50th percentile)": np.percentile(self.results, 50),
            "Bear Case  (10th percentile)": np.percentile(self.results, 10),
        }
        return scenarios

    # ------------------------------------------------------------------
    # Probability Target
    # ------------------------------------------------------------------

    def probability_of_target(self, target_value: float) -> float:
        """Return probability that the portfolio reaches or exceeds target_value."""
        if self.results is None:
            self.run()
        return np.sum(self.results >= target_value) / self.sims

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, save: bool = True) -> plt.Figure:
        """
        Generate a professional 4-panel dashboard:
          1. Simulated price paths (sample of 200)
          2. Final portfolio value distribution with VaR markers
          3. Cumulative distribution function
          4. Scenario / percentile breakdown
        """
        if self.results is None:
            self.run()

        metrics   = self.calculate_metrics()
        scenarios = self.scenario_analysis()

        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(
            f"Monte Carlo Portfolio Risk Analysis  —  "
            f"${self.initial:,.0f} Initial  |  {self.years}Y Horizon  |  "
            f"{self.sims:,} Simulations",
            fontsize=14, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

        # ── Panel 1: Price Paths ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        sample_paths = self.paths[:, :200]          # plot 200 paths only
        ax1.plot(sample_paths, alpha=0.05, linewidth=0.5, color="steelblue")
        ax1.plot(np.median(self.paths, axis=1), color="orange",
                 linewidth=2, label="Median Path")
        ax1.axhline(self.initial, color="red", linestyle="--",
                    linewidth=1, label="Initial Investment")
        ax1.set_title("Simulated Price Paths (200 sample)", fontweight="bold")
        ax1.set_xlabel("Trading Days")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=8)

        # ── Panel 2: Distribution with VaR ─────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.results, bins=120, alpha=0.7,
                 color="steelblue", edgecolor="white", linewidth=0.3)
        ax2.axvline(metrics["VaR (5%)"], color="orange", linestyle="--",
                    linewidth=1.5, label=f"VaR 5%: ${metrics['VaR (5%)']:,.0f}")
        ax2.axvline(metrics["VaR (1%)"], color="red", linestyle="--",
                    linewidth=1.5, label=f"VaR 1%: ${metrics['VaR (1%)']:,.0f}")
        ax2.axvline(metrics["Mean Final Value"], color="green", linestyle="-",
                    linewidth=1.5, label=f"Mean: ${metrics['Mean Final Value']:,.0f}")
        ax2.set_title("Final Portfolio Distribution", fontweight="bold")
        ax2.set_xlabel("Portfolio Value ($)")
        ax2.set_ylabel("Frequency")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax2.legend(fontsize=8)

        # ── Panel 3: CDF ────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_results = np.sort(self.results)
        cdf = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
        ax3.plot(sorted_results, cdf, color="steelblue", linewidth=1.5)
        ax3.axvline(self.initial, color="red", linestyle="--",
                    linewidth=1.2, label="Initial Investment")
        ax3.axhline(0.05, color="orange", linestyle=":",
                    linewidth=1, label="5% Confidence")
        ax3.set_title("Cumulative Distribution Function (CDF)", fontweight="bold")
        ax3.set_xlabel("Portfolio Value ($)")
        ax3.set_ylabel("Cumulative Probability")
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
        ax3.legend(fontsize=8)

        # ── Panel 4: Scenario Bar Chart ─────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        labels  = ["Bear\n(P10)", "Base\n(P50)", "Bull\n(P90)"]
        values  = list(scenarios.values())
        colors  = ["#d9534f", "#5bc0de", "#5cb85c"]
        bars    = ax4.bar(labels, values, color=colors, edgecolor="white",
                          linewidth=0.8, width=0.5)
        ax4.axhline(self.initial, color="black", linestyle="--",
                    linewidth=1.2, label="Initial Investment")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (max(values) * 0.01),
                     f"${val:,.0f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        ax4.set_title("Scenario Analysis  —  Bull / Base / Bear", fontweight="bold")
        ax4.set_ylabel("Portfolio Value ($)")
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax4.legend(fontsize=8)

        if save:
            plt.savefig("portfolio_analysis.png", dpi=300, bbox_inches="tight")
            print("\n✅  Chart saved as 'portfolio_analysis.png'")

        plt.show()
        return fig


# ======================================================================
#  Main Execution
# ======================================================================

if __name__ == "__main__":

    simulator = PortfolioMonteCarloSimulator(
        initial_investment = 100_000,   # $100,000 starting portfolio
        annual_return      = 0.10,      # 10% expected annual return
        volatility         = 0.15,      # 15% annual volatility
        years              = 5,         # 5-year investment horizon
        risk_free_rate     = 0.05,      # 5% risk-free rate (US T-Bill approx)
        simulations        = 10_000,    # 10,000 paths — runs fast on any machine
    )

    # Run simulation
    simulator.run()

    # ── Risk Metrics ────────────────────────────────────────────────────
    metrics = simulator.calculate_metrics()

    print("\n" + "=" * 62)
    print("  MONTE CARLO PORTFOLIO RISK ANALYSIS")
    print("  Talal Bin Waqas  |  Geometric Brownian Motion")
    print("=" * 62)

    for metric, value in metrics.items():
        if metric in ("Sharpe Ratio", "Skewness", "Kurtosis"):
            print(f"  {metric:.<45}  {value:>10.4f}")
        elif metric == "Prob. of Loss":
            print(f"  {metric:.<45}  {value:>9.2%}")
        else:
            print(f"  {metric:.<45} ${value:>12,.2f}")

    # ── Scenario Analysis ───────────────────────────────────────────────
    scenarios = simulator.scenario_analysis()

    print("\n" + "=" * 62)
    print("  SCENARIO ANALYSIS")
    print("=" * 62)
    for label, value in scenarios.items():
        ret = (value / simulator.initial - 1) * 100
        print(f"  {label:.<45} ${value:>12,.2f}  ({ret:+.1f}%)")

    # ── Probability Targets ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  PROBABILITY ANALYSIS")
    print("=" * 62)
    targets = [150_000, 200_000, 250_000]
    for t in targets:
        p = simulator.probability_of_target(t)
        print(f"  Prob. portfolio >= ${t:>10,.0f}  ............  {p:.2%}")

    print("\n")

    # ── Visualisation ───────────────────────────────────────────────────
    simulator.visualize(save=True)