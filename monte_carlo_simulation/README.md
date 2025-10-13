# Monte Carlo Simulation for Aggregate Insurance Losses

This project simulates **aggregate insurance losses** using **Monte Carlo methods**.  
It models the **frequency** of claims with a Poisson distribution and the **severity** (amount per claim) with either a Lognormal or Gamma distribution.  
The output includes key **risk metrics** such as Value-at-Risk (VaR), Tail Value-at-Risk (TVaR), and visualizations of the loss distribution.

---

## 📊 Overview

In insurance and actuarial science, the **aggregate loss** over a period is defined as:

\[
S = \sum_{i=1}^{N} X_i
\]

where:
- \( N \) is the number of claims (frequency),
- \( X_i \) is the severity (claim amount) for each claim.

This simulation approximates the distribution of \( S \) by generating thousands of random outcomes using **Monte Carlo simulation**.

---

## 🧠 Features

- Simulate aggregate losses with:
  - **Poisson** claim frequency
  - **Lognormal** or **Gamma** claim severity
- Compute and display:
  - Mean and Standard Deviation
  - **Value-at-Risk (VaR)** at 90%, 95%, 99%
  - **Tail Value-at-Risk (TVaR)** at 90%, 95%, 99%
- Visualize results with:
  - Histogram (linear scale)
  - Histogram (logarithmic scale)
  - Empirical Cumulative Distribution Function (ECDF)
- Save all results to `.csv` and `.png` files.

---

## ⚙️ Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/YOUR_USERNAME/monte-carlo-insurance.git
cd monte-carlo-insurance
pip install -r requirements.txt
