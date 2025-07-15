Aave V2 Wallet Credit Scoring Model

Overview

This project implements a machine learning model to assign credit scores (0–1000) to wallets interacting with the Aave V2 protocol, based on transaction-level data. Higher scores indicate reliable usage (e.g., timely repayments, low risk), while lower scores reflect risky or bot-like behavior (e.g., frequent liquidations, high-frequency trading).

Data

The input is a JSON file (wallet-transactions.json) containing transaction records with fields like _id, action (deposit, borrow, repay, redeemunderlying, liquidationcall), amount, timestamp, assetSymbol, userWallet, network, and protocol.

Feature Engineering





Transaction Frequency: Total transactions, counts of each action type.



Transaction Volume: Total deposit and borrow amounts, repay-to-borrow ratio, borrow-to-deposit ratio.



Risk Indicators: Liquidation events, high-frequency trading (transactions <1 hour apart).



Time-Based: Account age (days), recency of last transaction.



Asset Diversity: Number of unique assets and networks.



Protocol Interaction: Unique protocols used.

Model

A Random Forest Regressor predicts scores:





Initial Scoring: Heuristic-based scores (base=500):





+100 for repay-to-borrow ratio > 0.9.



+50 for account age > 30 days or unique assets > 2.



-200 per liquidation event, -100 for borrow-to-deposit > 1.5, -50 for >10 high-frequency transactions.



Training: Uses initial scores as pseudo-labels to train the model.



Scaling: Predicted scores are normalized to 0–1000.

Usage

Run the script:

python credit_score.py





Input: wallet-transactions.json



Output: wallet-scores.json with userWallet and credit_score.

Extensibility





New Features: Add fields like collateral ratio or interest paid by extending engineer_features.



Model Tuning: Adjust Random Forest parameters (e.g., n_estimators) or try gradient boosting.



Data Sources: Incorporate external data (e.g., wallet balances) via additional JSON fields.



Validation: Add cross-validation or clustering to refine unsupervised scoring.

Validation

The model penalizes risky behavior (liquidations, high leverage) and rewards reliability (repayments, longevity). Feature importance from the Random Forest can be inspected to ensure alignment with DeFi risk principles. Scores are capped at 0–1000 for consistency.