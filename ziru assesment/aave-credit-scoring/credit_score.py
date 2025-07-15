import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import uuid

def load_transactions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Flatten pages into a single list of transactions
    transactions = []
    for page in data['pages']:
        transactions.extend(page['content'])
    return transactions

def parse_timestamp(timestamp):
    try:
        if isinstance(timestamp, dict) and '$date' in timestamp:
            return pd.to_datetime(timestamp['$date'])
        return pd.to_datetime(timestamp, errors='coerce')
    except:
        return pd.NaT

def engineer_features(transactions):
    df = pd.DataFrame(transactions)
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    
    # Group by wallet
    features = df.groupby('userWallet').agg({
        'timestamp': ['count', 'min', 'max'],
        'amount': ['sum'],
        'action': [
            lambda x: sum(x == 'deposit'),
            lambda x: sum(x == 'borrow'),
            lambda x: sum(x == 'repay'),
            lambda x: sum(x == 'redeemunderlying'),
            lambda x: sum(x == 'liquidationcall')
        ],
        'assetSymbol': 'nunique',
        'network': 'nunique',
        'protocol': 'nunique'
    }).reset_index()
    
    features.columns = [
        'userWallet', 'total_transactions', 'first_transaction', 'last_transaction',
        'total_amount', 'deposit_count', 'borrow_count', 'repay_count',
        'redeem_count', 'liquidation_count', 'unique_assets', 'unique_networks',
        'unique_protocols'
    ]
    
    # Calculate additional features
    features['account_age_days'] = (features['last_transaction'] - features['first_transaction']).dt.total_seconds() / (24 * 3600)
    features['recency_days'] = (pd.Timestamp.now() - features['last_transaction']).dt.total_seconds() / (24 * 3600)
    features['repay_to_borrow'] = features['repay_count'] / features['borrow_count'].replace(0, 1)
    features['repay_to_borrow'] = features['repay_to_borrow'].clip(upper=1)
    features['borrow_to_deposit'] = features['borrow_count'] / features['deposit_count'].replace(0, 1)
    
    # Detect HFT: transactions within 1 hour
    df['time_diff'] = df.sort_values(['userWallet', 'timestamp']).groupby('userWallet')['timestamp'].diff().dt.total_seconds() / 3600
    hft = df[df['time_diff'] < 1].groupby('userWallet').size().reset_index(name='hft_count')
    features = features.merge(hft, on='userWallet', how='left').fillna({'hft_count': 0})
    
    return features

def compute_initial_scores(features):
    scores = np.full(len(features), 500.0)
    scores += np.where(features['repay_to_borrow'] > 0.9, 100, 0)
    scores += np.where(features['account_age_days'] > 30, 50, 0)
    scores += np.where(features['unique_assets'] > 2, 50, 0)
    scores -= features['liquidation_count'] * 200
    scores -= np.where(features['borrow_to_deposit'] > 1.5, 100, 0)
    scores -= np.where(features['hft_count'] > 10, 50, 0)
    return np.clip(scores, 0, 1000)

def train_and_score(features, initial_scores):
    X = features[[
        'total_transactions', 'deposit_count', 'borrow_count', 'repay_count',
        'redeem_count', 'liquidation_count', 'unique_assets', 'unique_networks',
        'unique_protocols', 'account_age_days', 'recency_days',
        'repay_to_borrow', 'borrow_to_deposit', 'hft_count'
    ]].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, initial_scores)
    raw_scores = model.predict(X)
    
    scaler = MinMaxScaler(feature_range=(0, 1000))
    final_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
    
    return pd.DataFrame({
        'userWallet': features['userWallet'],
        'credit_score': np.round(final_scores, 2)
    })

def main(file_path, output_path):
    transactions = load_transactions(file_path)
    features = engineer_features(transactions)
    initial_scores = compute_initial_scores(features)
    scores_df = train_and_score(features, initial_scores)
    scores_df.to_json(output_path, orient='records', lines=True)

if __name__ == "__main__":
    input_file = "wallet-transactions.json"
    output_file = "wallet-scores.json"
    main(input_file, output_file)