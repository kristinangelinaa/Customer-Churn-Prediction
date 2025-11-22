"""
SaaS Customer Churn Data Generator
Generates realistic customer data for churn prediction analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate customer data
num_customers = 5000

# Customer IDs
customer_ids = [f"CUST_{i:05d}" for i in range(1, num_customers + 1)]

# Subscription plans
plans = ['Basic', 'Professional', 'Enterprise']
plan_prices = {'Basic': 29, 'Professional': 99, 'Enterprise': 299}

# Generate customer data
customers = []

for customer_id in customer_ids:
    # Plan selection (40% Basic, 40% Pro, 20% Enterprise)
    plan = np.random.choice(plans, p=[0.4, 0.4, 0.2])
    monthly_charges = plan_prices[plan]

    # Account age in months (0-36 months)
    tenure_months = np.random.randint(0, 37)

    # Contract type
    contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                    p=[0.5, 0.3, 0.2])

    # Payment method
    payment_method = np.random.choice(['Credit Card', 'Bank Transfer', 'PayPal'],
                                     p=[0.6, 0.25, 0.15])

    # Usage metrics
    # Active users in the account
    num_users = np.random.choice([1, 2, 3, 5, 10, 20, 50], p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.04, 0.01])

    # Login frequency (days per month)
    avg_login_freq = max(1, np.random.normal(15, 8))
    avg_login_freq = min(30, avg_login_freq)  # Cap at 30

    # Feature usage (0-100 scale)
    feature_usage_score = np.random.beta(2, 2) * 100  # Bell curve-ish around 50

    # Support tickets
    support_tickets = np.random.poisson(2 if tenure_months > 3 else 1)

    # Customer service interactions
    num_service_calls = support_tickets + np.random.poisson(0.5)

    # Email engagement (open rate %)
    email_open_rate = np.random.beta(3, 5) * 100  # Skewed lower

    # Has purchased add-ons
    has_addons = np.random.choice([0, 1], p=[0.7, 0.3])

    # Number of integrations set up
    num_integrations = np.random.choice([0, 1, 2, 3, 5, 10], p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])

    # Company size (number of employees)
    company_size = np.random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+'],
                                   p=[0.4, 0.3, 0.2, 0.08, 0.02])

    # Industry
    industry = np.random.choice(['Technology', 'Healthcare', 'Finance', 'Retail', 'Education', 'Other'],
                               p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

    # Auto-renewal enabled
    auto_renewal = np.random.choice([0, 1], p=[0.3, 0.7])

    # Payment failures in last 3 months
    payment_failures = np.random.choice([0, 1, 2, 3], p=[0.85, 0.1, 0.04, 0.01])

    # Days since last login
    days_since_last_login = int(np.random.exponential(7))
    days_since_last_login = min(90, days_since_last_login)

    # Total charges (tenure * monthly charges with some variation)
    total_charges = monthly_charges * tenure_months * np.random.uniform(0.95, 1.05)

    # === CHURN LOGIC ===
    # Calculate churn probability based on various factors
    churn_prob = 0.2  # Base churn rate

    # Negative factors (increase churn)
    if contract_type == 'Month-to-Month':
        churn_prob += 0.25
    if tenure_months < 6:
        churn_prob += 0.2
    if avg_login_freq < 5:
        churn_prob += 0.2
    if feature_usage_score < 30:
        churn_prob += 0.15
    if support_tickets > 5:
        churn_prob += 0.15
    if payment_failures > 0:
        churn_prob += 0.3
    if not auto_renewal:
        churn_prob += 0.15
    if email_open_rate < 20:
        churn_prob += 0.1
    if days_since_last_login > 30:
        churn_prob += 0.2
    if num_integrations == 0:
        churn_prob += 0.1

    # Positive factors (decrease churn)
    if plan == 'Enterprise':
        churn_prob -= 0.2
    if tenure_months > 24:
        churn_prob -= 0.25
    if has_addons:
        churn_prob -= 0.15
    if num_integrations >= 3:
        churn_prob -= 0.15
    if contract_type == 'Two Year':
        churn_prob -= 0.3

    # Ensure probability is between 0 and 1
    churn_prob = max(0, min(1, churn_prob))

    # Determine churn
    churned = 1 if np.random.random() < churn_prob else 0

    customers.append({
        'customer_id': customer_id,
        'tenure_months': tenure_months,
        'plan': plan,
        'monthly_charges': monthly_charges,
        'total_charges': round(total_charges, 2),
        'contract_type': contract_type,
        'payment_method': payment_method,
        'num_users': num_users,
        'avg_login_frequency': round(avg_login_freq, 1),
        'feature_usage_score': round(feature_usage_score, 1),
        'support_tickets': support_tickets,
        'num_service_calls': num_service_calls,
        'email_open_rate': round(email_open_rate, 1),
        'has_addons': has_addons,
        'num_integrations': num_integrations,
        'company_size': company_size,
        'industry': industry,
        'auto_renewal': auto_renewal,
        'payment_failures': payment_failures,
        'days_since_last_login': days_since_last_login,
        'churned': churned
    })

# Create DataFrame
df = pd.DataFrame(customers)

# Save to CSV
df.to_csv('data/customer_churn_data.csv', index=False)

print("=" * 60)
print("CUSTOMER CHURN DATASET GENERATED")
print("=" * 60)
print(f"\nTotal Customers: {len(df):,}")
print(f"Churned Customers: {df['churned'].sum():,}")
print(f"Churn Rate: {df['churned'].mean()*100:.2f}%")
print(f"Active Customers: {(1-df['churned']).sum():,}")

print("\nChurn by Contract Type:")
print(df.groupby('contract_type')['churned'].agg(['sum', 'mean']))

print("\nChurn by Plan:")
print(df.groupby('plan')['churned'].agg(['sum', 'mean']))

print("\n" + "=" * 60)
print("Dataset saved to 'data/customer_churn_data.csv'")
print("=" * 60)
