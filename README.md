# Customer Churn Prediction for SaaS Business

A machine learning project that predicts customer churn for a SaaS company and identifies at-risk customers for proactive retention campaigns.

## Project Overview

This project demonstrates end-to-end data science workflow including:
- **Exploratory Data Analysis** of customer behavior patterns
- **Feature Engineering** to create predictive variables
- **Machine Learning Models** (Logistic Regression & Random Forest)
- **Risk Scoring** for active customers
- **Actionable Insights** for retention strategies

## Business Problem

Customer churn is a critical metric for SaaS businesses. This project aims to:
1. Identify key factors that drive customer churn
2. Build predictive models to forecast churn probability
3. Score active customers by churn risk
4. Provide data-driven retention recommendations

## Dataset

The dataset contains **5,000 customers** with the following features:

### Customer Information
- Customer ID, Plan (Basic/Professional/Enterprise)
- Contract type (Month-to-Month, One Year, Two Year)
- Tenure in months, Monthly charges, Total charges

### Usage Metrics
- Number of users, Average login frequency
- Feature usage score, Number of integrations
- Days since last login

### Engagement Metrics
- Email open rate, Has add-ons
- Support tickets, Service calls

### Business Metrics
- Payment method, Payment failures
- Auto-renewal status, Company size, Industry

### Target Variable
- **Churned** (0 = Active, 1 = Churned)

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest
- **Data Visualization**: Matplotlib, Seaborn
- **SQL**: Customer segmentation and cohort analysis
- **Model Persistence**: Pickle

## Project Structure

```
02-Customer-Churn-Prediction/
├── data/
│   ├── customer_churn_data.csv       # Generated customer data
│   └── high_risk_customers.csv       # Scored high-risk customers
├── sql/
│   └── churn_analysis_queries.sql    # SQL queries for churn analysis
├── notebooks/
│   └── churn_prediction.py           # ML analysis script
├── visualizations/                    # Generated charts
├── models/                            # Saved ML models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── generate_data.py                   # Data generation script
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Customer Data

```bash
python generate_data.py
```

This creates a realistic customer dataset with ~35% churn rate.

### 3. Run Churn Prediction Analysis

```bash
cd notebooks
python churn_prediction.py
```

This script:
- Performs exploratory data analysis
- Engineers features for modeling
- Trains Logistic Regression and Random Forest models
- Evaluates model performance
- Scores active customers by churn risk
- Generates visualizations
- Saves trained models

### 4. SQL Analysis (Optional)

Import data into a database and run queries:

```bash
sqlite3 churn.db
.mode csv
.import data/customer_churn_data.csv customers
```

Run queries from `sql/churn_analysis_queries.sql`.

## Model Performance

### Logistic Regression
- **Accuracy**: ~79%
- **ROC-AUC**: ~0.88
- **Interpretability**: High (coefficient-based feature importance)

### Random Forest
- **Accuracy**: ~84%
- **ROC-AUC**: ~0.91
- **Strength**: Captures non-linear relationships

## Key Findings

### Churn Drivers

1. **Contract Type**
   - Month-to-month: 45% churn rate
   - Annual contracts: 15% churn rate
   - Two-year contracts: 5% churn rate

2. **Tenure**
   - First 3 months: Highest risk period
   - Customers who survive 12+ months: Much lower churn

3. **Engagement**
   - Low login frequency (<5 days/month): 60%+ churn
   - High feature usage: Strong retention indicator

4. **Payment Issues**
   - Payment failures increase churn by 30%+

5. **Product Integration**
   - 0 integrations: 40% churn
   - 3+ integrations: 15% churn

### Top Features (by Importance)

1. Contract type (Month-to-month vs committed)
2. Tenure in months
3. Days since last login
4. Feature usage score
5. Payment failures
6. Number of integrations
7. Auto-renewal status
8. Support ticket volume
9. Email engagement
10. Monthly charges

## Business Recommendations

### Immediate Actions

1. **High-Risk Customer Outreach**
   - Identify customers with churn probability >60%
   - Personal outreach from customer success team
   - Offer incentives (discounts, free add-ons)

2. **Payment Failure Protocol**
   - Immediate notification system
   - Multiple payment method options
   - Grace period with proactive communication

3. **Contract Incentives**
   - Discount for annual commitment (15-20%)
   - Reduce month-to-month pricing attractiveness
   - Migration campaigns for MTM customers

### Medium-Term Strategies

4. **Onboarding Optimization**
   - 90-day success program for new customers
   - Usage milestones and engagement tracking
   - Personalized feature adoption plans

5. **Feature Adoption**
   - In-app guidance for low-usage customers
   - Integration marketplace and tutorials
   - Success stories highlighting power users

6. **Customer Segmentation**
   - Premium tier receives dedicated success manager
   - Different retention strategies by segment
   - Personalized communication based on usage patterns

## Visualizations

The project generates:
1. **Churn by Category** - Contract type, plan, payment method
2. **Numeric Features Comparison** - Churned vs active customers
3. **Feature Importance** - Top predictive features
4. **ROC Curves** - Model performance comparison
5. **Confusion Matrices** - Prediction accuracy breakdown

## SQL Queries Included

- Overall churn metrics and revenue impact
- Churn by contract type, plan, tenure
- Usage and engagement analysis
- High-risk customer identification
- Cohort analysis
- Geographic and industry breakdowns
- Payment-related churn analysis

## Files Generated

- `high_risk_customers.csv` - Prioritized list for retention campaigns
- Visualization PNGs in `visualizations/` folder
- Trained models in `models/` folder

## Skills Demonstrated

- **Machine Learning**: Classification, model selection, hyperparameter consideration
- **Feature Engineering**: Creating predictive variables from raw data
- **Data Analysis**: Statistical analysis, cohort analysis, segmentation
- **SQL**: Complex queries for business intelligence
- **Python**: Pandas, Scikit-learn, data visualization
- **Business Acumen**: Translating models into actionable strategies
- **Communication**: Clear documentation and insights presentation

## Future Enhancements

- Deep learning models (Neural Networks)
- Time-series analysis for churn prediction windows
- Customer lifetime value (CLV) integration
- A/B testing framework for retention campaigns
- Real-time risk scoring API
- Dashboard with Streamlit or Tableau

## License

This project is open source and available for portfolio purposes.
