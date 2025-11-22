-- Bank Customer Churn Analysis SQL Queries

-- ============================================
-- 1. OVERALL CHURN METRICS
-- ============================================

-- Overall churn statistics
SELECT
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    COUNT(*) - SUM(Exited) AS active_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct,
    AVG(Balance) AS avg_balance,
    AVG(EstimatedSalary) AS avg_salary
FROM customers;


-- ============================================
-- 2. CHURN BY DEMOGRAPHICS
-- ============================================

-- Churn by geography
SELECT
    Geography,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct,
    AVG(Balance) AS avg_balance
FROM customers
GROUP BY Geography
ORDER BY churn_rate_pct DESC;


-- Churn by gender
SELECT
    Gender,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct,
    AVG(Age) AS avg_age
FROM customers
GROUP BY Gender
ORDER BY churn_rate_pct DESC;


-- ============================================
-- 3. CHURN BY AGE
-- ============================================

-- Churn by age group
SELECT
    CASE
        WHEN Age < 25 THEN '18-25'
        WHEN Age < 35 THEN '25-35'
        WHEN Age < 45 THEN '35-45'
        WHEN Age < 55 THEN '45-55'
        ELSE '55+'
    END AS age_group,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY age_group
ORDER BY age_group;


-- ============================================
-- 4. CHURN BY ACCOUNT CHARACTERISTICS
-- ============================================

-- Churn by number of products
SELECT
    NumOfProducts,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY NumOfProducts
ORDER BY NumOfProducts;


-- Churn by active membership status
SELECT
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS status,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY IsActiveMember;


-- Churn by credit card ownership
SELECT
    CASE WHEN HasCrCard = 1 THEN 'Has Credit Card' ELSE 'No Credit Card' END AS card_status,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY HasCrCard;


-- ============================================
-- 5. CHURN BY TENURE
-- ============================================

-- Churn by tenure buckets
SELECT
    CASE
        WHEN Tenure <= 2 THEN '0-2 years'
        WHEN Tenure <= 5 THEN '3-5 years'
        WHEN Tenure <= 8 THEN '6-8 years'
        ELSE '9+ years'
    END AS tenure_group,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;


-- ============================================
-- 6. CHURN BY BALANCE
-- ============================================

-- Churn by balance range
SELECT
    CASE
        WHEN Balance = 0 THEN 'Zero Balance'
        WHEN Balance < 50000 THEN 'Low (0-50K)'
        WHEN Balance < 100000 THEN 'Medium (50K-100K)'
        WHEN Balance < 150000 THEN 'High (100K-150K)'
        ELSE 'Very High (150K+)'
    END AS balance_range,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY balance_range
ORDER BY churn_rate_pct DESC;


-- ============================================
-- 7. CHURN BY CREDIT SCORE
-- ============================================

-- Churn by credit score range
SELECT
    CASE
        WHEN CreditScore < 600 THEN 'Poor (<600)'
        WHEN CreditScore < 700 THEN 'Fair (600-700)'
        WHEN CreditScore < 800 THEN 'Good (700-800)'
        ELSE 'Excellent (800+)'
    END AS credit_range,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY credit_range
ORDER BY churn_rate_pct DESC;


-- ============================================
-- 8. HIGH-RISK CUSTOMERS
-- ============================================

-- Identify customers at high risk of churning
SELECT
    CustomerId,
    Geography,
    Gender,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    IsActiveMember,
    Exited
FROM customers
WHERE
    Exited = 0  -- Still active
    AND (
        (IsActiveMember = 0 AND Balance > 100000)  -- Inactive with high balance
        OR (NumOfProducts = 1 AND Tenure <= 2)  -- New customers with single product
        OR (Age > 50 AND NumOfProducts = 1)  -- Older with limited engagement
        OR Balance = 0  -- Zero balance accounts
    )
ORDER BY Balance DESC
LIMIT 100;


-- ============================================
-- 9. VALUABLE CUSTOMERS WHO CHURNED
-- ============================================

-- High-value customers who churned
SELECT
    CustomerId,
    Geography,
    Age,
    Tenure,
    Balance,
    NumOfProducts,
    EstimatedSalary
FROM customers
WHERE Exited = 1
  AND Balance > 100000
ORDER BY Balance DESC
LIMIT 50;


-- ============================================
-- 10. MULTI-FACTOR CHURN ANALYSIS
-- ============================================

-- Churn by geography and active status
SELECT
    Geography,
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS status,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY Geography, IsActiveMember
ORDER BY Geography, IsActiveMember;


-- Churn by number of products and gender
SELECT
    NumOfProducts,
    Gender,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY NumOfProducts, Gender
ORDER BY NumOfProducts, Gender;


-- ============================================
-- 11. CUSTOMER SEGMENTATION
-- ============================================

-- VIP customers (high balance, active, multiple products)
SELECT
    COUNT(*) AS vip_customers,
    SUM(Exited) AS vip_churned,
    ROUND(AVG(Exited) * 100, 2) AS vip_churn_rate
FROM customers
WHERE Balance > 150000
  AND NumOfProducts >= 2
  AND IsActiveMember = 1;


-- At-risk segment (inactive or single product)
SELECT
    COUNT(*) AS at_risk_customers,
    SUM(Exited) AS at_risk_churned,
    ROUND(AVG(Exited) * 100, 2) AS at_risk_churn_rate
FROM customers
WHERE IsActiveMember = 0
   OR NumOfProducts = 1;


-- ============================================
-- 12. RETENTION INSIGHTS
-- ============================================

-- Compare churned vs retained customers
SELECT
    CASE WHEN Exited = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*) AS customers,
    AVG(Age) AS avg_age,
    AVG(Tenure) AS avg_tenure,
    AVG(Balance) AS avg_balance,
    AVG(NumOfProducts) AS avg_products,
    AVG(CreditScore) AS avg_credit_score,
    AVG(CASE WHEN IsActiveMember = 1 THEN 100 ELSE 0 END) AS pct_active
FROM customers
GROUP BY Exited;
