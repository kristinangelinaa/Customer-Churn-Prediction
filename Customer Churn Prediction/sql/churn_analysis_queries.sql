-- Customer Churn Analysis SQL Queries
-- SaaS Business Churn Analytics

-- ============================================
-- 1. OVERALL CHURN METRICS
-- ============================================

-- Overall churn rate and customer summary
SELECT
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    COUNT(*) - SUM(churned) AS active_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_percent,
    ROUND(AVG(CASE WHEN churned = 0 THEN 1 ELSE 0 END) * 100, 2) AS retention_rate_percent
FROM customers;


-- Revenue impact of churn
SELECT
    SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END) AS lost_mrr,
    SUM(CASE WHEN churned = 0 THEN monthly_charges ELSE 0 END) AS active_mrr,
    SUM(monthly_charges) AS total_mrr,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END) * 100.0 / SUM(monthly_charges), 2) AS mrr_churn_rate
FROM customers;


-- ============================================
-- 2. CHURN BY CONTRACT TYPE
-- ============================================

SELECT
    contract_type,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate,
    AVG(tenure_months) AS avg_tenure,
    AVG(monthly_charges) AS avg_monthly_charges
FROM customers
GROUP BY contract_type
ORDER BY churn_rate DESC;


-- ============================================
-- 3. CHURN BY SUBSCRIPTION PLAN
-- ============================================

SELECT
    plan,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate,
    AVG(tenure_months) AS avg_tenure,
    SUM(monthly_charges) AS total_mrr
FROM customers
GROUP BY plan
ORDER BY churn_rate DESC;


-- ============================================
-- 4. CHURN BY TENURE (COHORT ANALYSIS)
-- ============================================

-- Churn by tenure buckets
SELECT
    CASE
        WHEN tenure_months <= 3 THEN '0-3 months'
        WHEN tenure_months <= 6 THEN '4-6 months'
        WHEN tenure_months <= 12 THEN '7-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        ELSE '24+ months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY tenure_group
ORDER BY
    CASE tenure_group
        WHEN '0-3 months' THEN 1
        WHEN '4-6 months' THEN 2
        WHEN '7-12 months' THEN 3
        WHEN '13-24 months' THEN 4
        ELSE 5
    END;


-- ============================================
-- 5. CHURN BY USAGE METRICS
-- ============================================

-- Churn by login frequency
SELECT
    CASE
        WHEN avg_login_frequency < 5 THEN 'Very Low (<5 days/mo)'
        WHEN avg_login_frequency < 10 THEN 'Low (5-10 days/mo)'
        WHEN avg_login_frequency < 20 THEN 'Medium (10-20 days/mo)'
        ELSE 'High (20+ days/mo)'
    END AS login_frequency_group,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY login_frequency_group
ORDER BY churn_rate DESC;


-- Churn by feature usage score
SELECT
    CASE
        WHEN feature_usage_score < 25 THEN 'Very Low (0-25)'
        WHEN feature_usage_score < 50 THEN 'Low (25-50)'
        WHEN feature_usage_score < 75 THEN 'Medium (50-75)'
        ELSE 'High (75-100)'
    END AS usage_group,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY usage_group
ORDER BY churn_rate DESC;


-- ============================================
-- 6. CHURN BY SUPPORT METRICS
-- ============================================

-- Churn by number of support tickets
SELECT
    CASE
        WHEN support_tickets = 0 THEN '0 tickets'
        WHEN support_tickets <= 2 THEN '1-2 tickets'
        WHEN support_tickets <= 5 THEN '3-5 tickets'
        ELSE '6+ tickets'
    END AS support_ticket_group,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY support_ticket_group
ORDER BY churn_rate DESC;


-- ============================================
-- 7. CHURN BY ENGAGEMENT METRICS
-- ============================================

-- Churn by email engagement
SELECT
    CASE
        WHEN email_open_rate < 20 THEN 'Very Low (<20%)'
        WHEN email_open_rate < 40 THEN 'Low (20-40%)'
        WHEN email_open_rate < 60 THEN 'Medium (40-60%)'
        ELSE 'High (60%+)'
    END AS email_engagement,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY email_engagement
ORDER BY churn_rate DESC;


-- Churn by integrations
SELECT
    CASE
        WHEN num_integrations = 0 THEN 'No integrations'
        WHEN num_integrations <= 2 THEN '1-2 integrations'
        WHEN num_integrations <= 5 THEN '3-5 integrations'
        ELSE '6+ integrations'
    END AS integration_group,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY integration_group
ORDER BY churn_rate DESC;


-- ============================================
-- 8. HIGH-RISK CUSTOMERS (CHURN PREDICTION)
-- ============================================

-- Identify high-risk customers (not yet churned)
SELECT
    customer_id,
    plan,
    tenure_months,
    monthly_charges,
    contract_type,
    avg_login_frequency,
    feature_usage_score,
    support_tickets,
    days_since_last_login,
    payment_failures
FROM customers
WHERE churned = 0
    AND (
        -- Risk factors
        (contract_type = 'Month-to-Month' AND tenure_months < 6)
        OR avg_login_frequency < 5
        OR feature_usage_score < 30
        OR payment_failures > 0
        OR days_since_last_login > 30
        OR support_tickets > 5
    )
ORDER BY
    CASE WHEN payment_failures > 0 THEN 1 ELSE 2 END,
    days_since_last_login DESC,
    avg_login_frequency ASC
LIMIT 100;


-- ============================================
-- 9. CHURN BY COMPANY SIZE
-- ============================================

SELECT
    company_size,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate,
    AVG(monthly_charges) AS avg_monthly_charges
FROM customers
GROUP BY company_size
ORDER BY
    CASE company_size
        WHEN '1-10' THEN 1
        WHEN '11-50' THEN 2
        WHEN '51-200' THEN 3
        WHEN '201-1000' THEN 4
        ELSE 5
    END;


-- ============================================
-- 10. CHURN BY INDUSTRY
-- ============================================

SELECT
    industry,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY industry
ORDER BY churn_rate DESC;


-- ============================================
-- 11. PAYMENT-RELATED CHURN
-- ============================================

-- Churn by payment failures
SELECT
    payment_failures,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY payment_failures
ORDER BY payment_failures;


-- Churn by auto-renewal status
SELECT
    CASE WHEN auto_renewal = 1 THEN 'Auto-renewal ON' ELSE 'Auto-renewal OFF' END AS renewal_status,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY auto_renewal;


-- ============================================
-- 12. CUSTOMER LIFETIME VALUE (CHURNED vs ACTIVE)
-- ============================================

SELECT
    CASE WHEN churned = 1 THEN 'Churned' ELSE 'Active' END AS customer_status,
    COUNT(*) AS num_customers,
    AVG(total_charges) AS avg_lifetime_value,
    AVG(tenure_months) AS avg_tenure_months,
    AVG(monthly_charges) AS avg_monthly_charges
FROM customers
GROUP BY churned;


-- ============================================
-- 13. ADD-ON IMPACT ON CHURN
-- ============================================

SELECT
    CASE WHEN has_addons = 1 THEN 'Has Add-ons' ELSE 'No Add-ons' END AS addon_status,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate
FROM customers
GROUP BY has_addons;
