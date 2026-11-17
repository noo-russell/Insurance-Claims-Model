-- Portfolio analysis: observed frequency and severity before any modelling.
-- Frequency = claims per policy-year of exposure. Severity = average cost per claim.

-- name: Portfolio summary
SELECT COUNT(*)                                             AS policies,
       ROUND(SUM(p.exposure), 0)                            AS exposure_years,
       SUM(p.claim_count)                                   AS claims,
       ROUND(1.0 * SUM(p.claim_count) / SUM(p.exposure), 4) AS claim_frequency,
       ROUND((SELECT AVG(claim_amount) FROM claims), 0)     AS avg_severity,
       ROUND((SELECT SUM(claim_amount) FROM claims) / SUM(p.exposure), 0) AS pure_premium
FROM policies p;

-- name: Frequency and severity by driver age band
WITH banded AS (
    SELECT policy_id, exposure, claim_count,
           CASE WHEN driver_age <= 20 THEN '18-20'
                WHEN driver_age <= 25 THEN '21-25'
                WHEN driver_age <= 30 THEN '26-30'
                WHEN driver_age <= 40 THEN '31-40'
                WHEN driver_age <= 50 THEN '41-50'
                WHEN driver_age <= 70 THEN '51-70'
                ELSE '71+' END AS driver_age_band
    FROM policies
),
losses AS (
    SELECT policy_id, SUM(claim_amount) AS total_amount
    FROM claims
    GROUP BY policy_id
)
SELECT b.driver_age_band,
       ROUND(SUM(b.exposure), 0)                              AS exposure_years,
       SUM(b.claim_count)                                     AS claims,
       ROUND(1.0 * SUM(b.claim_count) / SUM(b.exposure), 4)   AS claim_frequency,
       ROUND(SUM(l.total_amount) / SUM(b.claim_count), 0)     AS avg_severity,
       ROUND(SUM(COALESCE(l.total_amount, 0)) / SUM(b.exposure), 0) AS pure_premium
FROM banded b
LEFT JOIN losses l ON l.policy_id = b.policy_id
GROUP BY b.driver_age_band
ORDER BY b.driver_age_band;

-- name: Frequency by bonus-malus level
SELECT CASE WHEN bonus_malus = 50   THEN '1. 50 (max discount)'
            WHEN bonus_malus < 75   THEN '2. 51-74'
            WHEN bonus_malus < 100  THEN '3. 75-99'
            WHEN bonus_malus = 100  THEN '4. 100 (base)'
            ELSE                         '5. over 100 (surcharge)' END AS bonus_malus_band,
       COUNT(*)                                             AS policies,
       ROUND(SUM(exposure), 0)                              AS exposure_years,
       ROUND(1.0 * SUM(claim_count) / SUM(exposure), 4)     AS claim_frequency
FROM policies
GROUP BY bonus_malus_band
ORDER BY bonus_malus_band;

-- name: Frequency by area, relative to the portfolio
SELECT area,
       ROUND(AVG(density), 0)                               AS avg_density,
       ROUND(SUM(exposure), 0)                              AS exposure_years,
       ROUND(1.0 * SUM(claim_count) / SUM(exposure), 4)     AS claim_frequency,
       ROUND( (1.0 * SUM(claim_count) / SUM(exposure))
            / (SELECT 1.0 * SUM(claim_count) / SUM(exposure) FROM policies), 2) AS relativity
FROM policies
GROUP BY area
ORDER BY area;

-- name: Five regions with the highest claim frequency
SELECT region,
       ROUND(SUM(exposure), 0)                              AS exposure_years,
       ROUND(1.0 * SUM(claim_count) / SUM(exposure), 4)     AS claim_frequency,
       RANK() OVER (ORDER BY 1.0 * SUM(claim_count) / SUM(exposure) DESC) AS freq_rank
FROM policies
GROUP BY region
ORDER BY freq_rank
LIMIT 5;

-- name: Claim severity percentiles
-- SQLite has no percentile function, so rank the claims with a window function.
WITH ranked AS (
    SELECT claim_amount,
           ROW_NUMBER() OVER (ORDER BY claim_amount) AS rn,
           COUNT(*)     OVER ()                      AS n
    FROM claims
)
SELECT ROUND(MAX(CASE WHEN rn <= 0.50  * n THEN claim_amount END), 0) AS p50,
       ROUND(MAX(CASE WHEN rn <= 0.90  * n THEN claim_amount END), 0) AS p90,
       ROUND(MAX(CASE WHEN rn <= 0.99  * n THEN claim_amount END), 0) AS p99,
       ROUND(MAX(CASE WHEN rn <= 0.999 * n THEN claim_amount END), 0) AS p999,
       ROUND(MAX(claim_amount), 0)                                    AS largest
FROM ranked;

-- name: Loss concentration: share of total cost from the largest claims
WITH ranked AS (
    SELECT claim_amount,
           NTILE(100) OVER (ORDER BY claim_amount DESC) AS pct_bucket,
           SUM(claim_amount) OVER ()                    AS total_cost
    FROM claims
)
SELECT 'largest 1% of claims'  AS claim_group,
       COUNT(*)                AS n_claims,
       ROUND(100.0 * SUM(claim_amount) / MAX(total_cost), 1) AS pct_of_total_cost
FROM ranked WHERE pct_bucket = 1
UNION ALL
SELECT 'largest 10% of claims', COUNT(*),
       ROUND(100.0 * SUM(claim_amount) / MAX(total_cost), 1)
FROM ranked WHERE pct_bucket <= 10;

-- name: Effect of capping each claim at 100,000
SELECT SUM(CASE WHEN claim_amount > 100000 THEN 1 ELSE 0 END)   AS claims_capped,
       ROUND(SUM(claim_amount), 0)                               AS total_cost,
       ROUND(SUM(MIN(claim_amount, 100000)), 0)                  AS total_cost_capped,
       ROUND(100.0 * (1 - SUM(MIN(claim_amount, 100000)) / SUM(claim_amount)), 1) AS pct_cost_removed
FROM claims;

-- name: Train / test split balance
SELECT split,
       COUNT(*)                                          AS policies,
       ROUND(SUM(exposure), 0)                           AS exposure_years,
       ROUND(1.0 * SUM(claim_count) / SUM(exposure), 4)  AS claim_frequency
FROM v_frequency_model
GROUP BY split
ORDER BY split;
