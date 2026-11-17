-- Modelling views. The Python models read only from these views, so every
-- cleaning rule and feature band is defined once, in SQL.

-- Deterministic train / validation / test split.
-- Knuth multiplicative hash of policy_id mapped to a bucket 0-99, so the split
-- is reproducible in pure SQL and does not depend on row order.
CREATE VIEW v_policy_split AS
SELECT policy_id,
       CASE WHEN bucket < 20 THEN 'test'
            WHEN bucket < 36 THEN 'valid'
            ELSE 'train' END AS split
FROM (
    SELECT policy_id,
           ((policy_id * 2654435761) % 4294967296) * 100 / 4294967296 AS bucket
    FROM policies
);

-- Claims rolled up to policy level, with each claim capped at 100,000 so a
-- handful of very large losses do not dominate the severity model.
CREATE VIEW v_policy_claims AS
SELECT policy_id,
       COUNT(*)                        AS n_claims,
       SUM(claim_amount)               AS total_amount,
       SUM(MIN(claim_amount, 100000))  AS total_amount_capped
FROM claims
GROUP BY policy_id;

-- Frequency model input: one row per policy.
--   exposure capped at 1 year, claim count capped at 4 (data-entry outliers)
--   banded features are for the GLM; raw features are for gradient boosting
CREATE VIEW v_frequency_model AS
SELECT p.policy_id,
       s.split,
       MIN(p.exposure, 1.0)              AS exposure,
       MIN(p.claim_count, 4)             AS claim_count,
       -- raw features
       p.veh_power, p.veh_age, p.driver_age,
       MIN(p.bonus_malus, 150)           AS bonus_malus,
       p.veh_brand, p.veh_gas, p.area, p.density, p.region,
       -- banded features
       CASE WHEN p.driver_age <= 20 THEN '18-20'
            WHEN p.driver_age <= 25 THEN '21-25'
            WHEN p.driver_age <= 30 THEN '26-30'
            WHEN p.driver_age <= 40 THEN '31-40'
            WHEN p.driver_age <= 50 THEN '41-50'
            WHEN p.driver_age <= 70 THEN '51-70'
            ELSE '71+' END               AS driver_age_band,
       CASE WHEN p.veh_age = 0   THEN '00 new'
            WHEN p.veh_age <= 5  THEN '01-05'
            WHEN p.veh_age <= 10 THEN '06-10'
            WHEN p.veh_age <= 15 THEN '11-15'
            ELSE '16+' END               AS veh_age_band,
       CASE WHEN p.veh_power >= 9 THEN '9+'
            ELSE CAST(p.veh_power AS TEXT) END AS veh_power_band
FROM policies p
JOIN v_policy_split s ON s.policy_id = p.policy_id;

-- Severity model input: one row per policy that had at least one claim.
-- Target is the average capped cost per claim, weighted by number of claims.
CREATE VIEW v_severity_model AS
SELECT f.*,
       c.n_claims,
       c.total_amount_capped,
       c.total_amount_capped / c.n_claims AS avg_severity
FROM v_frequency_model f
JOIN v_policy_claims c ON c.policy_id = f.policy_id;

-- Pure premium check: every policy with its observed (capped) loss.
CREATE VIEW v_pure_premium AS
SELECT f.*,
       COALESCE(c.total_amount_capped, 0) AS loss_capped,
       COALESCE(c.total_amount, 0)        AS loss_uncapped
FROM v_frequency_model f
LEFT JOIN v_policy_claims c ON c.policy_id = f.policy_id;
