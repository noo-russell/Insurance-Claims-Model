-- Data-quality checks. Each block is run by src/sql_report.py and written
-- to results/sql_report.md. Blocks are separated by "-- name:" lines.

-- name: Row counts
SELECT 'policies' AS table_name, COUNT(*) AS n_rows FROM policies
UNION ALL
SELECT 'claims', COUNT(*) FROM claims;

-- name: Claims with no matching policy (expect 0)
SELECT COUNT(*) AS orphan_claims
FROM claims c
LEFT JOIN policies p ON p.policy_id = c.policy_id
WHERE p.policy_id IS NULL;

-- name: Policies whose claim_count disagrees with the claims table (expect 0)
WITH counted AS (
    SELECT policy_id, COUNT(*) AS n_in_claims
    FROM claims
    GROUP BY policy_id
)
SELECT COUNT(*) AS mismatched_policies
FROM policies p
LEFT JOIN counted c ON c.policy_id = p.policy_id
WHERE p.claim_count <> COALESCE(c.n_in_claims, 0);

-- name: Exposure above one policy-year
SELECT COUNT(*)                                   AS policies_over_1yr,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM policies), 2) AS pct_of_policies,
       ROUND(MAX(exposure), 2)                    AS max_exposure
FROM policies
WHERE exposure > 1;

-- name: Distribution of claim counts per policy
SELECT claim_count,
       COUNT(*)                       AS policies,
       ROUND(SUM(exposure), 0)        AS exposure_years
FROM policies
GROUP BY claim_count
ORDER BY claim_count;

-- name: Implausible attribute values
SELECT SUM(CASE WHEN driver_age  > 90  THEN 1 ELSE 0 END) AS drivers_over_90,
       SUM(CASE WHEN veh_age     > 30  THEN 1 ELSE 0 END) AS vehicles_over_30y,
       SUM(CASE WHEN bonus_malus > 150 THEN 1 ELSE 0 END) AS bonus_malus_over_150
FROM policies;

-- name: Most common claim amounts (fixed settlements)
-- Thousands of claims share exactly the same amount: these are flat-rate
-- settlements between insurers, not individually assessed losses.
SELECT claim_amount,
       COUNT(*) AS n_claims,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM claims), 1) AS pct_of_claims
FROM claims
GROUP BY claim_amount
ORDER BY n_claims DESC
LIMIT 5;
