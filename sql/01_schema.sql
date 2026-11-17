-- Schema: one row per policy, one row per claim.
-- A policy can have zero, one or many claims (one-to-many on policy_id).

DROP VIEW  IF EXISTS v_pure_premium;
DROP VIEW  IF EXISTS v_severity_model;
DROP VIEW  IF EXISTS v_frequency_model;
DROP VIEW  IF EXISTS v_policy_claims;
DROP VIEW  IF EXISTS v_policy_split;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS policies;

CREATE TABLE policies (
    policy_id    INTEGER PRIMARY KEY,
    claim_count  INTEGER NOT NULL CHECK (claim_count >= 0),
    exposure     REAL    NOT NULL CHECK (exposure > 0),   -- policy-years in force
    veh_power    INTEGER NOT NULL,
    veh_age      INTEGER NOT NULL CHECK (veh_age >= 0),
    driver_age   INTEGER NOT NULL CHECK (driver_age >= 18),
    bonus_malus  INTEGER NOT NULL,                        -- 50 = max discount, >100 = surcharge
    veh_brand    TEXT    NOT NULL,
    veh_gas      TEXT    NOT NULL,
    area         TEXT    NOT NULL,                        -- A (rural) .. F (urban centre)
    density      INTEGER NOT NULL,                        -- inhabitants per km2
    region       TEXT    NOT NULL
);

CREATE TABLE claims (
    claim_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id     INTEGER NOT NULL REFERENCES policies (policy_id),
    claim_amount  REAL    NOT NULL CHECK (claim_amount > 0)
);

CREATE INDEX idx_claims_policy ON claims (policy_id);
