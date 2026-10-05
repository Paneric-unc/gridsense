-- Optional: the same feature query in Databricks SQL (Spark SQL).
-- Upload demand and weather as Delta tables first (see README, "Optional: Databricks").
-- Differences from SQLite: hour()/month()/dayofyear() instead of strftime, and
-- dayofweek() returns 1 = Sunday, so we subtract 1 to match the SQLite version.

CREATE OR REPLACE TABLE gridsense_features AS
WITH hourly AS (
    SELECT d.ts, d.local_ts, d.demand_mwh, w.temp_c
    FROM demand AS d
    LEFT JOIN weather AS w ON w.ts = d.ts
)
SELECT
    ts,
    local_ts,
    hour(local_ts)          AS hour,
    dayofweek(local_ts) - 1 AS day_of_week,
    month(local_ts)         AS month,
    dayofyear(local_ts)     AS day_of_year,
    temp_c,
    LAG(temp_c, 24)      OVER (ORDER BY ts) AS temp_lag_24h,
    LAG(demand_mwh, 24)  OVER (ORDER BY ts) AS lag_24h,
    LAG(demand_mwh, 168) OVER (ORDER BY ts) AS lag_168h,
    AVG(demand_mwh) OVER (ORDER BY ts ROWS BETWEEN 191 PRECEDING AND 24 PRECEDING) AS avg_prior_week,
    demand_mwh AS target
FROM hourly;
