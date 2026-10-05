-- Feature table for day-ahead ERCOT demand forecasting (SQLite).
--
-- One row per hour. Every demand-based feature looks back at least 24 hours, so the model
-- only uses information available the day before: a true day-ahead forecast, no leakage.
-- Rows are ordered by UTC time (ts) so daylight-saving shifts don't break the lags;
-- calendar features use local Central time (local_ts) because people's routines follow the clock.

WITH hourly AS (
    SELECT d.ts, d.local_ts, d.demand_mwh, w.temp_c
    FROM demand AS d
    LEFT JOIN weather AS w ON w.ts = d.ts
)
SELECT
    ts,
    local_ts,
    CAST(strftime('%H', local_ts) AS INTEGER) AS hour,
    CAST(strftime('%w', local_ts) AS INTEGER) AS day_of_week,   -- 0 = Sunday
    CAST(strftime('%m', local_ts) AS INTEGER) AS month,
    CAST(strftime('%j', local_ts) AS INTEGER) AS day_of_year,
    temp_c,
    LAG(temp_c, 24)      OVER w AS temp_lag_24h,   -- temperature at the same hour yesterday
    LAG(demand_mwh, 24)  OVER w AS lag_24h,    -- same hour yesterday
    LAG(demand_mwh, 168) OVER w AS lag_168h,   -- same hour last week
    AVG(demand_mwh) OVER (ORDER BY ts ROWS BETWEEN 191 PRECEDING AND 24 PRECEDING) AS avg_prior_week,
    demand_mwh AS target
FROM hourly
WINDOW w AS (ORDER BY ts)
ORDER BY ts;
