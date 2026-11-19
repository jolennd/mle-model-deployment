-- Bad rows: pickup outside [2025-01-01, 2025-02-01), in staging or mart. Zero rows = pass.
select 'stg_trips' as model_name, pickup_at, dropoff_at
from {{ ref('stg_trips') }}
where pickup_at is null
   or pickup_at < timestamp '2025-01-01 00:00:00'
   or pickup_at >= timestamp '2025-02-01 00:00:00'

union all

select 'mart_trips' as model_name, pickup_at, dropoff_at
from {{ ref('mart_trips') }}
where pickup_at is null
   or pickup_at < timestamp '2025-01-01 00:00:00'
   or pickup_at >= timestamp '2025-02-01 00:00:00'
