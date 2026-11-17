-- Bad rows: dropoff before pickup, in staging or mart. Zero rows = pass.
select 'stg_trips' as model_name, pickup_at, dropoff_at
from {{ ref('stg_trips') }}
where dropoff_at < pickup_at

union all

select 'mart_trips' as model_name, pickup_at, dropoff_at
from {{ ref('mart_trips') }}
where dropoff_at < pickup_at
