-- Bad rows: zone id null or outside 1 to 265. Zero rows = pass.
select pickup_zone_id, dropoff_zone_id
from {{ ref('stg_trips') }}
where pickup_zone_id is null
   or dropoff_zone_id is null
   or pickup_zone_id not between 1 and 265
   or dropoff_zone_id not between 1 and 265
