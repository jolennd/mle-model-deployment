-- One row per trip. Allowed input columns only: no label, no post-trip columns.
select
    trip_id,
    pickup_at,
    dropoff_at,
    pickup_zone_id,
    dropoff_zone_id,
    trip_distance
from {{ ref('stg_trips_keyed') }}
