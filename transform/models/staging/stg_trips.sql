-- Typed and renamed trips: the source columns only, same rows as stg_trips_keyed.
select
    vendor_id,
    pickup_at,
    dropoff_at,
    passenger_count,
    trip_distance,
    rate_code_id,
    store_and_fwd_flag,
    pickup_zone_id,
    dropoff_zone_id,
    payment_type,
    fare_amount,
    extra,
    mta_tax,
    tip_amount,
    tolls_amount,
    improvement_surcharge,
    total_amount,
    congestion_surcharge,
    airport_fee,
    cbd_congestion_fee
from {{ ref('stg_trips_keyed') }}
