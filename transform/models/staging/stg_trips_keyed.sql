-- Staging with the trip key. Exact duplicate source rows stay in; the occurrence
-- number among identical rows makes each key unique. No deduplication.
with renamed as (
    select
        "VendorID" as vendor_id,
        tpep_pickup_datetime as pickup_at,
        tpep_dropoff_datetime as dropoff_at,
        passenger_count,
        trip_distance,
        "RatecodeID" as rate_code_id,
        store_and_fwd_flag,
        "PULocationID" as pickup_zone_id,
        "DOLocationID" as dropoff_zone_id,
        payment_type,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        improvement_surcharge,
        total_amount,
        congestion_surcharge,
        "Airport_fee" as airport_fee,
        cbd_congestion_fee,
        source_partition
    from {{ source('raw', 'raw_trips') }}
),

kept as (
    select *
    from renamed
    where coalesce(pickup_at >= timestamp '2025-01-01 00:00:00'
                   and pickup_at < timestamp '2025-02-01 00:00:00', false)
      and coalesce(dropoff_at >= pickup_at, false)
),

fingerprinted as (
    select
        *,
        md5(concat_ws('|',
            coalesce(source_partition, '<null>'),
            coalesce(cast(vendor_id as varchar), '<null>'),
            coalesce(cast(pickup_at as varchar), '<null>'),
            coalesce(cast(dropoff_at as varchar), '<null>'),
            coalesce(cast(passenger_count as varchar), '<null>'),
            coalesce(cast(trip_distance as varchar), '<null>'),
            coalesce(cast(rate_code_id as varchar), '<null>'),
            coalesce(store_and_fwd_flag, '<null>'),
            coalesce(cast(pickup_zone_id as varchar), '<null>'),
            coalesce(cast(dropoff_zone_id as varchar), '<null>'),
            coalesce(cast(payment_type as varchar), '<null>'),
            coalesce(cast(fare_amount as varchar), '<null>'),
            coalesce(cast(extra as varchar), '<null>'),
            coalesce(cast(mta_tax as varchar), '<null>'),
            coalesce(cast(tip_amount as varchar), '<null>'),
            coalesce(cast(tolls_amount as varchar), '<null>'),
            coalesce(cast(improvement_surcharge as varchar), '<null>'),
            coalesce(cast(total_amount as varchar), '<null>'),
            coalesce(cast(congestion_surcharge as varchar), '<null>'),
            coalesce(cast(airport_fee as varchar), '<null>'),
            coalesce(cast(cbd_congestion_fee as varchar), '<null>')
        )) as row_hash
    from kept
)

select
    md5(row_hash || '|' || cast(
        row_number() over (partition by row_hash order by row_hash) as varchar
    )) as trip_id,
    * exclude (row_hash)
from fingerprinted
