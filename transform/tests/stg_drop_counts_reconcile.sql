-- Bad row: raw row count differs from kept + dropped counts. Zero rows = pass.
with raw_count as (
    select count(*) as n from {{ source('raw', 'raw_trips') }}
),

accounted as (
    select
        coalesce(sum(row_count) filter (
            where reason in ('kept', 'dropped_date_window', 'dropped_dropoff_before_pickup')
        ), 0) as n
    from {{ ref('stg_trips_drop_counts') }}
)

select raw_count.n as raw_rows, accounted.n as accounted_rows
from raw_count
cross join accounted
where raw_count.n <> accounted.n
