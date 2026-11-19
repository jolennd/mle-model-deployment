-- One row per reason. Reasons are exclusive: the date window is counted first.
with flagged as (
    select
        coalesce(tpep_pickup_datetime >= timestamp '2025-01-01 00:00:00'
                 and tpep_pickup_datetime < timestamp '2025-02-01 00:00:00', false) as in_window,
        coalesce(tpep_dropoff_datetime >= tpep_pickup_datetime, false) as dropoff_ok
    from {{ source('raw', 'raw_trips') }}
)

select 'raw_rows' as reason, count(*) as row_count from flagged
union all
select 'dropped_date_window', count(*) from flagged where not in_window
union all
select 'dropped_dropoff_before_pickup', count(*) from flagged where in_window and not dropoff_ok
union all
select 'kept', count(*) from flagged where in_window and dropoff_ok
