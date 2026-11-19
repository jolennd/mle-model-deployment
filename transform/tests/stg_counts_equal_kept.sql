-- Bad rows: stg_trips or mart_trips row count differs from kept. Zero rows = pass.
with kept as (
    select row_count as n
    from {{ ref('stg_trips_drop_counts') }}
    where reason = 'kept'
),

counts as (
    select 'stg_trips' as model_name, count(*) as n from {{ ref('stg_trips') }}
    union all
    select 'mart_trips' as model_name, count(*) as n from {{ ref('mart_trips') }}
)

select counts.model_name, counts.n as model_rows, kept.n as kept_rows
from counts
cross join kept
where counts.n <> kept.n
