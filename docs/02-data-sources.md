| Source | Location | Refresh | Owner | Notes |
|---|---|---|---|---|
| NYC TLC yellow taxi trip records, January 2025 | https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet | Static, no refresh | NYC Taxi and Limousine Commission | Data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf. File SHA-256 recorded at the first download on 2026-11-17 (see Source file checks). |

## Source file checks
- Link check (HEAD request, checked 2026-11-17): status 200, content-length 59158238 bytes (about 59 MB), last-modified Wed, 23 Apr 2025 16:31:58 GMT.
- First download: 2026-11-17, 59158238 bytes (matches the content-length above). SHA-256: `9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a`.
- Expected SHA-256 is stored in `tests/reference/source_sha256.txt`. The TLC publishes no official hash, so the first download is trusted and later changes are detected. The load step and `tests/data_gates/test_source_file_hash.py` fail if the file on disk differs from that record.

## Data layer decisions
- Partition format: YYYY-MM, one partition: 2025-01.
- Trip key: the source has no natural key. `trip_id` is a hash of the partition and all source columns plus an occurrence number among identical rows. The key keeps any exact duplicates distinct, and they stay in. Duplicates were measured on the training days only (pickups Jan 1 to 24): none were found. The test week was not inspected, so duplicates there are unknown.
- Staging rename map: VendorID to vendor_id, tpep_pickup_datetime to pickup_at, tpep_dropoff_datetime to dropoff_at, RatecodeID to rate_code_id, PULocationID to pickup_zone_id, DOLocationID to dropoff_zone_id, Airport_fee to airport_fee. Other columns lowercased. trip_distance unchanged.
- Mart columns: trip_id, pickup_at, dropoff_at, pickup_zone_id, dropoff_zone_id, trip_distance. passenger_count stays in staging only.
- Drop reasons are exclusive and counted in order: date window first, then dropoff before pickup.
- Impossible values (distance, speed, zones 264 and 265, passenger_count): kept in staging and the mart, reported by the notebook. The scope rules applied in the Python code (duration, distance, speed, zone 264, embargo) are in ADR 7 with the rows each removes on the training days.
- Download settings in `config.py`: recorded in ADR 6.

## Dropped row counts
First real run, 2026-11-17. Raw rows equal kept rows plus dropped rows: 3475226 = 3475080 + 22 + 124. The mart has 3475080 rows (one per kept row).

| Reason | Rows |
|---|---|
| Raw rows | 3475226 |
| Dropped: pickup outside 2025-01-01 to 2025-01-31 | 22 |
| Dropped: dropoff before pickup | 124 |
| Kept (staging output) | 3475080 |

## Exploration findings (training days, pickups Jan 1 to 24)
Source: `notebooks/01_exploration.ipynb`, run on `stg_trips` rows with pickup from 2025-01-01 up to but not including 2025-01-25 (2641485 rows). The test week is not read. The notebook reads `DB_CONN` from the `.env` file itself when it is not set, because it runs from `notebooks/` where `.env` is not loaded; this is an exception for the notebook only, and the package code reads the environment only. Re-run with `make notebook-run`. Findings only; decisions go in ADR 7.

- Exact duplicates: none. 2641485 rows and 2641485 distinct rows across the 20 source columns, 0 duplicate groups.
- Distance: 0 null, 0 negative, 67392 zero (46620 of them inside the 1 to 60 minute trips). Quantiles: p50 1.670 miles, p99 19.620, p99.99 60.309, max 276423.570. The top 20 values are all between 109183.020 and 276423.570 miles. 117 trips of 1 to 60 minutes have more than 50 miles.
- Implied speed (distance / duration hours, duration above 0): 766 rows above 80 mph, 704 above 100, 616 above 150, out of 2640428. Inside the 1 to 60 minute trips: 228 above 80, 190 above 100, 142 above 150, out of 2587420. The highest speeds are in the hundreds of thousands of mph.
- Duration: 1057 zero-duration rows, 29589 positive but under 1 minute, 23419 over 60 minutes, 15 over 24 hours, 2587420 inside 1 to 60 minutes. Outside 1 to 60 minutes: 54065 rows (2.0468%). The scope is not applied in the notebook.
- Zones 264 and 265: as pickup 7326 rows (0.277%; 6212 for 264, 1114 for 265); as dropoff 18669 rows (0.707%; 9245 for 264, 9424 for 265); either side 21948 rows (0.831%). No zone id outside 1 to 265 and no null zone ids. For 1 to 60 minute trips, median duration is 11.650 minutes for other pickups, 13.760 for pickup 265, 11.480 for pickup 264; for dropoff 265 it is 29.350 minutes with median distance 14.430 miles (other dropoffs 11.630 minutes, 1.670 miles); dropoff 264 is 10.900 minutes, 1.600 miles.
- passenger_count: 397120 null (15.034%), 19230 zero, 1765244 with one passenger (66.828%), values 1 to 6 total 2225122, and 13 rows above 6 (7: 3 rows, 8: 7 rows, 9: 3 rows). The column stays in staging only.
- Congestion pricing (1 to 60 minute trips; before = Jan 1 to 4 with 355540 trips, after = Jan 5 to 24 with 2231880 trips):
  - Median duration 12.170 minutes before, 11.580 after. Mean 14.900 before, 14.090 after. p25 7.370 and 7.320, p75 19.470 and 17.950.
  - Weekdays: median 12.300 before, 11.750 after. Weekend: 11.800 before, 11.070 after.
  - Minutes per mile (distance above 0.5): median 5.826 before, 6.126 after. Weekdays 5.748 before, 6.348 after. Weekend 6.035 before, 5.537 after.
  - Matched weekdays (Thu Jan 2 and Fri Jan 3 against Thursdays and Fridays after): Thursday median 12.420 before, 12.130 after; Friday 12.020 before, 12.000 after. Thursday and Friday together, minutes per mile: 6.145 before, 6.523 after. The hourly table is in the notebook; weekday 8 o'clock median is higher after (Thu and Fri together 10.070 before, 12.120 after) and several night hours are lower after.
  - Confounds: only 4 days before, Jan 1 is a holiday, the weekday mix differs (before has no Sunday to Tuesday), and Jan 20 is a public holiday after. The matched comparison is a single day per weekday before. The medians are close and move in different directions by hour and by measure, so the data does not show a clear single effect.
  - `cbd_congestion_fee` is a post-trip column, used only to see which trips were charged: 0 charged trips on Jan 1 to 3, 401 on Jan 4 (0.410%), 51509 on Jan 5 (64.690%), 64.660% to 75.890% on later days. Values: 0.750 (1637487 rows), 0.000 (1003519), -0.750 (479). It is not a feature.
