# ADR 7: Scope rules for trips
Status: accepted
Context: The exploration notebook (docs/02-data-sources.md, findings section) found trips with zero or huge distances, implied speeds in the hundreds of thousands of mph, durations of zero or over 24 hours, and trips to or from zone 264 (Unknown) and 265 (Outside of NYC). The staging transform keeps these rows, so its drop counts stay as they are (date window and dropoff before pickup only). The brief puts the 1 to 60 minute scope in the Python label code. The same code must treat training rows and test week rows identically.
Decision: The Python scope code applies these rules. The same function is used for training rows and test week rows, and it is the only place they are applied. Bounds are inclusive.
1. Duration: keep trips of 1 to 60 minutes (the brief).
2. Distance: keep trip_distance from 0.1 to 100 miles. The API rejects other values with 422.
3. Speed: keep trips whose implied average speed (trip_distance divided by duration in hours) is 80 mph or less. This filter uses the target (duration), because it needs the label. It only defines which records are valid; it is applied identically to the test week; it cannot be applied to API requests, which carry no duration. The README names it next to the RMSE.
4. Zones: drop trips with zone 264 (Unknown) as pickup or dropoff. Keep 265 (Outside of NYC). The API accepts zone IDs 1 to 263 and 265 and rejects others with 422.
5. Embargo: as in the brief. Training keeps only trips whose dropoff is before 2025-01-25 00:00.
Congestion pricing (started 2025-01-05) is not modelled. The notebook found no clear single effect, and the four days before it are too few to separate it from day of week and the New Year holiday. The test week (Jan 25 to 31) lies fully after 2025-01-05, so the whole test week is on the same side of the change, and so are 20 of the 24 training days (Jan 5 to 24).
Rows removed on the training days (pickups Jan 1 to 24, 2641485 rows in stg_trips), counted with a read-only query on 2026-11-17. "Alone" counts each rule by itself. "Added" counts the rows a rule removes that earlier rules in this list did not:

| Rule | Alone | Added |
|---|---|---|
| 1. Duration outside 1 to 60 minutes | 54065 | 54065 |
| 2. Distance outside 0.1 to 100 miles | 83726 | 56282 |
| 3. Speed above 80 mph | 766 | 134 |
| 4. Zone 264 on either side | 12313 | 11135 |
| 5. Dropoff on or after 2025-01-25 | 1994 | 1902 |
| All rules | | 123518 |

Rows kept for training: 2641485 - 123518 = 2517967. The test week counts are not read here; they are reported by the evaluation step after the baseline lock exists.
Consequences: Reported RMSE covers valid trips only. The README says so next to the RMSE, together with the speed filter and the distance caveat from ADR 2. Predictions for requests outside the ranges are refused (422) or, for duration, not validated. Requests for zone 264 are refused, although 9245 training day rows (0.35 percent) had it as a dropoff and 6212 (0.24 percent) as a pickup. The speed filter removes trips with a bad distance reading but also fast real trips; at 80 mph that is a very small share. Changing a bound later needs a new ADR and a new locked baseline.
