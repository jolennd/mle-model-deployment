# ADR 4: Great Expectations not used
Status: accepted
Context: The data is one static file. The checks it needs are: pickups in January 2025, dropoff after pickup, no missing keys, valid zone IDs, and the file hash. dbt tests already express these as schema tests and SQL tests, and pytest runs them as one case.
Decision: Great Expectations is not added. dbt tests cover the tables. pytest covers the file hash, the dropped row counts and any check on Python code, including the 1 to 60 minute scope in the label code.
Consequences: One tool less to install and configure. No generated data docs and no profile-based expectations. Checks live in SQL and pytest, so they read as code in review. If checks must run on data outside the warehouse, a new ADR reopens this.
