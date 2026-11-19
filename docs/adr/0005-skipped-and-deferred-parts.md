# ADR 5: Skipped and deferred parts
Status: accepted
Context: The project is solo, local and ends in a README with a trained model behind one endpoint. Several standard steps add work this scope does not need. A skipped step runs only after a new ADR supersedes this one.
Decision:
- Notebook refactor: skipped. There is no notebook. Code is written in the package from the start, and the CLI (train, tune, evaluate) comes with the training step. The notebook-equals-package test does not apply.
- Orchestration: skipped, see ADR 3.
- Resource API: skipped. Nothing serves tables or records from the warehouse. The only endpoint is /predict.
- Cloud deploy: deferred. The deploy setting stays "later". Every port binds to 127.0.0.1.
- CI/CD: CI runs lint and the fast tests on every push and pull request (.github/workflows/tests.yml: uv sync --locked, then make validate-gates). A full CI/CD pipeline is deferred.
- Monitoring: deferred. The monitoring setting stays "later". MLflow run history is the only record.
- Rigorous evaluation: planned, not skipped. The tuning plan (docs/tuning-plan.md) and R9 cover it.
- Docker: optional and later. If added, it packages the /predict service, publishes only on 127.0.0.1, and reaches MLflow through host.docker.internal, which needs its own ADR. The container step normally follows the resource API, which is skipped here, so that ADR also records the change.
Consequences: Tests marked docker, deploy, data and model run only when started by hand, because CI runs the fast tests only. No drift or latency tracking after serving. The training, tuning and endpoint steps are listed as planned in the traceability table and are not built here.
