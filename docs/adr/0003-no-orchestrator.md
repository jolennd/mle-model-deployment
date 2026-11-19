# ADR 3: No orchestrator
Status: accepted
Context: The project asks for something easy to build, run and review on one machine. The source is one static file with no refresh, there is no schedule, and the steps (transform, train, tune, serve) run one after another by hand. An orchestrator adds a scheduler or server process and its own state.
Decision: No orchestrator. The orchestrator setting in stack.toml stays "later". make targets run the steps in order, and the README lists the order. Each command is safe to rerun and exits non-zero on failure, so make stops at the first failed step.
Consequences: No scheduling, automatic retries or run history beyond MLflow runs. Reruns are started by hand. Safe reruns come from the commands themselves: the transform rebuilds its tables and training uses a fixed seed. If the data starts refreshing or steps need a schedule, a new ADR supersedes this one and changes stack.toml.
