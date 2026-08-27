.PHONY: setup install-hooks clean-all

# One-command onboarding: installs dependencies, sets up the pre-commit
# hook, and runs the test suite so you know right away if something's
# wrong. See CONTRIBUTING.md.
setup:
	uv sync
	@scripts/install-hooks.sh
	uv run pytest -q

# One-time per-clone setup — see AGENTS.md. Also runs as part of `make setup`.
install-hooks:
	@scripts/install-hooks.sh

# Add one `clean-<experiment>` target per experiment as they're added, e.g.:
#   clean-my-experiment:
#   	rm -rf data/generated/my_experiment results/my_experiment
# then list it under clean-all.

clean-all:
	@echo "No experiments registered yet — add clean-<experiment> targets as you add experiments."
