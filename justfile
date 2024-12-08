@install_uv:
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

setup: install_uv
    uv sync --all-extras --all-groups

bump part='patch': install_uv
    uv run bump-my-version bump {{part}} --verbose

release: install_uv
    rm -rf dist
    uv build --no-sources
    uv publish

changelog: install_uv
    uvx git-changelog --output CHANGELOG.md

ruff: install_uv
    uvx ruff check --fix
    uvx ruff format

test: install_uv
    uv run tox -p
