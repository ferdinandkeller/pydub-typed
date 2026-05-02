set shell := ["zsh", "-cu"]
set dotenv-load

alias f := format
alias t := typecheck

@default:
    just --list

@install:
    uv sync --locked

@build:
    uv build

@format:
    uv run ruff check --fix
    uv run ruff format

@typecheck:
    uv run pyright src

# @unit:
#     uv run pytest -m unit

# @integration:
#     uv run pytest -m integration

# @test:
#     just unit integration
