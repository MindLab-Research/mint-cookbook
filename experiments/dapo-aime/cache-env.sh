#!/usr/bin/env bash
# Shared-cache defaults for this experiment. Source this before uv/pip/python commands.

_mint_cache_root_default="${XDG_CACHE_HOME:-${HOME}/.cache}/mint-cookbook"
export MINT_SHARED_CACHE_ROOT="${MINT_SHARED_CACHE_ROOT:-${_mint_cache_root_default}}"

# Tooling caches that otherwise land on the full root overlay.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${MINT_SHARED_CACHE_ROOT}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MINT_SHARED_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MINT_SHARED_CACHE_ROOT}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MINT_SHARED_CACHE_ROOT}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${MINT_SHARED_CACHE_ROOT}/wandb}"

# Model and dataset caches on shared storage.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MINT_SHARED_CACHE_ROOT}/hf-datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p   "${UV_CACHE_DIR}"   "${XDG_CACHE_HOME}"   "${PIP_CACHE_DIR}"   "${MPLCONFIGDIR}"   "${WANDB_DIR}"   "${HF_DATASETS_CACHE}"   "${HUGGINGFACE_HUB_CACHE}"
