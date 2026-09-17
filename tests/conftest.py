"""Shared fixtures. No GPU, no network, no clock, no tokenizer."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prefixkv import PrefixCache, load_trace  # noqa: E402

TRACE_DIR = os.path.join(ROOT, "tests", "traces")

ALL_TRACES = ["EXAMPLE_system_prompt.csv", "EXAMPLE_multiturn.csv",
              "EXAMPLE_agent.csv", "EXAMPLE_fewshot.csv",
              "EXAMPLE_no_sharing.csv", "EXAMPLE_shared_last.csv",
              "EXAMPLE_tiny.csv"]

#: Bigger than any trace here needs, so capacity never binds and the cache reaches
#: the infinite-memory ceiling.
UNLIMITED = 10 ** 7


def trace_path(name):
    return os.path.join(TRACE_DIR, name)


@pytest.fixture
def tiny():
    return load_trace(trace_path("EXAMPLE_tiny.csv"))


@pytest.fixture
def system_prompt():
    return load_trace(trace_path("EXAMPLE_system_prompt.csv"))


@pytest.fixture
def shared_last():
    return load_trace(trace_path("EXAMPLE_shared_last.csv"))


@pytest.fixture
def no_sharing():
    return load_trace(trace_path("EXAMPLE_no_sharing.csv"))


@pytest.fixture
def multiturn():
    return load_trace(trace_path("EXAMPLE_multiturn.csv"))


@pytest.fixture
def agent():
    return load_trace(trace_path("EXAMPLE_agent.csv"))


def unlimited(block_size=16, concurrency=1):
    return PrefixCache(UNLIMITED, block_size, concurrency)
