import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))     # tests/helpers.py


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end learning runs (a few seconds each)")
