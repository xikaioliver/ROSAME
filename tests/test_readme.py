"""The README's quickstart runs as written."""
import re
import warnings
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_quickstart_runs(tmp_path, monkeypatch):
    code = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)[0]
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "README quickstart", "exec"), {})
    assert (tmp_path / "learned.pddl").read_text().startswith("(define (domain switches)")
