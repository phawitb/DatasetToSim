import json

from dataset_to_sim.cli import main
from tests.test_dataset import write_dataset


def test_inspect_command_reports_dataset_contract(tmp_path, capsys):
    write_dataset(tmp_path)

    assert main(["inspect", "--dataset", str(tmp_path)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["episode_count"] == 1
    assert result["fps"] == 30.0
