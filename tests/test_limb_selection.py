import pytest

import launcher
from canine_limb.cli import parser, prompt_limb


def test_cli_rejects_all_limbs(capsys):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["walk.mp4", "--limb", "all"])
    assert exc.value.code == 2
    assert "invalid choice: 'all'" in capsys.readouterr().err


@pytest.mark.parametrize("answer", ["4", "right_hind"])
def test_cli_prompt_rejects_removed_choices(monkeypatch, capsys, answer):
    answers = iter(["all", "5", answer])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert prompt_limb() == "right-hind"
    output = capsys.readouterr().out
    assert "4. right-hind" in output and "5." not in output
    assert output.count("Choose 1-4") == 2


@pytest.mark.parametrize("answer", ["4", "right_hind"])
def test_launcher_prompt_rejects_removed_choices(tmp_path, monkeypatch, capsys, answer):
    video = tmp_path / "walk with spaces.mp4"
    video.touch()
    answers = iter([f'"{video}"', "all", "5", answer])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert launcher.prompt_tracking() == (video, "right-hind")
    output = capsys.readouterr().out
    assert "4. right-hind" in output and "5." not in output
    assert output.count("Choose 1–4") == 2
