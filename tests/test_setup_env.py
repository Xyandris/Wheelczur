"""Testy setupu niewymagające instalacji pakietów ani dostępu do sieci."""

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import setup_env


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "projekt ze spacjami"
        self.root.mkdir()
        models = self.root / "canine_limb" / "models"
        models.mkdir(parents=True)
        for name in setup_env.MODELS:
            (models / name).touch()
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        patch.object(setup_env, "ROOT", self.root).start()
        patch("sys.stdout", self.output).start()
        patch("sys.stderr", self.output).start()

    def test_unsupported_python_fails_before_creating_environment(self):
        with patch.object(setup_env.sys, "version_info", (3, 14, 0)):
            self.assertEqual(setup_env.main([]), 1)
        self.assertFalse((self.root / ".venv").exists())

    def test_missing_models_fail_before_installation(self):
        (self.root / "canine_limb" / "models" / setup_env.MODELS[0]).unlink()
        with patch.object(setup_env, "prepare_environment") as prepare:
            self.assertEqual(setup_env.main(["--device", "cpu"]), 1)
        prepare.assert_not_called()
        self.assertIn("Brak modeli", self.output.getvalue())

    def test_nonempty_directory_is_preserved(self):
        directory = self.root / ".venv"
        directory.mkdir()
        user_file = directory / "dane.txt"
        user_file.write_text("zachowaj", encoding="utf-8")
        self.assertEqual(setup_env.main([]), 1)
        self.assertEqual(user_file.read_text(encoding="utf-8"), "zachowaj")

    def test_existing_venv_is_reused_without_recreating_it(self):
        directory = self.root / ".venv"
        directory.mkdir()
        (directory / "pyvenv.cfg").touch()
        executable = directory / ("Scripts/python.exe" if setup_env.os.name == "nt" else "bin/python")
        executable.parent.mkdir()
        executable.touch()
        with patch.object(setup_env.venv, "EnvBuilder") as builder, patch.object(setup_env, "run"):
            self.assertEqual(setup_env.prepare_environment(directory), executable)
        builder.assert_not_called()

    def test_cpu_setup_uses_venv_python_and_handles_spaces(self):
        python = self.root / ".venv cpu" / "bin" / "python"
        with patch.object(setup_env, "prepare_environment", return_value=python) as prepare:
            with patch.object(setup_env.subprocess, "run") as process:
                self.assertEqual(setup_env.main(["--device", "cpu", "--dev", "--venv", ".venv cpu"]), 0)
        prepare.assert_called_once_with((self.root / ".venv cpu").resolve())
        commands = [call.args[0] for call in process.call_args_list]
        self.assertTrue(all(command[0] == str(python) for command in commands))
        self.assertIn([str(python), "-m", "pip", "install", "-e", f"{self.root}[cpu,test]"], commands)
        self.assertTrue(all(call.kwargs == {"cwd": self.root, "check": True}
                            for call in process.call_args_list))
        if setup_env.os.name != "nt":
            self.assertIn("--device cpu", self.output.getvalue())

    def test_failed_installation_returns_error_without_success_message(self):
        with patch.object(setup_env, "prepare_environment", return_value=self.root / "python"):
            with patch.object(setup_env, "run", side_effect=subprocess.CalledProcessError(1, ["pip"])):
                self.assertEqual(setup_env.main([]), 1)
        self.assertNotIn("Gotowe.", self.output.getvalue())
        self.assertIn("Setup nie powiódł się", self.output.getvalue())

    def test_default_cpu_and_explicit_cuda_setup(self):
        python = self.root / "python"
        for arguments, device in (([], "cpu"), (["--device", "cuda"], "cuda")):
            with self.subTest(device=device):
                self.output.seek(0)
                self.output.truncate()
                with patch.object(setup_env, "prepare_environment", return_value=python):
                    with patch.object(setup_env.subprocess, "run") as process:
                        self.assertEqual(setup_env.main(arguments), 0)
                commands = [call.args[0] for call in process.call_args_list]
                self.assertIn([str(python), "-m", "pip", "install", "-e", f"{self.root}[{device}]"], commands)
                if device == "cuda":
                    self.assertIn("--check-cuda", self.output.getvalue())
                    if setup_env.os.name != "nt":
                        self.assertIn("--check-cuda --device cuda:0", self.output.getvalue())
                else:
                    self.assertNotIn("--check-cuda", self.output.getvalue())

    def test_powershell_command_handles_spaces_and_apostrophes(self):
        with patch.object(setup_env.os, "name", "nt"):
            command = setup_env.command_text(["C:/Jan O'Brien/python.exe", "film psa.mp4"])
        self.assertEqual(command, "& 'C:/Jan O''Brien/python.exe' 'film psa.mp4'")


if __name__ == "__main__":
    unittest.main()
