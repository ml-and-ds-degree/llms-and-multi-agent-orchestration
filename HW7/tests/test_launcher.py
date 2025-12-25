"""Tests for launcher process management."""

import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch


class TestLauncherProcessManagement:
    """Test launcher's process management logic."""

    def test_launcher_imports(self):
        """Test that launcher can be imported."""
        # Add parent to path
        launcher_dir = Path(__file__).parent.parent.parent
        if str(launcher_dir) not in sys.path:
            sys.path.insert(0, str(launcher_dir))

        from HW7 import launcher

        assert hasattr(launcher, "start_process")
        assert hasattr(launcher, "shutdown_process")

    @patch("subprocess.Popen")
    def test_start_process(self, mock_popen):
        """Test process startup."""
        from HW7 import launcher

        mock_process = Mock()
        mock_popen.return_value = mock_process

        # Reset processes list
        launcher.processes = []

        process = launcher.start_process(["echo", "test"], cwd=".")

        assert process == mock_process
        assert len(launcher.processes) == 1
        mock_popen.assert_called_once()

    def test_shutdown_process_already_exited(self):
        """Test shutdown when process already exited."""
        from HW7 import launcher

        mock_process = Mock()
        mock_process.poll.return_value = 0  # Already exited

        # Should return immediately without sending signals
        launcher.shutdown_process(mock_process, "test")

        mock_process.send_signal.assert_not_called()

    def test_shutdown_process_sigint_success(self):
        """Test shutdown with successful SIGINT."""
        from HW7 import launcher

        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.return_value = None  # Exits after SIGINT

        launcher.shutdown_process(mock_process, "test", timeout=0.1)

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.wait.assert_called()

    def test_shutdown_process_needs_sigterm(self):
        """Test shutdown escalation to SIGTERM."""
        from HW7 import launcher

        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running

        # First wait times out, second succeeds
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 0.1),
            None,
        ]

        launcher.shutdown_process(mock_process, "test", timeout=0.1)

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.terminate.assert_called_once()

    def test_shutdown_process_needs_sigkill(self):
        """Test shutdown escalation to SIGKILL."""
        from HW7 import launcher

        mock_process = Mock()
        mock_process.poll.return_value = None  # Still running

        # Both waits timeout, needs SIGKILL
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 0.1),
            subprocess.TimeoutExpired("cmd", 0.1),
            None,
        ]

        launcher.shutdown_process(mock_process, "test", timeout=0.1)

        mock_process.send_signal.assert_called_once_with(signal.SIGINT)
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()


class TestLauncherConfiguration:
    """Test launcher configuration."""

    def test_launcher_settings(self):
        """Test that launcher uses settings correctly."""
        from shared.settings import settings

        assert settings.league_manager_port == 8000
        assert settings.referee_port == 8001
        assert len(settings.player_ports) == 4
        assert settings.player_ports == [8101, 8102, 8103, 8104]

    def test_launcher_player_ids(self):
        """Test that launcher has correct player IDs."""
        from shared.settings import settings

        assert len(settings.player_ids_list) == 4
        assert settings.player_ids_list == ["P01", "P02", "P03", "P04"]
