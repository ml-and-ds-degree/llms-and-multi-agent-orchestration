"""Pytest configuration and fixtures for HW7 tests."""

import sys
from pathlib import Path

import pytest

# Add HW7 to path
HW7_DIR = Path(__file__).parent.parent
if str(HW7_DIR) not in sys.path:
    sys.path.insert(0, str(HW7_DIR))


@pytest.fixture
def sample_player_meta():
    """Sample player metadata for testing."""
    from shared.schemas import PlayerMeta

    return PlayerMeta(
        display_name="Test Player",
        version="1.0.0",
        game_types=["even_odd"],
        contact_endpoint="http://localhost:9999/mcp",
    )


@pytest.fixture
def sample_referee_meta():
    """Sample referee metadata for testing."""
    from shared.schemas import RefereeMeta

    return RefereeMeta(
        display_name="Test Referee",
        version="1.0.0",
        game_types=["even_odd"],
        contact_endpoint="http://localhost:9998/mcp",
        max_concurrent_matches=5,
    )


@pytest.fixture
def sample_match_info():
    """Sample match info for testing."""
    from shared.enums import GameType
    from shared.schemas import MatchInfo

    return MatchInfo(
        match_id="TEST_M1",
        game_type=GameType.EVEN_ODD,
        player_A_id="P01",
        player_B_id="P02",
        referee_endpoint="http://localhost:8001/mcp",
    )
