"""Tests for FastAPI endpoints."""

import uuid

import arrow
import pytest
from fastapi.testclient import TestClient
from league_manager.main import app as league_manager_app
from shared.schemas import (
    LeagueRegisterRequest,
    RefereeRegisterRequest,
)


class TestLeagueManagerEndpoints:
    """Test League Manager API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for League Manager."""
        return TestClient(league_manager_app)

    def test_player_registration(self, client, sample_player_meta):
        """Test player registration endpoint."""
        request_data = LeagueRegisterRequest(
            sender="player:test",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            player_meta=sample_player_meta,
        )

        response = client.post("/mcp", json=request_data.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["message_type"] == "LEAGUE_REGISTER_RESPONSE"
        assert data["status"] == "ACCEPTED"
        assert "player_id" in data

    def test_referee_registration(self, client, sample_referee_meta):
        """Test referee registration endpoint."""
        request_data = RefereeRegisterRequest(
            sender="referee:test",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            referee_meta=sample_referee_meta,
        )

        response = client.post("/mcp", json=request_data.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["message_type"] == "REFEREE_REGISTER_RESPONSE"
        assert data["status"] == "ACCEPTED"
        assert "referee_id" in data

    def test_unknown_message_type(self, client):
        """Test handling of unknown message type."""
        response = client.post(
            "/mcp",
            json={
                "protocol": "league.v2",
                "message_type": "UNKNOWN_MESSAGE",
                "sender": "test",
                "timestamp": arrow.utcnow().isoformat(),
                "conversation_id": str(uuid.uuid4()),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data


class TestPlayerEndpoints:
    """Test Player API endpoints."""

    def test_player_module_exists(self):
        """Test that player module exists."""
        from pathlib import Path

        player_path = Path(__file__).parent.parent / "player" / "main.py"
        assert player_path.exists(), "Player module should exist"


class TestRefereeEndpoints:
    """Test Referee API endpoints."""

    def test_referee_module_exists(self):
        """Test that referee module exists."""
        from pathlib import Path

        referee_path = Path(__file__).parent.parent / "referee" / "main.py"
        assert referee_path.exists(), "Referee module should exist"
