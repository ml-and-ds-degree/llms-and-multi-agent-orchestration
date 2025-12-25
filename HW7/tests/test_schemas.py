"""Tests for message schemas and validation."""

import uuid

import arrow
import pytest
from pydantic import ValidationError
from shared.enums import GameType, MessageType
from shared.schemas import (
    ChooseParityCall,
    ChooseParityResponse,
    GameInvitation,
    GameJoinAck,
    LeagueRegisterRequest,
    ParityCallContext,
    PlayerMeta,
    RefereeRegisterRequest,
    StartLeague,
)


class TestPlayerMeta:
    """Test PlayerMeta schema validation."""

    def test_valid_player_meta(self, sample_player_meta):
        """Test valid player metadata."""
        assert sample_player_meta.display_name == "Test Player"
        assert sample_player_meta.version == "1.0.0"
        assert "even_odd" in sample_player_meta.game_types

    def test_player_meta_missing_fields(self):
        """Test player metadata with missing required fields."""
        with pytest.raises(ValidationError):
            PlayerMeta(display_name="Test")  # Missing other fields


class TestMessageSerialization:
    """Test message serialization to JSON."""

    def test_league_register_request_serialization(self, sample_player_meta):
        """Test LeagueRegisterRequest serialization."""
        msg = LeagueRegisterRequest(
            sender="test_player",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            player_meta=sample_player_meta,
        )

        # Should serialize without errors
        data = msg.model_dump(mode="json")

        assert data["message_type"] == "LEAGUE_REGISTER_REQUEST"
        assert data["sender"] == "test_player"
        assert "player_meta" in data

    def test_start_league_serialization(self):
        """Test StartLeague message serialization."""
        msg = StartLeague(
            sender="launcher",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            league_id="test_league",
        )

        data = msg.model_dump(mode="json")

        assert data["message_type"] == "START_LEAGUE"
        assert data["league_id"] == "test_league"
        assert isinstance(data["timestamp"], str)  # Should be serialized to string


class TestMatchInfo:
    """Test MatchInfo schema."""

    def test_valid_match_info(self, sample_match_info):
        """Test valid match info creation."""
        assert sample_match_info.match_id == "TEST_M1"
        assert sample_match_info.player_A_id == "P01"
        assert sample_match_info.player_B_id == "P02"

    def test_match_info_game_type(self, sample_match_info):
        """Test match info game type enum."""
        assert sample_match_info.game_type == GameType.EVEN_ODD


class TestGameMessages:
    """Test game-related message schemas."""

    def test_game_invitation(self):
        """Test GameInvitation message creation."""
        inv = GameInvitation(
            sender="referee:REF01",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            league_id="test_league",
            round_id=1,
            match_id="M1",
            game_type=GameType.EVEN_ODD,
            role_in_match="PLAYER_A",
            opponent_id="P02",
        )

        assert inv.role_in_match == "PLAYER_A"
        assert inv.opponent_id == "P02"

    def test_game_join_ack(self):
        """Test GameJoinAck message creation."""
        ack = GameJoinAck(
            sender="player:P01",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id="M1",
            player_id="P01",
            arrival_timestamp=arrow.utcnow().datetime,
            accept=True,
        )

        assert ack.accept is True
        assert ack.player_id == "P01"

    def test_choose_parity_call(self):
        """Test ChooseParityCall message creation."""
        call = ChooseParityCall(
            sender="referee:REF01",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id="M1",
            player_id="P01",
            game_type=GameType.EVEN_ODD,
            context=ParityCallContext(opponent_id="P02", round_id=1),
            deadline=arrow.utcnow().shift(seconds=30).datetime,
        )

        assert call.player_id == "P01"
        assert call.context.opponent_id == "P02"

    def test_choose_parity_response(self):
        """Test ChooseParityResponse message creation."""
        resp = ChooseParityResponse(
            sender="player:P01",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id="M1",
            player_id="P01",
            parity_choice="even",
        )

        assert resp.parity_choice == "even"


class TestRefereeMessages:
    """Test referee-related message schemas."""

    def test_referee_register_request(self, sample_referee_meta):
        """Test RefereeRegisterRequest creation."""
        req = RefereeRegisterRequest(
            sender="referee:REF01",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            referee_meta=sample_referee_meta,
        )

        assert req.referee_meta.max_concurrent_matches == 5
        assert req.message_type == MessageType.REFEREE_REGISTER_REQUEST
