from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .enums import GameResultStatus, GameType, MessageType, ProtocolVersion

# --- Base Models ---


class BaseMessage(BaseModel):
    protocol: ProtocolVersion = ProtocolVersion.V2
    message_type: MessageType
    sender: str
    timestamp: datetime
    conversation_id: str


# --- 1. LEAGUE_MANAGER SOURCE ---


class MatchInfo(BaseModel):
    match_id: str
    game_type: GameType
    player_A_id: str
    player_B_id: str
    referee_endpoint: str


class RoundAnnouncement(BaseMessage):
    message_type: MessageType = MessageType.ROUND_ANNOUNCEMENT
    league_id: str
    round_id: int
    matches: List[MatchInfo]


class RoundSummary(BaseModel):
    total_matches: int
    wins: int
    draws: int
    technical_losses: int


class RoundCompleted(BaseMessage):
    message_type: MessageType = MessageType.ROUND_COMPLETED
    league_id: str
    round_id: int
    matches_completed: int
    next_round_id: Optional[int] = None
    summary: RoundSummary


class StandingEntry(BaseModel):
    rank: int
    player_id: str
    display_name: Optional[str] = None
    played: Optional[int] = None
    wins: Optional[int] = None
    draws: Optional[int] = None
    losses: Optional[int] = None
    points: int


class LeagueStandingsUpdate(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_STANDINGS_UPDATE
    league_id: str
    round_id: int
    standings: List[StandingEntry]


class ChampionInfo(BaseModel):
    player_id: str
    display_name: str
    points: int


class LeagueCompleted(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_COMPLETED
    league_id: str
    total_rounds: int
    total_matches: int
    champion: ChampionInfo
    final_standings: List[StandingEntry]


class LeagueQueryResponse(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_QUERY_RESPONSE
    query_type: str
    success: bool
    data: Optional[Dict[str, Any]] = None


class LeagueError(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_ERROR
    error_code: str
    error_description: str
    original_message_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# --- 2. REFEREE SOURCE ---


class RefereeMeta(BaseModel):
    display_name: str
    version: str
    game_types: List[str]
    contact_endpoint: str
    max_concurrent_matches: int


class RefereeRegisterRequest(BaseMessage):
    message_type: MessageType = MessageType.REFEREE_REGISTER_REQUEST
    referee_meta: RefereeMeta


class RefereeRegisterResponse(BaseMessage):
    message_type: MessageType = MessageType.REFEREE_REGISTER_RESPONSE
    status: str
    referee_id: str
    reason: Optional[str] = None


class GameInvitation(BaseMessage):
    message_type: MessageType = MessageType.GAME_INVITATION
    league_id: str
    round_id: int
    match_id: str
    game_type: GameType
    role_in_match: str
    opponent_id: str


class ParityCallContext(BaseModel):
    opponent_id: str
    round_id: int
    your_standings: Optional[Dict[str, int]] = None


class ChooseParityCall(BaseMessage):
    message_type: MessageType = MessageType.CHOOSE_PARITY_CALL
    match_id: str
    player_id: str
    game_type: GameType
    context: ParityCallContext
    deadline: datetime


class GameResult(BaseModel):
    status: GameResultStatus
    winner_player_id: Optional[str] = None
    drawn_number: Optional[int] = None
    number_parity: Optional[str] = None
    choices: Optional[Dict[str, str]] = None
    reason: Optional[str] = None


class GameOver(BaseMessage):
    message_type: MessageType = MessageType.GAME_OVER
    match_id: str
    game_type: GameType
    game_result: GameResult


class MatchScore(BaseModel):
    winner: str
    score: Dict[str, int]
    details: Dict[str, Any]


class MatchResultReport(BaseMessage):
    message_type: MessageType = MessageType.MATCH_RESULT_REPORT
    league_id: str
    round_id: int
    match_id: str
    game_type: GameType
    result: MatchScore


class MatchResultAck(BaseMessage):
    message_type: MessageType = MessageType.MATCH_RESULT_ACK
    match_id: str
    status: str


class GameError(BaseMessage):
    message_type: MessageType = MessageType.GAME_ERROR
    match_id: str
    error_code: str
    error_description: str
    affected_player: str
    action_required: str
    retry_info: Optional[Dict[str, Any]] = None
    consequence: Optional[str] = None


# --- 3. PLAYER SOURCE ---


class PlayerMeta(BaseModel):
    display_name: str
    version: str
    game_types: List[str]
    contact_endpoint: str


class LeagueRegisterRequest(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_REGISTER_REQUEST
    player_meta: PlayerMeta


class LeagueRegisterResponse(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_REGISTER_RESPONSE
    status: str
    player_id: str
    reason: Optional[str] = None


class GameJoinAck(BaseMessage):
    message_type: MessageType = MessageType.GAME_JOIN_ACK
    match_id: str
    player_id: str
    arrival_timestamp: datetime
    accept: bool


class ChooseParityResponse(BaseMessage):
    message_type: MessageType = MessageType.CHOOSE_PARITY_RESPONSE
    match_id: str
    player_id: str
    parity_choice: str


class LeagueQuery(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_QUERY
    auth_token: str
    league_id: str
    query_type: str
    query_params: Optional[Dict[str, Any]] = None


# --- 4. LAUNCHER SOURCE ---


class StartLeague(BaseMessage):
    message_type: MessageType = MessageType.START_LEAGUE
    league_id: str


class LeagueStatus(BaseMessage):
    message_type: MessageType = MessageType.LEAGUE_STATUS
    league_id: str
    status: str
    current_round: int
    total_rounds: int
    matches_completed: int
