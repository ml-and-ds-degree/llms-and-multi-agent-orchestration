import asyncio
import random
import uuid
from typing import Dict, List, Optional

import arrow
import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from loguru import logger
from shared.enums import GameType, MessageType
from shared.schemas import (
    ChampionInfo,
    LeagueCompleted,
    LeagueRegisterRequest,
    LeagueRegisterResponse,
    LeagueStandingsUpdate,
    LeagueStatus,
    MatchInfo,
    MatchResultAck,
    MatchResultReport,
    RefereeRegisterRequest,
    RefereeRegisterResponse,
    RoundAnnouncement,
    RoundCompleted,
    RoundSummary,
    StandingEntry,
    StartLeague,
)
from shared.settings import settings

app = FastAPI(title="League Manager")

# --- State ---
players: Dict[str, LeagueRegisterRequest] = {}
referee: Optional[RefereeRegisterRequest] = None
referee_id: Optional[str] = None
matches_schedule: List[List[MatchInfo]] = []  # List of rounds, each round has matches
match_results: Dict[str, MatchResultReport] = {}
current_round_index = 0
league_status = "pending"  # pending, running, completed
standings: Dict[str, StandingEntry] = {}

# --- Logic ---


def generate_schedule(player_ids: List[str]):
    """Generates a Round Robin schedule."""
    if len(player_ids) % 2 != 0:
        player_ids.append("BYE")

    n = len(player_ids)
    rounds = []
    for r in range(n - 1):
        round_matches = []
        for i in range(n // 2):
            p1 = player_ids[i]
            p2 = player_ids[n - 1 - i]
            if p1 != "BYE" and p2 != "BYE":
                # Randomize roles
                if random.choice([True, False]):
                    pA, pB = p1, p2
                else:
                    pA, pB = p2, p1

                round_matches.append(
                    MatchInfo(
                        match_id=f"R{r + 1}M{i + 1}",
                        game_type=GameType.EVEN_ODD,
                        player_A_id=pA,
                        player_B_id=pB,
                        referee_endpoint=referee.referee_meta.contact_endpoint
                        if referee
                        else "http://localhost:8001/mcp",
                    )
                )
        rounds.append(round_matches)

        # Rotate list

        player_ids = [player_ids[0]] + [player_ids[-1]] + player_ids[1:-1]

    return rounds


async def run_round():
    global current_round_index
    if current_round_index >= len(matches_schedule):
        await finish_league()
        return

    round_matches = matches_schedule[current_round_index]

    # Broadcast Round Announcement
    announcement = RoundAnnouncement(
        message_type=MessageType.ROUND_ANNOUNCEMENT,
        sender="league_manager",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        league_id=settings.league_id,
        round_id=current_round_index + 1,
        matches=round_matches,
    )

    logger.debug("Starting Round {}", current_round_index + 1)

    # Send to Referee
    async with httpx.AsyncClient() as client:
        try:
            endpoint = (
                referee.referee_meta.contact_endpoint
                if referee
                else "http://localhost:8001/mcp"
            )
            await client.post(endpoint, json=announcement.model_dump(mode="json"))
        except Exception as e:
            logger.error("Error sending round announcement to referee: {}", e)


async def process_match_result(report: MatchResultReport):
    match_results[report.match_id] = report

    # Update standings
    winner_id = report.result.winner
    scores = report.result.score

    for pid in scores:
        if pid not in standings:
            # Should not happen if registered correctly
            continue

        standings[pid].played = (standings[pid].played or 0) + 1

        if pid == winner_id:
            standings[pid].wins = (standings[pid].wins or 0) + 1
            standings[pid].points += 3  # 3 points for win
        elif winner_id:  # There was a winner, so this is a loser
            standings[pid].losses = (standings[pid].losses or 0) + 1
        else:  # Draw
            standings[pid].draws = (standings[pid].draws or 0) + 1
            standings[pid].points += 1  # 1 point for draw

    # Check if round is complete
    current_round_matches = matches_schedule[current_round_index]
    completed_in_round = [
        m.match_id for m in current_round_matches if m.match_id in match_results
    ]

    if len(completed_in_round) == len(current_round_matches):
        await finish_round()


async def finish_round():
    global current_round_index

    # Calculate summary
    round_matches = matches_schedule[current_round_index]
    wins = 0
    draws = 0
    tech_loss = 0

    for m in round_matches:
        res = match_results[m.match_id]
        # In a real impl, we'd parse the 'status' from the GameResult, but here we infer from report
        # The schema for MatchResultReport doesn't have the status enum directly nested,
        # it has 'winner' and 'score'.
        # Simulating summary based on winners:
        if res.result.winner:
            wins += 1
        else:
            draws += 1

    summary = RoundSummary(
        total_matches=len(round_matches),
        wins=wins,
        draws=draws,
        technical_losses=tech_loss,
    )

    completed_msg = RoundCompleted(
        sender="league_manager",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        league_id=settings.league_id,
        round_id=current_round_index + 1,
        matches_completed=len(round_matches),
        next_round_id=current_round_index + 2
        if current_round_index + 1 < len(matches_schedule)
        else None,
        summary=summary,
    )

    # Broadcast Round Completed & Standings to Players
    # (In a real system we'd loop through all players. For sim we can just log or send if players had listener endpoints)
    logger.debug("Round {} Completed. Summary: {}", current_round_index + 1, summary)

    # Broadcast Standings
    sorted_standings = sorted(standings.values(), key=lambda x: x.points, reverse=True)
    for i, s in enumerate(sorted_standings):
        s.rank = i + 1

    standings_update = LeagueStandingsUpdate(
        sender="league_manager",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        league_id=settings.league_id,
        round_id=current_round_index + 1,
        standings=sorted_standings,
    )

    # Advance Round
    current_round_index += 1

    # Trigger next round or finish
    asyncio.create_task(run_round())


async def finish_league():
    global league_status
    league_status = "completed"

    sorted_standings = sorted(standings.values(), key=lambda x: x.points, reverse=True)
    champion_entry = sorted_standings[0]

    champion = ChampionInfo(
        player_id=champion_entry.player_id,
        display_name=champion_entry.display_name,
        points=champion_entry.points,
    )

    msg = LeagueCompleted(
        sender="league_manager",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        league_id=settings.league_id,
        total_rounds=len(matches_schedule),
        total_matches=sum(len(r) for r in matches_schedule),
        champion=champion,
        final_standings=sorted_standings,
    )

    logger.success("LEAGUE COMPLETED! Champion: {}", champion.display_name)


# --- Endpoints ---


@app.post("/mcp")
async def mcp_endpoint(payload: dict, background_tasks: BackgroundTasks):
    msg_type = payload.get("message_type")

    if msg_type == MessageType.LEAGUE_REGISTER_REQUEST:
        req = LeagueRegisterRequest(**payload)
        pid = f"P{len(players) + 1:02d}"  # P01, P02...
        players[pid] = req

        # Init standing
        standings[pid] = StandingEntry(
            rank=0,
            player_id=pid,
            display_name=req.player_meta.display_name,
            played=0,
            wins=0,
            draws=0,
            losses=0,
            points=0,
        )

        return LeagueRegisterResponse(
            sender="league_manager",
            timestamp=arrow.utcnow().datetime,
            conversation_id=req.conversation_id,
            status="ACCEPTED",
            player_id=pid,
        )

    elif msg_type == MessageType.REFEREE_REGISTER_REQUEST:
        req = RefereeRegisterRequest(**payload)
        global referee, referee_id
        referee = req
        referee_id = "REF01"
        return RefereeRegisterResponse(
            sender="league_manager",
            timestamp=arrow.utcnow().datetime,
            conversation_id=req.conversation_id,
            status="ACCEPTED",
            referee_id=referee_id,
        )

    elif msg_type == MessageType.START_LEAGUE:
        req = StartLeague(**payload)
        global matches_schedule, league_status
        matches_schedule = generate_schedule(list(players.keys()))
        league_status = "running"

        background_tasks.add_task(run_round)

        return LeagueStatus(
            sender="league_manager",
            timestamp=arrow.utcnow().datetime,
            conversation_id=req.conversation_id,
            league_id=settings.league_id,
            status=league_status,
            current_round=current_round_index + 1,
            total_rounds=len(matches_schedule),
            matches_completed=0,
        )

    elif msg_type == MessageType.MATCH_RESULT_REPORT:
        req = MatchResultReport(**payload)
        background_tasks.add_task(process_match_result, req)
        return MatchResultAck(
            sender="league_manager",
            timestamp=arrow.utcnow().datetime,
            conversation_id=req.conversation_id,
            match_id=req.match_id,
            status="recorded",
        )

    elif msg_type == MessageType.LEAGUE_QUERY:
        # Simplified query handler for LEAGUE_QUERY just to return player endpoints
        # Real impl would check auth tokens etc.
        # Referee needs to find player endpoints

        # Assume query is "GET_PLAYER_ENDPOINT" or similar implicit in contracts
        # Contracts say "GET_NEXT_MATCH" is an example.
        # For this simulation, Referee needs endpoints.

        # We'll just return all player metadata if query_type is generic or custom
        # But strictly following the schema, let's look at payload.

        # For the purpose of this assignment, the Referee needs to know how to contact P01, P02.
        # The RoundAnnouncement gave the IDs, but NOT the endpoints of the players.
        # The Referee needs to query the League Manager to get the player endpoint.

        # Hack for simulation: Return a map of all players if query_type is "GET_PLAYER_DIRECTORY"

        return {
            "protocol": "league.v2",
            "message_type": MessageType.LEAGUE_QUERY_RESPONSE,
            "sender": "league_manager",
            "timestamp": arrow.utcnow().isoformat(),
            "conversation_id": payload.get("conversation_id"),
            "query_type": payload.get("query_type"),
            "success": True,
            "data": {
                "players": {
                    pid: p.player_meta.contact_endpoint for pid, p in players.items()
                }
            },
        }

    return {"error": "Unknown message type"}


if __name__ == "__main__":
    uvicorn.run(
        app, host=settings.league_manager_host, port=settings.league_manager_port
    )
