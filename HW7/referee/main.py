import asyncio
import random
import uuid
from typing import Dict, Optional

import arrow
import httpx
import uvicorn
from fastapi import FastAPI
from loguru import logger
from shared.enums import MessageType
from shared.schemas import (
    ChooseParityCall,
    ChooseParityResponse,
    GameInvitation,
    GameJoinAck,
    GameOver,
    GameResult,
    GameResultStatus,
    LeagueQuery,
    MatchInfo,
    MatchResultReport,
    MatchScore,
    ParityCallContext,
    RefereeMeta,
    RefereeRegisterRequest,
    RoundAnnouncement,
)
from shared.settings import settings

matches_queue: asyncio.Queue = asyncio.Queue()
player_endpoints: Dict[str, str] = {}  # Cache for player endpoints
league_id: Optional[str] = settings.league_id


async def get_player_endpoint(player_id: str) -> Optional[str]:
    if player_id in player_endpoints:
        return player_endpoints[player_id]

    query = LeagueQuery(
        sender=f"referee:{settings.referee_id}",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        auth_token="simulated_token",
        league_id=league_id or "unknown",
        query_type="GET_PLAYER_DIRECTORY",
    )

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                settings.league_manager_url, json=query.model_dump(mode="json")
            )
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and "players" in data["data"]:
                    player_endpoints.update(data["data"]["players"])
                    return player_endpoints.get(player_id)
        except Exception as e:
            logger.error("Error querying league manager: {}", e)
            return None
    return None


async def run_match(match: MatchInfo, round_id: int):
    logger.debug(
        "Starting Match {} ({} vs {})",
        match.match_id,
        match.player_A_id,
        match.player_B_id,
    )

    endpoint_A = await get_player_endpoint(match.player_A_id)
    endpoint_B = await get_player_endpoint(match.player_B_id)

    if not endpoint_A or not endpoint_B:
        logger.error("Could not resolve endpoints for match {}", match.match_id)
        return

    async with httpx.AsyncClient() as client:
        invitation_A = GameInvitation(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            league_id=league_id or "unknown",
            round_id=round_id,
            match_id=match.match_id,
            game_type=match.game_type,
            role_in_match="PLAYER_A",
            opponent_id=match.player_B_id,
        )
        invitation_B = GameInvitation(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            league_id=league_id or "unknown",
            round_id=round_id,
            match_id=match.match_id,
            game_type=match.game_type,
            role_in_match="PLAYER_B",
            opponent_id=match.player_A_id,
        )

        async def invite_player(url, invite_msg):
            try:
                resp = await client.post(
                    url, json=invite_msg.model_dump(mode="json"), timeout=5.0
                )
                if resp.status_code == 200:
                    return GameJoinAck(**resp.json())
            except Exception as e:
                logger.error("Invitation failed: {}", e)
            return None

        ack_A, ack_B = await asyncio.gather(
            invite_player(endpoint_A, invitation_A),
            invite_player(endpoint_B, invitation_B),
        )

        if not ack_A or not ack_A.accept or not ack_B or not ack_B.accept:
            logger.warning(
                "Match {} aborted: One or more players declined or timed out.",
                match.match_id,
            )
            return

        drawn_number = random.randint(
            settings.random_number_min, settings.random_number_max
        )
        parity = "even" if drawn_number % 2 == 0 else "odd"

        deadline = arrow.utcnow().shift(seconds=settings.parity_choice_timeout).datetime

        call_A = ChooseParityCall(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id=match.match_id,
            player_id=match.player_A_id,
            game_type=match.game_type,
            context=ParityCallContext(opponent_id=match.player_B_id, round_id=round_id),
            deadline=deadline,
        )
        call_B = ChooseParityCall(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id=match.match_id,
            player_id=match.player_B_id,
            game_type=match.game_type,
            context=ParityCallContext(opponent_id=match.player_A_id, round_id=round_id),
            deadline=deadline,
        )

        async def get_move(url, call_msg):
            try:
                resp = await client.post(
                    url, json=call_msg.model_dump(mode="json"), timeout=30.0
                )
                if resp.status_code == 200:
                    return ChooseParityResponse(**resp.json())
            except Exception as e:
                logger.error("Move request failed: {}", e)
            return None

        resp_A, resp_B = await asyncio.gather(
            get_move(endpoint_A, call_A), get_move(endpoint_B, call_B)
        )

        choice_A = resp_A.parity_choice if resp_A else None
        choice_B = resp_B.parity_choice if resp_B else None

        winner = None
        status = GameResultStatus.DRAW
        reason = ""

        is_A_correct = choice_A == parity
        is_B_correct = choice_B == parity

        if choice_A is None or choice_B is None:
            status = GameResultStatus.TECHNICAL_LOSS
            reason = "Timeout/Connection Error"
            if choice_A is None and choice_B is not None:
                winner = match.player_B_id
                status = GameResultStatus.WIN
            elif choice_B is None and choice_A is not None:
                winner = match.player_A_id
                status = GameResultStatus.WIN
        elif is_A_correct == is_B_correct:
            status = GameResultStatus.DRAW
            reason = f"Both {'correct' if is_A_correct else 'wrong'} ({drawn_number} is {parity})"
        elif is_A_correct:
            winner = match.player_A_id
            status = GameResultStatus.WIN
            reason = (
                f"P_A correct ({choice_A}), P_B wrong ({choice_B}) for {drawn_number}"
            )
        else:
            winner = match.player_B_id
            status = GameResultStatus.WIN
            reason = (
                f"P_B correct ({choice_B}), P_A wrong ({choice_A}) for {drawn_number}"
            )

        logger.info("Match {} Result: {} Winner: {}", match.match_id, status, winner)

        game_res = GameResult(
            status=status,
            winner_player_id=winner,
            drawn_number=drawn_number,
            number_parity=parity,
            choices={
                match.player_A_id: str(choice_A),
                match.player_B_id: str(choice_B),
            },
            reason=reason,
        )

        game_over = GameOver(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            match_id=match.match_id,
            game_type=match.game_type,
            game_result=game_res,
        )

        asyncio.create_task(
            client.post(endpoint_A, json=game_over.model_dump(mode="json"))
        )
        asyncio.create_task(
            client.post(endpoint_B, json=game_over.model_dump(mode="json"))
        )

        report = MatchResultReport(
            sender=f"referee:{settings.referee_id}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=str(uuid.uuid4()),
            league_id=league_id or "unknown",
            round_id=round_id,
            match_id=match.match_id,
            game_type=match.game_type,
            result=MatchScore(
                winner=winner or "",
                score={match.player_A_id: 0, match.player_B_id: 0},
                details={
                    "drawn_number": drawn_number,
                    "choices": {
                        match.player_A_id: choice_A,
                        match.player_B_id: choice_B,
                    },
                },
            ),
        )
        if winner == match.player_A_id:
            report.result.score[match.player_A_id] = 3
            report.result.score[match.player_B_id] = 0
        elif winner == match.player_B_id:
            report.result.score[match.player_A_id] = 0
            report.result.score[match.player_B_id] = 3
        else:
            report.result.score[match.player_A_id] = 1
            report.result.score[match.player_B_id] = 1

        await client.post(
            settings.league_manager_url, json=report.model_dump(mode="json")
        )


async def process_queue():
    while True:
        task = await matches_queue.get()
        match_info, round_id = task
        asyncio.create_task(run_match(match_info, round_id))
        matches_queue.task_done()


async def register_referee():
    req = RefereeRegisterRequest(
        sender=f"referee:{settings.referee_id}",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        referee_meta=RefereeMeta(
            display_name=settings.referee_display_name,
            version=settings.referee_version,
            game_types=[settings.game_type],
            contact_endpoint=settings.referee_contact_endpoint,
            max_concurrent_matches=settings.referee_max_concurrent_matches,
        ),
    )

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                settings.league_manager_url, json=req.model_dump(mode="json")
            )
            logger.success("Referee Registered")
        except Exception as e:
            logger.error("Referee Registration Failed: {}", e)


async def lifespan(app: FastAPI):
    asyncio.create_task(process_queue())
    await asyncio.sleep(2)
    await register_referee()
    yield


app = FastAPI(title="Referee", lifespan=lifespan)


@app.post("/mcp")
async def mcp_endpoint(payload: dict):
    msg_type = payload.get("message_type")

    if msg_type == MessageType.ROUND_ANNOUNCEMENT:
        announcement = RoundAnnouncement(**payload)
        global league_id
        league_id = announcement.league_id

        for m in announcement.matches:
            if m.referee_endpoint == settings.referee_contact_endpoint:
                await matches_queue.put((m, announcement.round_id))

        return {"status": "received"}

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.referee_host, port=settings.referee_port)
