import asyncio
import random
import sys
import uuid
from contextlib import asynccontextmanager

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
    LeagueRegisterRequest,
    PlayerMeta,
)
from shared.settings import settings

# Get port and ID from args
# Usage: python main.py <PORT> <PLAYER_ID>
if len(sys.argv) < 3:
    logger.error("Usage: python main.py <PORT> <PLAYER_ID>")
    sys.exit(1)

PORT = int(sys.argv[1])
PLAYER_ID = sys.argv[2]  # e.g. P01
ENDPOINT = f"http://localhost:{PORT}/mcp"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wait for LM
    await asyncio.sleep(3)

    req = LeagueRegisterRequest(
        sender=f"player:{PLAYER_ID}",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        player_meta=PlayerMeta(
            display_name=f"Agent {PLAYER_ID}",
            version=settings.player_version,
            game_types=[settings.game_type],
            contact_endpoint=ENDPOINT,
        ),
    )

    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                settings.league_manager_url, json=req.model_dump(mode="json")
            )
            logger.success("Player {} Registered", PLAYER_ID)
        except Exception as e:
            logger.error("Player {} Registration Failed: {}", PLAYER_ID, e)

    yield


app = FastAPI(title=f"Player {PLAYER_ID}", lifespan=lifespan)


@app.post("/mcp")
async def mcp_endpoint(payload: dict):
    msg_type = payload.get("message_type")

    if msg_type == MessageType.GAME_INVITATION:
        inv = GameInvitation(**payload)
        return GameJoinAck(
            sender=f"player:{PLAYER_ID}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=inv.conversation_id,
            match_id=inv.match_id,
            player_id=PLAYER_ID,  # Note: LM assigns ID, but for sim we pre-assign/assume consistency
            arrival_timestamp=arrow.utcnow().datetime,
            accept=True,
        )

    elif msg_type == MessageType.CHOOSE_PARITY_CALL:
        call = ChooseParityCall(**payload)

        # Strategy: Random
        choice = random.choice(["even", "odd"])

        return ChooseParityResponse(
            sender=f"player:{PLAYER_ID}",
            timestamp=arrow.utcnow().datetime,
            conversation_id=call.conversation_id,  # Should this be new or related? Using same for thread.
            match_id=call.match_id,
            player_id=PLAYER_ID,
            parity_choice=choice,
        )

    elif msg_type == MessageType.GAME_OVER:
        # Just ack or log
        # print(f"Player {PLAYER_ID} received GAME_OVER: {payload}")
        return {"status": "ok"}

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.player_host, port=PORT)
