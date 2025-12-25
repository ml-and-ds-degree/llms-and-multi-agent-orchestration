import asyncio
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import arrow
import httpx
from loguru import logger

# Ensure HW7 package is on sys.path
BASE_DIR = Path(__file__).resolve().parent


if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from shared.schemas import StartLeague
from shared.settings import settings

processes = []


def start_process(cmd, cwd="."):
    # Start each service in its own session so signals can be sent cleanly
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )
    processes.append(p)
    return p


def shutdown_process(p: subprocess.Popen, name: str, timeout: float = 5.0):
    if timeout is None:
        timeout = settings.shutdown_sigint_timeout
    if p.poll() is not None:
        return
    try:
        logger.info("Stopping {} (SIGINT)...", name)
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=timeout)
            logger.success("{} stopped after SIGINT", name)
            return
        except subprocess.TimeoutExpired:
            logger.warning("{} did not stop after SIGINT, sending SIGTERM...", name)
        p.terminate()
        try:
            p.wait(timeout=timeout)
            logger.success("{} stopped after SIGTERM", name)
            return
        except subprocess.TimeoutExpired:
            logger.error("{} did not stop after SIGTERM, sending SIGKILL...", name)
        p.kill()
        p.wait()
        logger.success("{} killed", name)
    except Exception as e:
        logger.error("Error stopping {}: {}", name, e)


async def main():
    logger.info(">>> Starting League Simulation <<<")

    # 1. Start League Manager
    logger.info("Starting League Manager...")
    start_process([sys.executable, "-m", "league_manager.main"], cwd=str(BASE_DIR))

    # 2. Start Referee
    logger.info("Starting Referee...")
    start_process([sys.executable, "-m", "referee.main"], cwd=str(BASE_DIR))

    # 3. Start Players
    for port, pid in zip(settings.player_ports, settings.player_ids_list):
        logger.info("Starting Player {} on port {}...", pid, port)
        start_process([sys.executable, "-m", "player.main", str(port), pid], cwd=str(BASE_DIR))

    # 4. Wait for startup
    logger.info("Waiting for agents to initialize ({}s)...", settings.startup_wait_time)
    await asyncio.sleep(settings.startup_wait_time)

    # 5. Start League
    logger.info("Sending START_LEAGUE command...")
    msg = StartLeague(
        sender="launcher",
        timestamp=arrow.utcnow().datetime,
        conversation_id=str(uuid.uuid4()),
        league_id=settings.league_id,
    )

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                settings.league_manager_url,
                json=msg.model_dump(mode="json"),
            )
            logger.success("League Started: {}", resp.json())
        except Exception as e:
            logger.error("Failed to start league: {}", e)

    # 6. Monitor Loop (Optional, just keep running)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping simulation...")
    finally:
        # Stop all child processes gracefully
        names = [
            (processes[0] if len(processes) > 0 else None, "League Manager"),
            (processes[1] if len(processes) > 1 else None, "Referee"),
        ]
        # Players
        for i in range(2, len(processes)):
            names.append((processes[i], f"Player {settings.player_ids_list[i - 2]}"))

        for p, name in names:
            if p is not None:
                shutdown_process(p, name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
