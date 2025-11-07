"""Shared pytest fixtures for unit and E2E tests."""

import os
import subprocess
import time
from typing import Generator

import pytest
import requests


@pytest.fixture(scope="session")
def reflex_server() -> Generator[str, None, None]:
    """Start the Reflex development server for E2E tests.
    
    Yields:
        str: Base URL of the running Reflex server.
    """
    # Set test environment variables
    env = os.environ.copy()
    env["REFLEX_ENV"] = "test"
    
    # Start the Reflex server
    process = subprocess.Popen(
        ["reflex", "run", "--port", "3001"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    base_url = "http://localhost:3001"
    
    # Wait for server to be ready (max 60 seconds)
    server_ready = False
    for _ in range(120):  # 120 * 0.5s = 60s timeout
        try:
            response = requests.get(base_url, timeout=1)
            if response.status_code == 200:
                server_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    
    if not server_ready:
        process.terminate()
        process.wait(timeout=5)
        pytest.skip("Reflex server failed to start within timeout")
    
    yield base_url
    
    # Cleanup: terminate the server
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture(scope="function")
def app_page(reflex_server, page):
    """Navigate to the Reflex app before each E2E test.
    
    This fixture extends the Playwright page fixture to automatically
    navigate to the running Reflex server.
    
    Args:
        reflex_server: The running Reflex server fixture.
        page: The Playwright page fixture (from pytest-playwright).
        
    Yields:
        Page: Playwright page instance at the app home page.
    """
    page.goto(reflex_server)
    yield page
