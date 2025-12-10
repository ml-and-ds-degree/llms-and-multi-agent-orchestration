# Testing Guide

This guide covers the testing setup and best practices for the LLMs and Multi-Agent Orchestration project.

## Table of Contents

- [Quick Start](#quick-start)
- [Testing Stack](#testing-stack)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Writing Tests](#writing-tests)
- [Best Practices](#best-practices)

## Quick Start

### 1. Install Testing Dependencies

```bash
# Install dev dependencies including pytest and Playwright
uv sync --extra dev

# Install Playwright browsers (for E2E tests)
uv run playwright install chromium
```

### 2. Run All Tests

```bash
# Run all tests
uv run pytest

# Run only unit tests (fast)
uv run pytest -m unit

# Run only E2E tests
uv run pytest -m e2e

# Run with verbose output
uv run pytest -v
```

## Testing Stack

- **pytest** - Testing framework
- **pytest-asyncio** - Support for async tests
- **pytest-mock** - Mocking utilities
- **pytest-playwright** - Browser automation for E2E tests
- **Playwright** - Browser testing library

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test Types

```bash
# Unit tests only (fast)
uv run pytest -m unit

# E2E tests only
uv run pytest -m e2e

# Exclude slow tests
uv run pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Run state tests
uv run pytest tests/test_state.py

# Run E2E tests
uv run pytest tests/e2e/test_chat_e2e.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
uv run pytest tests/test_state.py::TestStateSessionManagement

# Run a specific test function
uv run pytest tests/test_state.py::TestStateSessionManagement::test_create_new_session
```

### Run with Coverage

```bash
# Add pytest-cov to dev dependencies if needed
uv add --dev pytest-cov

# Run tests with coverage report
uv run pytest --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Structure

```
tests/
├── conftest.py                    # Shared pytest fixtures
├── test_state.py                  # Unit tests for state management
└── e2e/
    ├── __init__.py
    └── test_chat_e2e.py          # End-to-end browser tests
```

### Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Tests that use external services
- `@pytest.mark.e2e` - End-to-end browser tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.asyncio` - Async tests

## Writing Tests

### Unit Tests

Unit tests focus on testing individual functions and classes in isolation.

```python
import pytest
from app.state import State

def test_create_new_session():
    """Test that create_new_session creates a new chat session."""
    state = State()
    state.create_new_session()
    
    assert len(state.chat_sessions) == 1
    assert state.current_session_id != ""
```

### Using Fixtures

Fixtures help reduce code duplication and provide consistent test data:

```python
import pytest
from app.state import State

@pytest.fixture
def state():
    """Create a fresh State instance for each test."""
    return State()

@pytest.fixture
def state_with_session(state):
    """Create a State with one active session."""
    state.create_new_session()
    return state

def test_with_fixture(state_with_session):
    """Use the fixture in your test."""
    assert len(state_with_session.chat_sessions) == 1
```

### Testing Async Methods

Use `pytest.mark.asyncio` for async tests:

```python
import pytest
from app.state import State

@pytest.mark.asyncio
async def test_answer_creates_session(state):
    """Test async answer method."""
    state.question = "Hello"
    await state.answer()
    
    assert len(state.chat_sessions) >= 1
```

### Mocking External Dependencies

Mock external services to isolate your tests:

```python
from unittest.mock import patch, MagicMock

def test_with_mock(state):
    """Test with mocked external service."""
    mock_agent = MagicMock()
    mock_agent.run_stream.return_value = "Test response"
    
    with patch("app.state.Agent", return_value=mock_agent):
        # Your test code here
        pass
```

### E2E Tests with Playwright

E2E tests verify the complete user journey in a real browser:

```python
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.e2e
def test_send_message(page: Page):
    """Test sending a message through the UI."""
    # Find the input field
    message_input = page.locator('textarea').first
    
    # Type and send message
    message_input.fill("Hello")
    message_input.press("Enter")
    
    # Verify message appears
    expect(page.locator('text="Hello"')).to_be_visible()
```

## Best Practices

### 1. **Test State, Not UI Structure**

Focus on testing business logic in State classes:

```python
# Good: Tests business logic
def test_delete_session_behavior(state):
    state.create_new_session()
    session_id = state.current_session_id
    state.delete_session(session_id)
    assert len(state.chat_sessions) == 1  # New session created

# Avoid: Testing internal implementation details
def test_session_list_length():
    # This is fragile and tests implementation, not behavior
    pass
```

### 2. **Use Descriptive Test Names**

Test names should describe what they test and expected behavior:

```python
# Good
def test_delete_session_creates_new_when_none_left(state):
    pass

# Bad
def test_delete(state):
    pass
```

### 3. **Arrange-Act-Assert Pattern**

Structure tests clearly:

```python
def test_switch_session(state):
    # Arrange: Set up test data
    state.create_new_session()
    first_id = state.current_session_id
    state.create_new_session()
    
    # Act: Perform the action being tested
    state.switch_session(first_id)
    
    # Assert: Verify the outcome
    assert state.current_session_id == first_id
```

### 4. **Test Edge Cases**

Always test boundary conditions:

```python
def test_update_title_exactly_50_chars(state_with_session):
    """Test edge case: exactly 50 characters."""
    question_50 = "a" * 50
    state_with_session.update_session_title(question_50)
    
    session = state_with_session.chat_sessions[0]
    assert session.title == question_50
    assert "..." not in session.title
```

### 5. **Isolate Tests**

Each test should be independent:

```python
# Good: Uses fixture for fresh state
def test_create_session(state):
    state.create_new_session()
    assert len(state.chat_sessions) == 1

# Bad: Depends on global state or previous tests
global_state = State()
def test_create_session_bad():
    global_state.create_new_session()  # Affects other tests!
```

### 6. **Use Markers to Organize Tests**

Tag tests for easy filtering:

```python
@pytest.mark.unit
def test_fast_unit():
    pass

@pytest.mark.e2e
def test_browser_flow():
    pass

@pytest.mark.slow
def test_performance():
    pass
```

### 7. **Prefer Unit Tests Over E2E Tests**

- Unit tests are faster and more reliable
- Use E2E tests only for critical user journeys
- Aim for 80% unit tests, 20% E2E tests

### 8. **Add data-testid Attributes for E2E**

Make E2E selectors more robust:

```python
# In your Reflex component:
rx.button("Submit", data_testid="submit-button")

# In your E2E test:
page.click('[data-testid="submit-button"]')
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: |
          uv sync --extra dev
          uv run playwright install chromium
      
      - name: Run unit tests
        run: uv run pytest -m unit
      
      - name: Run E2E tests
        run: uv run pytest -m e2e
```

## Debugging Tests

### Run Tests in Verbose Mode

```bash
uv run pytest -vv
```

### See Print Statements

```bash
uv run pytest -s
```

### Run Specific Test with Debugging

```bash
uv run pytest tests/test_state.py::test_name -vv -s
```

### Debug E2E Tests (Non-Headless)

Modify `pytest.ini` or run with:

```bash
uv run pytest --headed --slowmo 100
```

### Use pytest-pdb for Debugging

```bash
uv run pytest --pdb  # Drop into debugger on failure
```

## Common Issues

### E2E Tests Fail to Start Server

- Ensure dependencies are installed: `uv sync --extra dev`
- Check if port 3001 is available
- Increase timeout in `conftest.py`

### Import Errors in Tests

- Run `uv sync` to ensure all dependencies are installed
- Ensure the project root is in PYTHONPATH
- Check that `conftest.py` properly stubs dependencies

### Async Tests Not Running

- Ensure `pytest-asyncio` is installed: `uv sync --extra dev`
- Add `@pytest.mark.asyncio` decorator
- Check `pytest.ini` has `asyncio_mode = "auto"`

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Playwright for Python](https://playwright.dev/python/)
- [pytest-playwright](https://github.com/microsoft/playwright-pytest)
- [Reflex Testing Guide](https://reflex.dev/docs/)
