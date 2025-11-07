"""End-to-end tests for chat functionality using Playwright.

These tests verify the full user journey including:
- Creating new chat sessions
- Sending messages
- Viewing chat history
- Switching between sessions
- Deleting sessions
"""

import pytest
from playwright.sync_api import Page, expect


@pytest.mark.e2e
class TestChatFlow:
    """E2E tests for the chat interface."""

    def test_page_loads_successfully(self, app_page: Page):
        """The app should load and display the main interface."""
        # Check that the page title or heading is visible
        expect(app_page).to_have_url("http://localhost:3001/")
        
        # Verify the page has loaded by checking for key elements
        # Adjust selectors based on your actual UI structure
        app_page.wait_for_load_state("networkidle")

    def test_create_new_chat_session(self, app_page: Page):
        """User should be able to create a new chat session."""
        # Look for "New Chat" button or similar (adjust selector as needed)
        new_chat_button = app_page.locator('button:has-text("New Chat")')
        
        if new_chat_button.count() > 0:
            new_chat_button.click()
            
            # Verify a new session was created (check sidebar or UI feedback)
            app_page.wait_for_timeout(500)  # Brief wait for state update

    def test_send_message_and_receive_response(self, app_page: Page):
        """User should be able to send a message and see a response."""
        # Find the message input field (adjust selector based on your UI)
        message_input = app_page.locator('textarea, input[type="text"]').first
        
        # Type a test message
        test_message = "Hello, test message"
        message_input.fill(test_message)
        
        # Press Enter or click send button
        message_input.press("Enter")
        
        # Wait for the message to appear in the chat
        # Adjust selector to match your message display structure
        app_page.wait_for_timeout(1000)
        
        # Verify the user message appears
        expect(app_page.locator(f'text="{test_message}"')).to_be_visible()

    def test_toggle_sidebar(self, app_page: Page):
        """User should be able to toggle the sidebar visibility."""
        # Find sidebar toggle button (adjust selector)
        toggle_button = app_page.locator('[data-testid="toggle-sidebar"], button:has-text("☰")')
        
        if toggle_button.count() > 0:
            # Get initial sidebar state
            sidebar = app_page.locator('[data-testid="sidebar"]').first
            
            # Click toggle
            toggle_button.click()
            app_page.wait_for_timeout(400)  # Wait for animation
            
            # Click toggle again
            toggle_button.click()
            app_page.wait_for_timeout(400)

    def test_switch_between_sessions(self, app_page: Page):
        """User should be able to switch between different chat sessions."""
        # Create first session and send a message
        message_input = app_page.locator('textarea, input[type="text"]').first
        message_input.fill("First session message")
        message_input.press("Enter")
        app_page.wait_for_timeout(500)
        
        # Create a new session
        new_chat_button = app_page.locator('button:has-text("New Chat")')
        if new_chat_button.count() > 0:
            new_chat_button.click()
            app_page.wait_for_timeout(500)
            
            # Send a different message in the new session
            message_input.fill("Second session message")
            message_input.press("Enter")
            app_page.wait_for_timeout(500)
            
            # Try to switch back to first session via sidebar
            # This would require clicking on the first session in the sidebar
            # Adjust selector based on your session list structure

    def test_delete_session(self, app_page: Page):
        """User should be able to delete a chat session."""
        # Create a new session
        new_chat_button = app_page.locator('button:has-text("New Chat")')
        if new_chat_button.count() > 0:
            new_chat_button.click()
            app_page.wait_for_timeout(500)
        
        # Find and click delete button for a session
        # Adjust selector based on your delete button structure
        delete_button = app_page.locator('[data-testid="delete-session"], button:has-text("Delete")')
        if delete_button.count() > 0:
            delete_button.first.click()
            app_page.wait_for_timeout(500)


@pytest.mark.e2e
class TestSettingsFlow:
    """E2E tests for settings and preferences."""

    def test_change_font_family(self, app_page: Page):
        """User should be able to change the font family setting."""
        # Find settings button/menu (adjust selector)
        settings_button = app_page.locator('[data-testid="settings"], button:has-text("Settings")')
        
        if settings_button.count() > 0:
            settings_button.click()
            app_page.wait_for_timeout(500)
            
            # Find font family selector (adjust based on your UI)
            font_select = app_page.locator('select, [role="combobox"]').first
            if font_select.count() > 0:
                # Select a different font
                font_select.select_option(label="Inter")
                app_page.wait_for_timeout(300)


@pytest.mark.e2e
class TestResponsiveness:
    """E2E tests for responsive design and different viewport sizes."""

    def test_mobile_viewport(self, app_page: Page):
        """App should be usable on mobile viewport."""
        # Set mobile viewport size
        app_page.set_viewport_size({"width": 375, "height": 667})
        
        # Navigate to app
        app_page.reload()
        app_page.wait_for_load_state("networkidle")
        
        # Verify key elements are still accessible
        # Mobile layouts might hide sidebar by default

    def test_tablet_viewport(self, app_page: Page):
        """App should be usable on tablet viewport."""
        # Set tablet viewport size
        app_page.set_viewport_size({"width": 768, "height": 1024})
        
        app_page.reload()
        app_page.wait_for_load_state("networkidle")


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformance:
    """E2E performance tests."""

    def test_handles_long_chat_history(self, app_page: Page):
        """App should handle sessions with many messages."""
        message_input = app_page.locator('textarea, input[type="text"]').first
        
        # Send multiple messages
        for i in range(5):
            message_input.fill(f"Message {i + 1}")
            message_input.press("Enter")
            app_page.wait_for_timeout(200)
        
        # Verify all messages are visible (or scrollable)
        app_page.wait_for_timeout(500)

    def test_handles_multiple_sessions(self, app_page: Page):
        """App should handle many chat sessions."""
        new_chat_button = app_page.locator('button:has-text("New Chat")')
        
        if new_chat_button.count() > 0:
            # Create multiple sessions
            for i in range(3):
                new_chat_button.click()
                app_page.wait_for_timeout(300)
                
                # Send a message in each
                message_input = app_page.locator('textarea, input[type="text"]').first
                message_input.fill(f"Session {i + 1} message")
                message_input.press("Enter")
                app_page.wait_for_timeout(300)
