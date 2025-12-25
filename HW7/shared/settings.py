"""
Configuration settings for the Multi-Agent League Management System.

This module uses pydantic-settings to load configuration from environment
variables and .env files. All settings are centralized here for easy management.
"""

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # League Manager Configuration
    league_manager_host: str = Field(
        default="0.0.0.0", description="League Manager host"
    )
    league_manager_port: int = Field(default=8000, description="League Manager port")
    league_id: str = Field(default="league_2025_even_odd", description="League ID")

    # Referee Configuration
    referee_host: str = Field(default="0.0.0.0", description="Referee host")
    referee_port: int = Field(default=8001, description="Referee port")
    referee_id: str = Field(default="REF01", description="Referee ID")
    referee_display_name: str = Field(
        default="Referee Alpha", description="Referee display name"
    )
    referee_version: str = Field(default="1.0.0", description="Referee version")
    referee_max_concurrent_matches: int = Field(
        default=10, description="Maximum concurrent matches for referee"
    )

    # Player Configuration
    player_base_port: int = Field(default=8101, description="Base port for players")
    player_host: str = Field(default="0.0.0.0", description="Player host")
    player_ids: str = Field(
        default="P01,P02,P03,P04", description="Player IDs (comma-separated)"
    )
    player_version: str = Field(default="1.0.0", description="Player version")

    # Game Configuration
    game_type: str = Field(default="even_odd", description="Game type")
    game_invitation_timeout: int = Field(
        default=5, description="Game invitation timeout (seconds)"
    )
    parity_choice_timeout: int = Field(
        default=30, description="Parity choice timeout (seconds)"
    )
    http_request_timeout: int = Field(
        default=10, description="HTTP request timeout (seconds)"
    )
    random_number_min: int = Field(default=0, description="Minimum random number")
    random_number_max: int = Field(default=100, description="Maximum random number")

    # Launcher Configuration
    startup_wait_time: int = Field(default=5, description="Startup wait time (seconds)")
    shutdown_sigint_timeout: int = Field(
        default=5, description="SIGINT shutdown timeout (seconds)"
    )
    shutdown_sigterm_timeout: int = Field(
        default=5, description="SIGTERM shutdown timeout (seconds)"
    )

    # Logging Configuration
    log_level: str = Field(default="DEBUG", description="Log level")

    # Computed Properties

    @property
    def league_manager_url(self) -> str:
        """Full League Manager URL."""
        return f"http://localhost:{self.league_manager_port}/mcp"

    @property
    def referee_url(self) -> str:
        """Full Referee URL."""
        return f"http://localhost:{self.referee_port}/mcp"

    @property
    def referee_contact_endpoint(self) -> str:
        """Referee contact endpoint."""
        return self.referee_url

    @property
    def player_ids_list(self) -> List[str]:
        """List of player IDs."""
        return [pid.strip() for pid in self.player_ids.split(",")]

    @property
    def player_ports(self) -> List[int]:
        """List of player ports."""
        return [self.player_base_port + i for i in range(len(self.player_ids_list))]

    def get_player_url(self, port: int) -> str:
        """Get player URL for a given port."""
        return f"http://localhost:{port}/mcp"

    def get_player_endpoint(self, player_index: int) -> str:
        """Get player endpoint for a given player index."""
        port = self.player_ports[player_index]
        return self.get_player_url(port)


# Global settings instance
settings = Settings()
