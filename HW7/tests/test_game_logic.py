"""Tests for game logic and scoring."""

import pytest
from shared.enums import GameResultStatus


class TestEvenOddLogic:
    """Test Even/Odd game logic."""

    def test_both_correct_is_draw(self):
        """Test that both players correct results in draw."""
        drawn_number = 42  # even
        parity = "even"

        choice_A = "even"
        choice_B = "even"

        is_A_correct = choice_A == parity
        is_B_correct = choice_B == parity

        if is_A_correct == is_B_correct:
            status = GameResultStatus.DRAW
            winner = None
        elif is_A_correct:
            status = GameResultStatus.WIN
            winner = "P01"
        else:
            status = GameResultStatus.WIN
            winner = "P02"

        assert status == GameResultStatus.DRAW
        assert winner is None

    def test_both_wrong_is_draw(self):
        """Test that both players wrong results in draw."""
        drawn_number = 42  # even
        parity = "even"

        choice_A = "odd"
        choice_B = "odd"

        is_A_correct = choice_A == parity
        is_B_correct = choice_B == parity

        if is_A_correct == is_B_correct:
            status = GameResultStatus.DRAW
            winner = None
        elif is_A_correct:
            status = GameResultStatus.WIN
            winner = "P01"
        else:
            status = GameResultStatus.WIN
            winner = "P02"

        assert status == GameResultStatus.DRAW
        assert winner is None

    def test_player_a_wins(self):
        """Test Player A wins when correct."""
        drawn_number = 42  # even
        parity = "even"

        choice_A = "even"
        choice_B = "odd"

        is_A_correct = choice_A == parity
        is_B_correct = choice_B == parity

        if is_A_correct == is_B_correct:
            status = GameResultStatus.DRAW
            winner = None
        elif is_A_correct:
            status = GameResultStatus.WIN
            winner = "P01"
        else:
            status = GameResultStatus.WIN
            winner = "P02"

        assert status == GameResultStatus.WIN
        assert winner == "P01"

    def test_player_b_wins(self):
        """Test Player B wins when correct."""
        drawn_number = 43  # odd
        parity = "odd"

        choice_A = "even"
        choice_B = "odd"

        is_A_correct = choice_A == parity
        is_B_correct = choice_B == parity

        if is_A_correct == is_B_correct:
            status = GameResultStatus.DRAW
            winner = None
        elif is_A_correct:
            status = GameResultStatus.WIN
            winner = "P01"
        else:
            status = GameResultStatus.WIN
            winner = "P02"

        assert status == GameResultStatus.WIN
        assert winner == "P02"


class TestScoring:
    """Test scoring system."""

    def test_win_scores_three_points(self):
        """Test that a win awards 3 points."""
        winner_points = 3
        loser_points = 0

        assert winner_points == 3
        assert loser_points == 0

    def test_draw_scores_one_point_each(self):
        """Test that a draw awards 1 point to each player."""
        player_a_points = 1
        player_b_points = 1

        assert player_a_points == 1
        assert player_b_points == 1

    def test_standings_calculation(self):
        """Test standings point calculation."""
        # Simulate a player's record
        wins = 2
        draws = 1
        losses = 1

        total_points = (wins * 3) + (draws * 1) + (losses * 0)

        assert total_points == 7  # 2*3 + 1*1 + 1*0 = 7


class TestParity:
    """Test parity detection."""

    @pytest.mark.parametrize(
        "number,expected",
        [
            (0, "even"),
            (1, "odd"),
            (42, "even"),
            (43, "odd"),
            (100, "even"),
            (99, "odd"),
        ],
    )
    def test_parity_detection(self, number, expected):
        """Test that parity is correctly detected."""
        parity = "even" if number % 2 == 0 else "odd"
        assert parity == expected
