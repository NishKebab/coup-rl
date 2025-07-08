"""Command line interface for Coup game."""

from typing import List, Optional
from .game import CoupGame
from .types import ActionType, Character
from .player import CoupPlayer


class CoupCLI:
    """Command line interface for playing Coup."""
    
    def __init__(self) -> None:
        self.game = CoupGame()
    
    def run(self) -> None:
        """Main game loop."""
        print("Welcome to Coup!")
        print("================")
        
        self._setup_game()
        
        while self.game.state.phase.value == "Active":
            self._display_game_state()
            self._handle_player_turn()
            
            if self.game.state.is_game_over():
                break
        
        self._display_winner()
    
    def _setup_game(self) -> None:
        """Set up the game with players."""
        while True:
            try:
                num_players = int(input("Number of players (2-6): "))
                if 2 <= num_players <= 6:
                    break
                print("Please enter a number between 2 and 6")
            except ValueError:
                print("Please enter a valid number")
        
        for i in range(num_players):
            name = input(f"Enter name for player {i + 1}: ")
            self.game.add_player(name)
        
        self.game.start_game()
        print(f"\nGame started with {len(self.game.state.players)} players!")
    
    def _display_game_state(self) -> None:
        """Display current game state."""
        print("\n" + "="*50)
        print("GAME STATE")
        print("="*50)
        
        for i, player in enumerate(self.game.state.players):
            if player.is_eliminated:
                print(f"{i+1}. {player.name}: ELIMINATED")
            else:
                current = " (CURRENT)" if player == self.game.state.get_current_player() else ""
                print(f"{i+1}. {player.name}: {player.coins} coins, {len(player.cards)} cards{current}")
        
        if self.game.state.action_log:
            print(f"\nLast action: {self.game.state.action_log[-1]}")
    
    def _handle_player_turn(self) -> None:
        """Handle a single player's turn."""
        current_player = self.game.state.get_current_player()
        print(f"\n{current_player.name}'s turn")
        
        if current_player.must_coup():
            print("You must coup (10+ coins)")
            self._handle_coup_action()
            return
        
        available_actions = self.game.get_available_actions(current_player)
        
        print("Available actions:")
        for i, action in enumerate(available_actions):
            print(f"{i+1}. {action.value}")
        
        try:
            choice = int(input("Choose action (number): ")) - 1
            if 0 <= choice < len(available_actions):
                action_type = available_actions[choice]
                self._execute_action(action_type)
            else:
                print("Invalid choice")
        except ValueError:
            print("Please enter a valid number")
    
    def _execute_action(self, action_type: ActionType) -> None:
        """Execute the chosen action."""
        target = None
        
        if action_type in [ActionType.COUP, ActionType.ASSASSINATE, ActionType.STEAL]:
            target = self._choose_target()
            if not target:
                return
        
        result = self.game.attempt_action(action_type, target)
        print(f"\n{result.message}")
        
        if not result.success:
            return
        
        if self.game.state.challenge_window_open:
            self._handle_challenge_window()
        elif self.game.state.block_window_open:
            self._handle_block_window()
        else:
            self.game.resolve_pending_action()
    
    def _choose_target(self) -> Optional[CoupPlayer]:
        """Choose a target for targeted actions."""
        active_players = [p for p in self.game.state.players 
                         if not p.is_eliminated and p != self.game.state.get_current_player()]
        
        if not active_players:
            print("No valid targets")
            return None
        
        print("Choose target:")
        for i, player in enumerate(active_players):
            print(f"{i+1}. {player.name}")
        
        try:
            choice = int(input("Choose target (number): ")) - 1
            if 0 <= choice < len(active_players):
                return active_players[choice]
            else:
                print("Invalid choice")
                return None
        except ValueError:
            print("Please enter a valid number")
            return None
    
    def _handle_challenge_window(self) -> None:
        """Handle challenge phase."""
        print("\nChallenge window open!")
        
        other_players = [p for p in self.game.state.players 
                        if not p.is_eliminated and p != self.game.state.get_current_player()]
        
        for player in other_players:
            challenge = input(f"{player.name}, do you want to challenge? (y/n): ").lower()
            if challenge == 'y':
                result = self.game.challenge_action(player)
                print(f"\n{result.message}")
                
                if "Challenge succeeded" in result.message:
                    return
                elif self.game.state.block_window_open:
                    self._handle_block_window()
                    return
                else:
                    self.game.resolve_pending_action()
                    return
        
        print("No challenges made")
        if self.game.state.block_window_open:
            self._handle_block_window()
        else:
            self.game.resolve_pending_action()
    
    def _handle_block_window(self) -> None:
        """Handle block phase."""
        print("\nBlock window open!")
        
        potential_blockers = self.game.block_system.get_potential_blockers(
            self.game.state.pending_action, self.game.state.pending_target
        )
        
        for player in potential_blockers:
            block = input(f"{player.name}, do you want to block? (y/n): ").lower()
            if block == 'y':
                result = self.game.block_action(player)
                print(f"\n{result.message}")
                return
        
        print("No blocks made")
        self.game.resolve_pending_action()
    
    def _handle_coup_action(self) -> None:
        """Handle mandatory coup action."""
        target = self._choose_target()
        if target:
            result = self.game.attempt_action(ActionType.COUP, target)
            print(f"\n{result.message}")
    
    def _display_winner(self) -> None:
        """Display the winner."""
        winner = self.game.state.get_winner()
        if winner:
            print(f"\n🎉 {winner.name} wins the game! 🎉")
        else:
            print("\nGame ended in a draw")


def main() -> None:
    """Main entry point."""
    cli = CoupCLI()
    cli.run()


if __name__ == "__main__":
    main()