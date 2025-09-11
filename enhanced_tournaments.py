"""
Enhanced Multi-Agent Tournament System

This module provides an enhanced multi-agent tournament system with improved
tournament types, better performance tracking, and advanced analytics.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from model.rl_system.integration.rl_system import RLSystemManager
from model.rl_system.environments.trading_env import TradingEnvironment

class TournamentType(Enum):
    """Types of tournaments."""
    ROUND_ROBIN = "round_robin"
    ELIMINATION = "elimination"
    SWISS = "swiss"
    LADDER = "ladder"
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"

class GameMode(Enum):
    """Game modes for multi-agent interactions."""
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    MIXED = "mixed"

class TournamentStatus(Enum):
    """Status of a tournament."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class AgentProfile:
    """Profile of a tournament agent."""
    agent_id: str
    algorithm_type: str
    algorithm_config: Dict[str, Any]
    training_episodes: int
    performance_metrics: Dict[str, float]
    creation_timestamp: datetime
    last_updated: datetime

@dataclass
class Match:
    """Single match between agents."""
    match_id: str
    tournament_id: str
    agent_ids: List[str]
    environment_config: Dict[str, Any]
    game_mode: GameMode
    episodes: int
    results: Dict[str, Any]
    winner: Optional[str]
    timestamp: datetime
    duration_seconds: float

@dataclass
class TournamentResult:
    """Results from a complete tournament."""
    tournament_id: str
    tournament_type: TournamentType
    participants: List[AgentProfile]
    matches: List[Match]
    final_rankings: List[Tuple[str, float]]  # (agent_id, score)
    champion: str
    tournament_stats: Dict[str, Any]
    timestamp: datetime

class EnhancedMultiAgentTournament:
    """Enhanced multi-agent tournament system."""
    
    def __init__(self, tournament_id: str, tournament_type: TournamentType):
        """Initialize the tournament system.
        
        Args:
            tournament_id: Unique identifier for the tournament
            tournament_type: Type of tournament to run
        """
        self.tournament_id = tournament_id
        self.tournament_type = tournament_type
        self.status = TournamentStatus.PENDING
        self.participants: List[AgentProfile] = []
        self.matches: List[Match] = []
        self.results: Optional[TournamentResult] = None
        self.logger = logging.getLogger(f"tournament_{tournament_id}")
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.match_counter = 0
        
    def add_participant(self, agent_profile: AgentProfile) -> bool:
        """Add a participant to the tournament.
        
        Args:
            agent_profile: Agent profile to add
            
        Returns:
            True if participant was added, False otherwise
        """
        if self.status != TournamentStatus.PENDING:
            self.logger.warning("Cannot add participants to active tournament")
            return False
            
        # Check if agent already exists
        for participant in self.participants:
            if participant.agent_id == agent_profile.agent_id:
                self.logger.warning(f"Agent {agent_profile.agent_id} already registered")
                return False
                
        self.participants.append(agent_profile)
        self.logger.info(f"Added participant: {agent_profile.agent_id}")
        return True
        
    def remove_participant(self, agent_id: str) -> bool:
        """Remove a participant from the tournament.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if participant was removed, False otherwise
        """
        if self.status != TournamentStatus.PENDING:
            self.logger.warning("Cannot remove participants from active tournament")
            return False
            
        for i, participant in enumerate(self.participants):
            if participant.agent_id == agent_id:
                removed = self.participants.pop(i)
                self.logger.info(f"Removed participant: {removed.agent_id}")
                return True
                
        self.logger.warning(f"Participant {agent_id} not found")
        return False
        
    def start_tournament(self, environment_config: Dict[str, Any]) -> bool:
        """Start the tournament.
        
        Args:
            environment_config: Configuration for trading environment
            
        Returns:
            True if tournament started successfully, False otherwise
        """
        if self.status != TournamentStatus.PENDING:
            self.logger.error("Tournament is not in pending state")
            return False
            
        if len(self.participants) < 2:
            self.logger.error("Need at least 2 participants to start tournament")
            return False
            
        self.status = TournamentStatus.RUNNING
        self.start_time = datetime.now()
        self.logger.info(f"Tournament {self.tournament_id} started with {len(self.participants)} participants")
        
        try:
            # Run the tournament based on type
            if self.tournament_type == TournamentType.ROUND_ROBIN:
                self._run_round_robin_tournament(environment_config)
            elif self.tournament_type == TournamentType.ELIMINATION:
                self._run_elimination_tournament(environment_config)
            elif self.tournament_type == TournamentType.SWISS:
                self._run_swiss_tournament(environment_config)
            elif self.tournament_type == TournamentType.LADDER:
                self._run_ladder_tournament(environment_config)
            else:
                self.logger.error(f"Unsupported tournament type: {self.tournament_type}")
                self.status = TournamentStatus.CANCELLED
                return False
                
            self.end_time = datetime.now()
            self.status = TournamentStatus.COMPLETED
            self._generate_results()
            self.logger.info(f"Tournament {self.tournament_id} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Tournament failed: {e}")
            self.status = TournamentStatus.CANCELLED
            return False
            
    def _run_round_robin_tournament(self, environment_config: Dict[str, Any]):
        """Run a round-robin tournament."""
        self.logger.info("Starting round-robin tournament")
        
        # Generate all pairwise matches
        matches_to_run = []
        for i in range(len(self.participants)):
            for j in range(i + 1, len(self.participants)):
                agent1 = self.participants[i]
                agent2 = self.participants[j]
                matches_to_run.append((agent1, agent2))
                
        # Run matches
        for agent1, agent2 in matches_to_run:
            match_result = self._run_match(
                [agent1, agent2],
                environment_config,
                GameMode.COMPETITIVE,
                episodes=10
            )
            self.matches.append(match_result)
            
    def _run_elimination_tournament(self, environment_config: Dict[str, Any]):
        """Run an elimination tournament."""
        self.logger.info("Starting elimination tournament")
        
        # For simplicity, we'll implement a single elimination bracket
        participants = self.participants.copy()
        
        round_num = 1
        while len(participants) > 1:
            self.logger.info(f"Elimination round {round_num} with {len(participants)} participants")
            
            next_round_participants = []
            # Pair participants
            for i in range(0, len(participants) - 1, 2):
                agent1 = participants[i]
                agent2 = participants[i + 1]
                
                match_result = self._run_match(
                    [agent1, agent2],
                    environment_config,
                    GameMode.COMPETITIVE,
                    episodes=15
                )
                self.matches.append(match_result)
                
                # Winner advances
                winner_id = match_result.winner
                for agent in [agent1, agent2]:
                    if agent.agent_id == winner_id:
                        next_round_participants.append(agent)
                        break
                        
            # Handle odd number of participants
            if len(participants) % 2 == 1:
                next_round_participants.append(participants[-1])
                self.logger.info(f"Bye given to {participants[-1].agent_id}")
                
            participants = next_round_participants
            round_num += 1
            
    def _run_swiss_tournament(self, environment_config: Dict[str, Any]):
        """Run a Swiss-system tournament."""
        self.logger.info("Starting Swiss-system tournament")
        
        # Initialize scores
        scores = {agent.agent_id: 0.0 for agent in self.participants}
        rounds_played = {agent.agent_id: 0 for agent in self.participants}
        
        # Run 3 rounds of Swiss tournament
        for round_num in range(3):
            self.logger.info(f"Swiss round {round_num + 1}")
            
            # Sort participants by score
            sorted_participants = sorted(
                self.participants,
                key=lambda x: scores[x.agent_id],
                reverse=True
            )
            
            # Pair participants
            used_agents = set()
            for i in range(len(sorted_participants)):
                if sorted_participants[i].agent_id in used_agents:
                    continue
                    
                # Find opponent with similar score who hasn't been paired
                opponent = None
                for j in range(i + 1, len(sorted_participants)):
                    candidate_id = sorted_participants[j].agent_id
                    if candidate_id not in used_agents:
                        opponent = sorted_participants[j]
                        break
                        
                if opponent:
                    agent1 = sorted_participants[i]
                    agent2 = opponent
                    
                    match_result = self._run_match(
                        [agent1, agent2],
                        environment_config,
                        GameMode.COMPETITIVE,
                        episodes=10
                    )
                    self.matches.append(match_result)
                    
                    # Update scores
                    if match_result.winner:
                        scores[match_result.winner] += 1.0
                    else:
                        # Draw
                        scores[agent1.agent_id] += 0.5
                        scores[agent2.agent_id] += 0.5
                        
                    used_agents.add(agent1.agent_id)
                    used_agents.add(agent2.agent_id)
                    rounds_played[agent1.agent_id] += 1
                    rounds_played[agent2.agent_id] += 1
                    
    def _run_ladder_tournament(self, environment_config: Dict[str, Any]):
        """Run a ladder tournament."""
        self.logger.info("Starting ladder tournament")
        
        # Sort participants by initial performance
        sorted_participants = sorted(
            self.participants,
            key=lambda x: x.performance_metrics.get('win_rate', 0.0),
            reverse=True
        )
        
        # Run matches between adjacent participants
        for i in range(len(sorted_participants) - 1):
            agent1 = sorted_participants[i]
            agent2 = sorted_participants[i + 1]
            
            match_result = self._run_match(
                [agent1, agent2],
                environment_config,
                GameMode.COMPETITIVE,
                episodes=10
            )
            self.matches.append(match_result)
            
    def _run_match(self, agents: List[AgentProfile], 
                   environment_config: Dict[str, Any],
                   game_mode: GameMode,
                   episodes: int) -> Match:
        """Run a single match between agents.
        
        Args:
            agents: List of agents to compete
            environment_config: Environment configuration
            game_mode: Game mode for the match
            episodes: Number of episodes to run
            
        Returns:
            Match result
        """
        start_time = datetime.now()
        
        # Create unique match ID
        match_id = f"{self.tournament_id}_match_{self.match_counter:04d}"
        self.match_counter += 1
        
        # Simulate match execution (in a real implementation, this would run the actual agents)
        match_results = {
            'episodes_played': episodes,
            'total_rewards': {agent.agent_id: np.random.random() * 100 for agent in agents},
            'avg_rewards': {agent.agent_id: np.random.random() * 10 for agent in agents},
        }
        
        # Determine winner (highest total reward)
        winner = max(match_results['total_rewards'].items(), key=lambda x: x[1])[0]
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        match = Match(
            match_id=match_id,
            tournament_id=self.tournament_id,
            agent_ids=[agent.agent_id for agent in agents],
            environment_config=environment_config,
            game_mode=game_mode,
            episodes=episodes,
            results=match_results,
            winner=winner,
            timestamp=start_time,
            duration_seconds=duration
        )
        
        self.logger.info(f"Match {match_id} completed. Winner: {winner}")
        return match
        
    def _generate_results(self):
        """Generate final tournament results."""
        if not self.matches:
            self.logger.warning("No matches played, cannot generate results")
            return
            
        # Calculate final rankings
        agent_scores = {}
        for match in self.matches:
            for agent_id in match.agent_ids:
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = 0.0
                    
            # Award points based on match result
            if match.winner:
                agent_scores[match.winner] += 1.0
            else:
                # Draw - split points
                for agent_id in match.agent_ids:
                    agent_scores[agent_id] += 0.5
                    
        # Sort by score
        final_rankings = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find champion
        champion = final_rankings[0][0] if final_rankings else "Unknown"
        
        # Calculate tournament stats
        tournament_stats = {
            'total_matches': len(self.matches),
            'total_duration_seconds': sum(match.duration_seconds for match in self.matches),
            'average_match_duration': np.mean([match.duration_seconds for match in self.matches]),
            'participant_count': len(self.participants)
        }
        
        self.results = TournamentResult(
            tournament_id=self.tournament_id,
            tournament_type=self.tournament_type,
            participants=self.participants,
            matches=self.matches,
            final_rankings=final_rankings,
            champion=champion,
            tournament_stats=tournament_stats,
            timestamp=datetime.now()
        )
        
    def get_results(self) -> Optional[TournamentResult]:
        """Get tournament results.
        
        Returns:
            Tournament results or None if tournament hasn't completed
        """
        return self.results
        
    def save_results(self, filepath: str = None) -> str:
        """Save tournament results to file.
        
        Args:
            filepath: File path to save results (optional)
            
        Returns:
            Path to saved file
        """
        if not self.results:
            raise ValueError("No results to save")
            
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"tournament_results_{self.tournament_id}_{timestamp}.json"
            
        # Convert results to dictionary for JSON serialization
        results_dict = asdict(self.results)
        
        # Convert enum objects to strings and datetime objects to strings
        results_dict['tournament_type'] = self.results.tournament_type.value
        results_dict['timestamp'] = self.results.timestamp.isoformat()
        for i, match in enumerate(results_dict['matches']):
            results_dict['matches'][i]['game_mode'] = match['game_mode'].value
            results_dict['matches'][i]['timestamp'] = match['timestamp'].isoformat() if hasattr(match['timestamp'], 'isoformat') else str(match['timestamp'])
            
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to {filepath}")
        return filepath
        
    def load_results(self, filepath: str) -> TournamentResult:
        """Load tournament results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded tournament results
        """
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
            
        # Convert back to TournamentResult
        # This is a simplified conversion - in practice, you'd need more detailed conversion
        self.results = TournamentResult(**results_dict)
        
        self.logger.info(f"Results loaded from {filepath}")
        return self.results
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a detailed tournament report.
        
        Returns:
            Dictionary containing tournament report
        """
        if not self.results:
            raise ValueError("No results available")
            
        report = {
            'tournament_id': self.results.tournament_id,
            'tournament_type': self.results.tournament_type.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            'participant_count': len(self.results.participants),
            'match_count': len(self.results.matches),
            'champion': self.results.champion,
            'final_rankings': self.results.final_rankings,
            'tournament_stats': self.results.tournament_stats,
            'participants': [
                {
                    'agent_id': p.agent_id,
                    'algorithm_type': p.algorithm_type,
                    'performance_metrics': p.performance_metrics
                }
                for p in self.results.participants
            ]
        }
        
        return report
        
    def plot_results(self, filepath: str = None):
        """Plot tournament results.
        
        Args:
            filepath: File path to save plot (optional)
        """
        if not self.results:
            raise ValueError("No results to plot")
            
        # Create ranking chart
        agent_ids = [rank[0] for rank in self.results.final_rankings]
        scores = [rank[1] for rank in self.results.final_rankings]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(agent_ids)), scores)
        plt.xlabel('Agents')
        plt.ylabel('Score')
        plt.title(f'Tournament Results - {self.tournament_id}')
        plt.xticks(range(len(agent_ids)), agent_ids, rotation=45)
        
        # Color the winner bar differently
        if agent_ids:
            bars[0].set_color('gold')
            
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Results plot saved to {filepath}")
        else:
            plt.show()
            plt.close()

# Tournament engine class for managing multiple tournaments
class TournamentEngine:
    """Engine for managing multiple tournaments."""
    
    def __init__(self):
        """Initialize the tournament engine."""
        self.tournaments: Dict[str, EnhancedMultiAgentTournament] = {}
        self.logger = logging.getLogger("tournament_engine")
        
    def create_tournament(self, tournament_id: str, tournament_type: TournamentType) -> EnhancedMultiAgentTournament:
        """Create a new tournament.
        
        Args:
            tournament_id: Unique identifier for the tournament
            tournament_type: Type of tournament to create
            
        Returns:
            Created tournament instance
        """
        if tournament_id in self.tournaments:
            raise ValueError(f"Tournament {tournament_id} already exists")
            
        tournament = EnhancedMultiAgentTournament(tournament_id, tournament_type)
        self.tournaments[tournament_id] = tournament
        
        self.logger.info(f"Created tournament {tournament_id} of type {tournament_type.value}")
        return tournament
        
    def get_tournament(self, tournament_id: str) -> Optional[EnhancedMultiAgentTournament]:
        """Get a tournament by ID.
        
        Args:
            tournament_id: Tournament ID
            
        Returns:
            Tournament instance or None if not found
        """
        return self.tournaments.get(tournament_id)
        
    def list_tournaments(self) -> List[Dict[str, Any]]:
        """List all tournaments.
        
        Returns:
            List of tournament information
        """
        return [
            {
                'tournament_id': t.tournament_id,
                'tournament_type': t.tournament_type.value,
                'status': t.status.value,
                'participant_count': len(t.participants),
                'match_count': len(t.matches)
            }
            for t in self.tournaments.values()
        ]
        
    def remove_tournament(self, tournament_id: str) -> bool:
        """Remove a tournament.
        
        Args:
            tournament_id: ID of tournament to remove
            
        Returns:
            True if tournament was removed, False otherwise
        """
        if tournament_id in self.tournaments:
            del self.tournaments[tournament_id]
            self.logger.info(f"Removed tournament {tournament_id}")
            return True
        return False

# Example usage
def run_sample_tournament():
    """Run a sample tournament to demonstrate functionality."""
    print("Enhanced Multi-Agent Tournament System")
    print("=" * 40)
    
    # Create tournament engine
    engine = TournamentEngine()
    
    # Create sample agents
    agents = [
        AgentProfile(
            agent_id=f"agent_{i}",
            algorithm_type="DQN",
            algorithm_config={"learning_rate": 0.001, "gamma": 0.99},
            training_episodes=1000,
            performance_metrics={"win_rate": 0.6 + 0.1 * i, "avg_reward": 50 + 10 * i},
            creation_timestamp=datetime.now(),
            last_updated=datetime.now()
        )
        for i in range(4)
    ]
    
    # Create tournament
    tournament = engine.create_tournament("sample_tournament", TournamentType.ROUND_ROBIN)
    
    # Add participants
    for agent in agents:
        tournament.add_participant(agent)
        
    print(f"Created tournament with {len(agents)} participants")
    
    # Start tournament (with mock environment config)
    env_config = {
        "symbol": "BTC/USDT",
        "features": ["open", "high", "low", "close", "volume"],
        "limit": 100
    }
    
    success = tournament.start_tournament(env_config)
    
    if success:
        print("Tournament completed successfully!")
        
        # Get results
        results = tournament.get_results()
        if results:
            print(f"Champion: {results.champion}")
            print("Final rankings:")
            for rank, (agent_id, score) in enumerate(results.final_rankings, 1):
                print(f"  {rank}. {agent_id}: {score}")
                
        # Generate report
        report = tournament.generate_report()
        print(f"Tournament duration: {report['duration_seconds']:.2f} seconds")
        
    else:
        print("Tournament failed to complete")
        
    print("\nTournament engine status:")
    tournaments = engine.list_tournaments()
    for t in tournaments:
        print(f"  {t['tournament_id']}: {t['status']} ({t['participant_count']} participants)")

if __name__ == "__main__":
    run_sample_tournament()