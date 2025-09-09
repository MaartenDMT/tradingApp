"""
Multi-Agent Experiments and Tournament System

This module provides advanced multi-agent RL capabilities including
tournaments, competitive training, and collaborative experiments.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.rl_system.integration.rl_system import RLSystemManager
from model.rl_system.environments.trading_env import TradingEnvironment

class TournamentType(Enum):
    """Types of tournaments."""
    ROUND_ROBIN = "round_robin"
    ELIMINATION = "elimination"
    SWISS = "swiss"
    LADDER = "ladder"

class GameMode(Enum):
    """Game modes for multi-agent interactions."""
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    MIXED = "mixed"

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

class MultiAgentEnvironment:
    """Environment for multi-agent interactions."""
    
    def __init__(self, base_env_config: Dict[str, Any], num_agents: int, game_mode: GameMode):
        self.base_config = base_env_config
        self.num_agents = num_agents
        self.game_mode = game_mode
        self.environments = []
        self.current_states = []
        
        # Create separate environments for each agent
        for i in range(num_agents):
            env = TradingEnvironment(**base_env_config)
            self.environments.append(env)
    
    def reset(self) -> List[np.ndarray]:
        """Reset all environments and return initial states."""
        self.current_states = []
        for env in self.environments:
            state = env.reset()
            self.current_states.append(state)
        return self.current_states
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """Execute actions for all agents."""
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.environments, actions)):
            state, reward, done, info = env.step(action)
            
            # Modify rewards based on game mode
            if self.game_mode == GameMode.COMPETITIVE:
                # In competitive mode, one agent's gain is another's loss
                reward = self._apply_competitive_reward(reward, i, rewards)
            elif self.game_mode == GameMode.COOPERATIVE:
                # In cooperative mode, all agents share rewards
                reward = self._apply_cooperative_reward(reward, i)
            
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        self.current_states = next_states
        return next_states, rewards, dones, infos
    
    def _apply_competitive_reward(self, reward: float, agent_idx: int, other_rewards: List[float]) -> float:
        """Apply competitive reward structure."""
        # Penalty based on other agents' performance
        if other_rewards:
            competitive_penalty = np.mean(other_rewards) * 0.1
            return reward - competitive_penalty
        return reward
    
    def _apply_cooperative_reward(self, reward: float, agent_idx: int) -> float:
        """Apply cooperative reward structure."""
        # Bonus for positive performance (shared success)
        cooperation_bonus = max(0, reward * 0.1)
        return reward + cooperation_bonus

class AgentManager:
    """Manage multiple RL agents for tournaments."""
    
    def __init__(self):
        self.rl_manager = RLSystemManager()
        self.agents: Dict[str, Any] = {}
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_agent(
        self,
        agent_id: str,
        algorithm_type: str,
        algorithm_config: Optional[Dict[str, Any]] = None,
        training_episodes: int = 1000
    ) -> str:
        """Create and register a new agent."""
        
        try:
            # Create agent using RL system
            agent = self.rl_manager.create_agent(algorithm_type, **(algorithm_config or {}))
            
            # Store agent
            self.agents[agent_id] = agent
            
            # Create profile
            profile = AgentProfile(
                agent_id=agent_id,
                algorithm_type=algorithm_type,
                algorithm_config=algorithm_config or {},
                training_episodes=training_episodes,
                performance_metrics={},
                creation_timestamp=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.agent_profiles[agent_id] = profile
            
            self.logger.info(f"Created agent {agent_id} with algorithm {algorithm_type}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Error creating agent {agent_id}: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Any:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID."""
        return self.agent_profiles.get(agent_id)
    
    def update_agent_performance(self, agent_id: str, metrics: Dict[str, float]):
        """Update agent performance metrics."""
        if agent_id in self.agent_profiles:
            self.agent_profiles[agent_id].performance_metrics.update(metrics)
            self.agent_profiles[agent_id].last_updated = datetime.now()
    
    def train_agent(
        self,
        agent_id: str,
        environment_config: Dict[str, Any],
        episodes: int = 1000
    ) -> Dict[str, Any]:
        """Train a specific agent."""
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        env = TradingEnvironment(**environment_config)
        
        training_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Store experience for learning
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train if agent supports it
                if hasattr(agent, 'train') and episode > 50:
                    agent.train()
                
                state = next_state
                episode_reward += reward
            
            training_rewards.append(episode_reward)
        
        # Update performance metrics
        metrics = {
            'mean_training_reward': np.mean(training_rewards),
            'std_training_reward': np.std(training_rewards),
            'best_training_reward': np.max(training_rewards),
            'training_episodes': episodes
        }
        
        self.update_agent_performance(agent_id, metrics)
        
        return {
            'agent_id': agent_id,
            'training_rewards': training_rewards,
            'metrics': metrics
        }

class TournamentEngine:
    """Engine for running multi-agent tournaments."""
    
    def __init__(self, output_dir: str = "tournament_results"):
        self.agent_manager = AgentManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.active_tournaments: Dict[str, Dict[str, Any]] = {}
    
    async def create_tournament(
        self,
        tournament_type: TournamentType,
        agents: List[Dict[str, Any]],  # List of agent configs
        environment_config: Dict[str, Any],
        game_mode: GameMode = GameMode.COMPETITIVE,
        episodes_per_match: int = 100
    ) -> str:
        """Create and initialize a tournament."""
        
        tournament_id = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create all agents
        agent_ids = []
        for i, agent_config in enumerate(agents):
            agent_id = f"{tournament_id}_agent_{i}"
            self.agent_manager.create_agent(
                agent_id,
                agent_config['algorithm_type'],
                agent_config.get('algorithm_config', {}),
                agent_config.get('training_episodes', 1000)
            )
            agent_ids.append(agent_id)
        
        # Train agents individually first
        self.logger.info(f"Training {len(agent_ids)} agents for tournament {tournament_id}")
        training_tasks = []
        
        for agent_id in agent_ids:
            task = asyncio.create_task(
                self._train_agent_async(agent_id, environment_config)
            )
            training_tasks.append(task)
        
        # Wait for all training to complete
        await asyncio.gather(*training_tasks)
        
        # Store tournament info
        self.active_tournaments[tournament_id] = {
            'tournament_type': tournament_type,
            'agent_ids': agent_ids,
            'environment_config': environment_config,
            'game_mode': game_mode,
            'episodes_per_match': episodes_per_match,
            'created_at': datetime.now()
        }
        
        self.logger.info(f"Tournament {tournament_id} created with {len(agent_ids)} agents")
        return tournament_id
    
    async def _train_agent_async(self, agent_id: str, environment_config: Dict[str, Any]):
        """Train agent asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.agent_manager.train_agent,
            agent_id,
            environment_config,
            1000
        )
    
    async def run_tournament(self, tournament_id: str) -> TournamentResult:
        """Run a complete tournament."""
        
        if tournament_id not in self.active_tournaments:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        tournament_info = self.active_tournaments[tournament_id]
        tournament_type = tournament_info['tournament_type']
        
        self.logger.info(f"Starting tournament {tournament_id} of type {tournament_type.value}")
        
        if tournament_type == TournamentType.ROUND_ROBIN:
            return await self._run_round_robin_tournament(tournament_id)
        elif tournament_type == TournamentType.ELIMINATION:
            return await self._run_elimination_tournament(tournament_id)
        elif tournament_type == TournamentType.SWISS:
            return await self._run_swiss_tournament(tournament_id)
        elif tournament_type == TournamentType.LADDER:
            return await self._run_ladder_tournament(tournament_id)
        else:
            raise ValueError(f"Unsupported tournament type: {tournament_type}")
    
    async def _run_round_robin_tournament(self, tournament_id: str) -> TournamentResult:
        """Run round-robin tournament where every agent plays every other agent."""
        
        tournament_info = self.active_tournaments[tournament_id]
        agent_ids = tournament_info['agent_ids']
        matches = []
        
        # Generate all possible matches
        from itertools import combinations
        match_pairs = list(combinations(agent_ids, 2))
        
        self.logger.info(f"Running {len(match_pairs)} round-robin matches")
        
        # Run all matches
        for i, (agent1_id, agent2_id) in enumerate(match_pairs):
            match = await self._run_match(
                tournament_id,
                [agent1_id, agent2_id],
                tournament_info['environment_config'],
                tournament_info['game_mode'],
                tournament_info['episodes_per_match']
            )
            matches.append(match)
            
            self.logger.info(f"Completed match {i+1}/{len(match_pairs)}: {match.winner} won")
        
        # Calculate final rankings
        agent_scores = {agent_id: 0 for agent_id in agent_ids}
        
        for match in matches:
            if match.winner:
                agent_scores[match.winner] += 1
            
            # Add performance-based scoring
            for agent_id in match.agent_ids:
                if agent_id in match.results:
                    agent_scores[agent_id] += match.results[agent_id].get('normalized_score', 0)
        
        final_rankings = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        champion = final_rankings[0][0]
        
        # Create tournament result
        participants = [self.agent_manager.get_agent_profile(aid) for aid in agent_ids]
        
        tournament_stats = {
            'total_matches': len(matches),
            'total_episodes': sum(match.episodes for match in matches),
            'average_match_duration': np.mean([match.duration_seconds for match in matches]),
            'total_tournament_time': sum(match.duration_seconds for match in matches)
        }
        
        result = TournamentResult(
            tournament_id=tournament_id,
            tournament_type=TournamentType.ROUND_ROBIN,
            participants=participants,
            matches=matches,
            final_rankings=final_rankings,
            champion=champion,
            tournament_stats=tournament_stats,
            timestamp=datetime.now()
        )
        
        # Save results
        await self._save_tournament_result(result)
        
        return result
    
    async def _run_elimination_tournament(self, tournament_id: str) -> TournamentResult:
        """Run single-elimination tournament."""
        
        tournament_info = self.active_tournaments[tournament_id]
        agent_ids = tournament_info['agent_ids'].copy()
        matches = []
        round_num = 1
        
        while len(agent_ids) > 1:
            self.logger.info(f"Round {round_num}: {len(agent_ids)} agents remaining")
            
            round_matches = []
            next_round_agents = []
            
            # Pair up agents for this round
            for i in range(0, len(agent_ids), 2):
                if i + 1 < len(agent_ids):
                    # Normal match
                    agent1_id = agent_ids[i]
                    agent2_id = agent_ids[i + 1]
                    
                    match = await self._run_match(
                        tournament_id,
                        [agent1_id, agent2_id],
                        tournament_info['environment_config'],
                        tournament_info['game_mode'],
                        tournament_info['episodes_per_match']
                    )
                    
                    round_matches.append(match)
                    next_round_agents.append(match.winner)
                    
                else:
                    # Bye (odd number of agents)
                    next_round_agents.append(agent_ids[i])
            
            matches.extend(round_matches)
            agent_ids = next_round_agents
            round_num += 1
        
        # Champion is the last remaining agent
        champion = agent_ids[0]
        
        # Calculate rankings based on elimination order
        final_rankings = [(champion, len(matches) + 1)]  # Champion gets highest score
        
        # Add other agents based on when they were eliminated
        eliminated_agents = {}
        for match in reversed(matches):
            for agent_id in match.agent_ids:
                if agent_id != match.winner and agent_id not in eliminated_agents:
                    eliminated_agents[agent_id] = len(matches) - matches.index(match)
        
        for agent_id, score in eliminated_agents.items():
            final_rankings.append((agent_id, score))
        
        final_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Create result
        participants = [self.agent_manager.get_agent_profile(aid) for aid in tournament_info['agent_ids']]
        
        tournament_stats = {
            'total_matches': len(matches),
            'rounds': round_num - 1,
            'total_episodes': sum(match.episodes for match in matches),
            'average_match_duration': np.mean([match.duration_seconds for match in matches]),
            'total_tournament_time': sum(match.duration_seconds for match in matches)
        }
        
        result = TournamentResult(
            tournament_id=tournament_id,
            tournament_type=TournamentType.ELIMINATION,
            participants=participants,
            matches=matches,
            final_rankings=final_rankings,
            champion=champion,
            tournament_stats=tournament_stats,
            timestamp=datetime.now()
        )
        
        await self._save_tournament_result(result)
        return result
    
    async def _run_swiss_tournament(self, tournament_id: str) -> TournamentResult:
        """Run Swiss-style tournament with multiple rounds."""
        
        tournament_info = self.active_tournaments[tournament_id]
        agent_ids = tournament_info['agent_ids']
        num_rounds = min(len(agent_ids) - 1, 5)  # Limit to reasonable number of rounds
        
        matches = []
        agent_scores = {agent_id: 0 for agent_id in agent_ids}
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"Swiss Round {round_num}/{num_rounds}")
            
            # Pair agents based on current standings
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            round_matches = []
            
            paired_agents = set()
            for i in range(0, len(sorted_agents), 2):
                if i + 1 < len(sorted_agents):
                    agent1_id = sorted_agents[i][0]
                    agent2_id = sorted_agents[i + 1][0]
                    
                    if agent1_id not in paired_agents and agent2_id not in paired_agents:
                        match = await self._run_match(
                            tournament_id,
                            [agent1_id, agent2_id],
                            tournament_info['environment_config'],
                            tournament_info['game_mode'],
                            tournament_info['episodes_per_match']
                        )
                        
                        round_matches.append(match)
                        paired_agents.add(agent1_id)
                        paired_agents.add(agent2_id)
                        
                        # Update scores
                        if match.winner:
                            agent_scores[match.winner] += 1
            
            matches.extend(round_matches)
        
        # Final rankings
        final_rankings = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        champion = final_rankings[0][0]
        
        # Create result
        participants = [self.agent_manager.get_agent_profile(aid) for aid in agent_ids]
        
        tournament_stats = {
            'total_matches': len(matches),
            'rounds': num_rounds,
            'total_episodes': sum(match.episodes for match in matches),
            'average_match_duration': np.mean([match.duration_seconds for match in matches]),
            'total_tournament_time': sum(match.duration_seconds for match in matches)
        }
        
        result = TournamentResult(
            tournament_id=tournament_id,
            tournament_type=TournamentType.SWISS,
            participants=participants,
            matches=matches,
            final_rankings=final_rankings,
            champion=champion,
            tournament_stats=tournament_stats,
            timestamp=datetime.now()
        )
        
        await self._save_tournament_result(result)
        return result
    
    async def _run_ladder_tournament(self, tournament_id: str) -> TournamentResult:
        """Run ladder-style tournament with ongoing matches."""
        
        tournament_info = self.active_tournaments[tournament_id]
        agent_ids = tournament_info['agent_ids']
        
        # Initialize ladder with random order
        np.random.shuffle(agent_ids)
        ladder = {i: agent_ids[i] for i in range(len(agent_ids))}
        
        matches = []
        num_rounds = len(agent_ids) * 2  # Each agent can challenge multiple times
        
        for round_num in range(num_rounds):
            # Random agent challenges someone above them
            challenger_pos = np.random.randint(1, len(ladder))
            challenged_pos = np.random.randint(0, challenger_pos)
            
            challenger_id = ladder[challenger_pos]
            challenged_id = ladder[challenged_pos]
            
            match = await self._run_match(
                tournament_id,
                [challenger_id, challenged_id],
                tournament_info['environment_config'],
                tournament_info['game_mode'],
                tournament_info['episodes_per_match']
            )
            
            matches.append(match)
            
            # Update ladder if challenger wins
            if match.winner == challenger_id:
                # Swap positions
                ladder[challenger_pos] = challenged_id
                ladder[challenged_pos] = challenger_id
                
                self.logger.info(f"Ladder update: {challenger_id} moved up to position {challenged_pos}")
        
        # Final rankings based on ladder position
        final_rankings = [(ladder[i], len(ladder) - i) for i in range(len(ladder))]
        champion = ladder[0]
        
        # Create result
        participants = [self.agent_manager.get_agent_profile(aid) for aid in agent_ids]
        
        tournament_stats = {
            'total_matches': len(matches),
            'ladder_changes': sum(1 for match in matches if match.winner != match.agent_ids[1]),
            'total_episodes': sum(match.episodes for match in matches),
            'average_match_duration': np.mean([match.duration_seconds for match in matches]),
            'total_tournament_time': sum(match.duration_seconds for match in matches)
        }
        
        result = TournamentResult(
            tournament_id=tournament_id,
            tournament_type=TournamentType.LADDER,
            participants=participants,
            matches=matches,
            final_rankings=final_rankings,
            champion=champion,
            tournament_stats=tournament_stats,
            timestamp=datetime.now()
        )
        
        await self._save_tournament_result(result)
        return result
    
    async def _run_match(
        self,
        tournament_id: str,
        agent_ids: List[str],
        environment_config: Dict[str, Any],
        game_mode: GameMode,
        episodes: int
    ) -> Match:
        """Run a single match between agents."""
        
        start_time = datetime.now()
        match_id = f"{tournament_id}_match_{len(self.active_tournaments.get(tournament_id, {}).get('matches', []))}"
        
        # Create multi-agent environment
        multi_env = MultiAgentEnvironment(environment_config, len(agent_ids), game_mode)
        
        # Get agents
        agents = [self.agent_manager.get_agent(agent_id) for agent_id in agent_ids]
        
        agent_total_rewards = {agent_id: [] for agent_id in agent_ids}
        
        # Run episodes
        for episode in range(episodes):
            states = multi_env.reset()
            episode_rewards = {agent_id: 0 for agent_id in agent_ids}
            done = False
            
            while not done:
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(agents):
                    action = agent.act(states[i], explore=False)  # No exploration in tournament
                    actions.append(action)
                
                # Step environment
                next_states, rewards, dones, infos = multi_env.step(actions)
                
                # Update episode rewards
                for i, agent_id in enumerate(agent_ids):
                    episode_rewards[agent_id] += rewards[i]
                
                states = next_states
                done = any(dones)
            
            # Store episode results
            for agent_id in agent_ids:
                agent_total_rewards[agent_id].append(episode_rewards[agent_id])
        
        # Calculate match results
        agent_mean_rewards = {
            agent_id: np.mean(rewards) for agent_id, rewards in agent_total_rewards.items()
        }
        
        # Determine winner
        winner = max(agent_mean_rewards.keys(), key=lambda x: agent_mean_rewards[x])
        
        # Normalize scores for ranking
        max_reward = max(agent_mean_rewards.values())
        min_reward = min(agent_mean_rewards.values())
        
        normalized_scores = {}
        if max_reward != min_reward:
            for agent_id, reward in agent_mean_rewards.items():
                normalized_scores[agent_id] = (reward - min_reward) / (max_reward - min_reward)
        else:
            normalized_scores = {agent_id: 0.5 for agent_id in agent_ids}
        
        results = {
            agent_id: {
                'mean_reward': agent_mean_rewards[agent_id],
                'total_reward': sum(agent_total_rewards[agent_id]),
                'std_reward': np.std(agent_total_rewards[agent_id]),
                'normalized_score': normalized_scores[agent_id],
                'episode_rewards': agent_total_rewards[agent_id]
            }
            for agent_id in agent_ids
        }
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return Match(
            match_id=match_id,
            tournament_id=tournament_id,
            agent_ids=agent_ids,
            environment_config=environment_config,
            game_mode=game_mode,
            episodes=episodes,
            results=results,
            winner=winner,
            timestamp=start_time,
            duration_seconds=duration
        )
    
    async def _save_tournament_result(self, result: TournamentResult):
        """Save tournament results to file."""
        
        # Convert to serializable format
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        # Convert match timestamps
        for match_dict in result_dict['matches']:
            match_dict['timestamp'] = match_dict['timestamp'].isoformat() if isinstance(match_dict['timestamp'], datetime) else match_dict['timestamp']
        
        # Convert participant timestamps
        for participant_dict in result_dict['participants']:
            if participant_dict:
                participant_dict['creation_timestamp'] = participant_dict['creation_timestamp'].isoformat() if isinstance(participant_dict['creation_timestamp'], datetime) else participant_dict['creation_timestamp']
                participant_dict['last_updated'] = participant_dict['last_updated'].isoformat() if isinstance(participant_dict['last_updated'], datetime) else participant_dict['last_updated']
        
        # Save JSON
        result_path = self.output_dir / f"{result.tournament_id}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Tournament result saved to {result_path}")

# Global tournament engine
tournament_engine = TournamentEngine()
