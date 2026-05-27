"""Consensus voting for multi-agent decision-making.

ConsensusManager provides structured voting and disagreement resolution for
agent teams. Agents evaluate options independently, then vote using one of
four modes: MAJORITY, WEIGHTED, UNANIMOUS, or RANKED.

When agents disagree, resolve_disagreement() runs a structured debate (max 3
rounds) then calls a final vote. Deadlocks escalate to the orchestrator.
Full transcripts are stored in the MessageBus for audit.

Created: 2026-05-27 CST
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Voting modes ─────────────────────────────────────────────────────────────

class VoteMode(str, Enum):
    """Determines how individual votes are tallied into a decision."""

    MAJORITY = "MAJORITY"
    """Simple majority. Most votes wins. Ties broken alphabetically."""

    WEIGHTED = "WEIGHTED"
    """Votes are weighted by the agent's expertise_weight for the topic."""

    UNANIMOUS = "UNANIMOUS"
    """All agents must agree. Any disagreement triggers escalation."""

    RANKED = "RANKED"
    """Ranked-choice (instant-runoff). Agents rank options 1..N."""


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AgentVote:
    """A single agent's vote on a topic."""

    agent_name: str
    choice: str            # The option the agent voted for (MAJORITY / WEIGHTED)
    ranking: list[str]     # Ordered preference list (RANKED mode)
    reasoning: str         # The agent's stated rationale
    confidence: float      # 0.0–1.0 self-reported confidence
    weight: float = 1.0    # Effective vote weight (applied in WEIGHTED mode)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class VoteRecord:
    """Full record of a voting session."""

    topic: str
    options: list[str]
    mode: VoteMode
    votes: list[AgentVote]
    winner: str | None            # Winning option, or None if deadlocked
    deadlocked: bool
    rounds: list[dict[str, Any]]  # Debate transcript (if resolve_disagreement)
    vote_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "vote_id": self.vote_id,
            "topic": self.topic,
            "options": self.options,
            "mode": self.mode.value,
            "votes": [
                {
                    "agent": v.agent_name,
                    "choice": v.choice,
                    "ranking": v.ranking,
                    "reasoning": v.reasoning,
                    "confidence": v.confidence,
                    "weight": v.weight,
                    "timestamp": v.timestamp,
                }
                for v in self.votes
            ],
            "winner": self.winner,
            "deadlocked": self.deadlocked,
            "rounds": self.rounds,
            "timestamp": self.timestamp,
        }


# ── ConsensusManager ──────────────────────────────────────────────────────────

class ConsensusManager:
    """Structured decision-making for multi-agent teams.

    Coordinates voting sessions and disagreement resolution between agents
    in a TeamManager. Uses the MessageBus to deliver vote requests and
    collect responses.

    Usage::
        cm = ConsensusManager(team_manager=tm)

        # Simple majority vote
        record = await cm.vote(
            topic="Which database should we use?",
            options=["PostgreSQL", "MongoDB", "SQLite"],
            agents=["planner", "backend-dev", "architect"],
            mode=VoteMode.MAJORITY,
        )
        print(record.winner)

        # Resolve a disagreement
        record = await cm.resolve_disagreement(
            agent_a="backend-dev",
            agent_a_position="We should use REST",
            agent_b="architect",
            agent_b_position="We should use GraphQL",
            topic="API design pattern for the user service",
        )
    """

    def __init__(
        self,
        team_manager: Any,  # TeamManager — avoid circular import with Any
        agent_weights: dict[str, float] | None = None,
        max_vote_wait_seconds: int = 180,
        max_debate_rounds: int = 3,
    ) -> None:
        """Initialize the ConsensusManager.

        Args:
            team_manager: The TeamManager instance whose agents will vote.
            agent_weights: Dict of {agent_name: weight} for WEIGHTED voting.
                Defaults to 1.0 for all agents.
            max_vote_wait_seconds: Seconds to wait for each agent's vote response.
            max_debate_rounds: Maximum rounds before declaring deadlock.
        """
        self.tm = team_manager
        self.agent_weights = agent_weights or {}
        self.max_vote_wait_seconds = max_vote_wait_seconds
        self.max_debate_rounds = max_debate_rounds
        self._vote_history: list[VoteRecord] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    async def vote(
        self,
        topic: str,
        options: list[str],
        agents: list[str],
        mode: VoteMode = VoteMode.MAJORITY,
        thread_id: str | None = None,
    ) -> VoteRecord:
        """Ask a set of agents to evaluate options and vote.

        Each agent receives a structured vote request via the MessageBus and
        responds with their choice, ranking, and reasoning. Results are tallied
        according to the mode.

        Args:
            topic: The question or decision being voted on.
            options: The choices agents can vote for.
            agents: Names of agents who will participate in the vote.
            mode: Voting mode (MAJORITY, WEIGHTED, UNANIMOUS, RANKED).
            thread_id: Optional thread ID for grouping vote messages.

        Returns:
            VoteRecord with winner (or None if deadlocked) and full transcript.

        Raises:
            ValueError: If options or agents list is empty.
        """
        if not options:
            raise ValueError("options must be non-empty")
        if not agents:
            raise ValueError("agents must be non-empty")

        vote_id = str(uuid.uuid4())[:8]
        effective_thread_id = thread_id or f"vote-{vote_id}"

        logger.info(
            "Starting vote '%s' on topic: %s | mode=%s | agents=%s",
            vote_id, topic[:80], mode.value, agents,
        )

        # Deliver vote request to each agent
        vote_prompt = self._build_vote_prompt(topic, options, mode, vote_id)
        for agent_name in agents:
            await self.tm.send_to(
                agent_name,
                vote_prompt,
                from_name="consensus-manager",
                thread_id=effective_thread_id,
            )

        # Collect responses
        collected_votes = await self._collect_votes(
            agents=agents,
            vote_id=vote_id,
            thread_id=effective_thread_id,
            options=options,
        )

        # Tally
        winner, deadlocked = self._tally(collected_votes, options, mode)

        record = VoteRecord(
            topic=topic,
            options=options,
            mode=mode,
            votes=collected_votes,
            winner=winner,
            deadlocked=deadlocked,
            rounds=[],
            vote_id=vote_id,
        )

        self._vote_history.append(record)
        self._store_record_in_bus(record)

        if deadlocked:
            logger.warning("Vote '%s' deadlocked — escalation required", vote_id)
        else:
            logger.info("Vote '%s' resolved: winner=%s", vote_id, winner)

        return record

    async def resolve_disagreement(
        self,
        agent_a: str,
        agent_a_position: str,
        agent_b: str,
        agent_b_position: str,
        topic: str,
        arbiter_agents: list[str] | None = None,
        mode: VoteMode = VoteMode.MAJORITY,
    ) -> VoteRecord:
        """Resolve a disagreement between two agents via structured debate + vote.

        Protocol:
          1. agent_a and agent_b receive each other's positions.
          2. Up to max_debate_rounds rounds of rebuttal exchange.
          3. After debate, all participating agents (a, b, + arbiters) vote.
          4. If still deadlocked, escalate: record.deadlocked = True.

        Args:
            agent_a: First agent's name.
            agent_a_position: agent_a's stated position.
            agent_b: Second agent's name.
            agent_b_position: agent_b's stated position.
            topic: Description of the disagreement.
            arbiter_agents: Additional agents who observe the debate and vote
                at the end. Useful for breaking ties (odd number of voters).
            mode: Voting mode for the final decision. Defaults to MAJORITY.

        Returns:
            VoteRecord with the final decision. Check record.deadlocked if
            no majority was reached — escalate to human orchestrator.
        """
        debate_thread = f"debate-{str(uuid.uuid4())[:8]}"
        rounds: list[dict[str, Any]] = []
        all_voters = [agent_a, agent_b] + (arbiter_agents or [])

        logger.info(
            "resolve_disagreement: %s vs %s on '%s' (thread=%s)",
            agent_a, agent_b, topic[:80], debate_thread,
        )

        # Seed both agents with opposing positions
        seed_a = (
            f"DISAGREEMENT RESOLUTION — YOUR POSITION vs OPPONENT\n\n"
            f"Topic: {topic}\n\n"
            f"Your position: {agent_a_position}\n\n"
            f"Opponent ({agent_b}) position: {agent_b_position}\n\n"
            f"Round 1 of up to {self.max_debate_rounds}: "
            f"Send your strongest rebuttal to '{agent_b}' via SendTeamMessage "
            f"with thread_id='{debate_thread}'. Be specific. Max 300 words."
        )
        seed_b = (
            f"DISAGREEMENT RESOLUTION — YOUR POSITION vs OPPONENT\n\n"
            f"Topic: {topic}\n\n"
            f"Your position: {agent_b_position}\n\n"
            f"Opponent ({agent_a}) position: {agent_a_position}\n\n"
            f"Wait for {agent_a}'s rebuttal, then respond via SendTeamMessage "
            f"with thread_id='{debate_thread}'. Be specific. Max 300 words."
        )
        await self.tm.send_to(agent_a, seed_a, from_name="consensus-manager", thread_id=debate_thread)
        await self.tm.send_to(agent_b, seed_b, from_name="consensus-manager", thread_id=debate_thread)

        # Notify arbiters (observers only — vote at the end)
        for arbiter in (arbiter_agents or []):
            await self.tm.send_to(
                arbiter,
                (
                    f"DEBATE OBSERVER — you will vote at the end\n\n"
                    f"Topic: {topic}\n"
                    f"{agent_a} position: {agent_a_position}\n"
                    f"{agent_b} position: {agent_b_position}\n\n"
                    f"Observe thread_id='{debate_thread}'. "
                    f"You will receive a vote request when the debate concludes."
                ),
                from_name="consensus-manager",
                thread_id=debate_thread,
            )

        # Run debate rounds
        seen_ids: set[str] = set()
        start_ms = int(time.time() * 1000)

        for round_num in range(1, self.max_debate_rounds + 1):
            speaker = agent_a if round_num % 2 == 1 else agent_b
            listener = agent_b if round_num % 2 == 1 else agent_a

            round_entry: dict[str, Any] = {"round": round_num, "from": speaker, "messages": []}
            deadline = time.time() + self.max_vote_wait_seconds

            message_received = False
            while time.time() < deadline:
                all_msgs = self.tm.bus.get_all()
                new_msgs = [
                    m for m in all_msgs
                    if m.thread_id == debate_thread
                    and m.from_agent == speaker
                    and m.to_agent in (listener, "*")
                    and m.message_id not in seen_ids
                    and m.timestamp_ms >= start_ms
                ]
                if new_msgs:
                    latest = max(new_msgs, key=lambda m: m.timestamp_ms)
                    seen_ids.add(latest.message_id)
                    round_entry["messages"].append({
                        "from": latest.from_agent,
                        "to": latest.to_agent,
                        "content": latest.content,
                        "timestamp": latest.timestamp,
                    })
                    message_received = True

                    # Check for early consensus
                    content_upper = latest.content.upper()
                    if "AGREE" in content_upper or "ACCEPT" in content_upper or "LGTM" in content_upper:
                        round_entry["early_consensus"] = True
                        rounds.append(round_entry)
                        logger.info("Early consensus detected in debate round %d", round_num)
                        break

                    # Prompt next speaker if not final round
                    if round_num < self.max_debate_rounds:
                        next_prompt = (
                            f"DEBATE ROUND {round_num + 1} — respond to this rebuttal:\n\n"
                            f"{latest.content}\n\n"
                            f"Address the strongest point. Concede where they are right. "
                            f"Hold firm where they are not. Max 300 words. "
                            f"Reply to '{speaker}' via SendTeamMessage "
                            f"with thread_id='{debate_thread}'."
                        )
                        await self.tm.send_to(
                            listener, next_prompt,
                            from_name="consensus-manager", thread_id=debate_thread,
                        )
                    break

                await asyncio.sleep(5)

            if not message_received:
                round_entry["timeout"] = True
                logger.warning("Debate round %d timed out waiting for %s", round_num, speaker)

            rounds.append(round_entry)

            # Stop if early consensus was flagged
            if round_entry.get("early_consensus"):
                break

        # Final vote after debate
        options = [agent_a_position, agent_b_position]
        vote_record = await self.vote(
            topic=f"Final decision: {topic}",
            options=options,
            agents=all_voters,
            mode=mode,
            thread_id=f"{debate_thread}-final-vote",
        )

        # Attach debate transcript to vote record
        vote_record.rounds = rounds
        self._store_record_in_bus(vote_record)

        return vote_record

    def vote_history(self) -> list[VoteRecord]:
        """Return all vote records from this session."""
        return list(self._vote_history)

    def get_vote(self, vote_id: str) -> VoteRecord | None:
        """Look up a specific vote record by ID."""
        for r in self._vote_history:
            if r.vote_id == vote_id:
                return r
        return None

    # ── Internal: prompt building ──────────────────────────────────────────────

    def _build_vote_prompt(
        self,
        topic: str,
        options: list[str],
        mode: VoteMode,
        vote_id: str,
    ) -> str:
        options_str = "\n".join(f"  {i + 1}. {opt}" for i, opt in enumerate(options))

        if mode == VoteMode.RANKED:
            instruction = (
                "Rank ALL options from most preferred (1) to least preferred. "
                "Reply with a JSON object:\n"
                '{"vote_id": "' + vote_id + '", "ranking": ["first choice", "second choice", ...], '
                '"reasoning": "your analysis", "confidence": 0.0-1.0}'
            )
        else:
            instruction = (
                "Choose ONE option. "
                "Reply with a JSON object:\n"
                '{"vote_id": "' + vote_id + '", "choice": "exact option text", '
                '"reasoning": "your analysis", "confidence": 0.0-1.0}'
            )

        return (
            f"CONSENSUS VOTE REQUEST (vote_id={vote_id})\n\n"
            f"Topic: {topic}\n\n"
            f"Options:\n{options_str}\n\n"
            f"Mode: {mode.value}\n\n"
            f"{instruction}\n\n"
            f"Be objective. Base your decision on technical merit, not social dynamics. "
            f"Include your vote JSON in your response output — "
            f"do NOT send it via SendTeamMessage."
        )

    # ── Internal: response collection ─────────────────────────────────────────

    async def _collect_votes(
        self,
        agents: list[str],
        vote_id: str,
        thread_id: str,
        options: list[str],
    ) -> list[AgentVote]:
        """Poll the message bus for vote responses from all agents."""
        collected: dict[str, AgentVote] = {}
        start_ms = int(time.time() * 1000)
        deadline = time.time() + self.max_vote_wait_seconds

        while time.time() < deadline and len(collected) < len(agents):
            all_msgs = self.tm.bus.get_all()

            for msg in all_msgs:
                if msg.from_agent not in agents:
                    continue
                if msg.from_agent in collected:
                    continue
                if msg.timestamp_ms < start_ms:
                    continue

                vote = self._parse_vote_from_message(
                    agent_name=msg.from_agent,
                    content=msg.content,
                    vote_id=vote_id,
                    options=options,
                )
                if vote is not None:
                    collected[msg.from_agent] = vote

            if len(collected) < len(agents):
                await asyncio.sleep(5)

        # Timeout: record abstentions for non-responding agents
        for agent_name in agents:
            if agent_name not in collected:
                logger.warning("Agent '%s' did not vote within timeout — recording abstention", agent_name)
                collected[agent_name] = AgentVote(
                    agent_name=agent_name,
                    choice=options[0] if options else "",
                    ranking=list(options),
                    reasoning="ABSTAIN — no response within timeout",
                    confidence=0.0,
                    weight=self.agent_weights.get(agent_name, 1.0),
                )

        return list(collected.values())

    def _parse_vote_from_message(
        self,
        agent_name: str,
        content: str,
        vote_id: str,
        options: list[str],
    ) -> AgentVote | None:
        """Attempt to extract a structured vote from a message content.

        Tries JSON parse first, then falls back to heuristic matching.
        Returns None if no valid vote can be extracted.
        """
        # Try JSON extraction
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])

                if data.get("vote_id") != vote_id:
                    return None  # Different vote — skip

                choice = data.get("choice", "")
                ranking = data.get("ranking", [choice] if choice else list(options))
                reasoning = data.get("reasoning", "")
                confidence = float(data.get("confidence", 0.5))

                return AgentVote(
                    agent_name=agent_name,
                    choice=choice,
                    ranking=ranking,
                    reasoning=reasoning,
                    confidence=min(1.0, max(0.0, confidence)),
                    weight=self.agent_weights.get(agent_name, 1.0),
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Heuristic fallback: look for option text in the message
        content_lower = content.lower()
        for option in options:
            if option.lower() in content_lower:
                return AgentVote(
                    agent_name=agent_name,
                    choice=option,
                    ranking=[option],
                    reasoning=content[:500],
                    confidence=0.4,  # Lower confidence for heuristic match
                    weight=self.agent_weights.get(agent_name, 1.0),
                )

        return None

    # ── Internal: tallying ─────────────────────────────────────────────────────

    def _tally(
        self,
        votes: list[AgentVote],
        options: list[str],
        mode: VoteMode,
    ) -> tuple[str | None, bool]:
        """Tally votes and return (winner, is_deadlocked)."""
        if not votes:
            return None, True

        if mode == VoteMode.UNANIMOUS:
            return self._tally_unanimous(votes, options)
        elif mode == VoteMode.WEIGHTED:
            return self._tally_weighted(votes, options)
        elif mode == VoteMode.RANKED:
            return self._tally_ranked(votes, options)
        else:  # MAJORITY (default)
            return self._tally_majority(votes, options)

    def _tally_majority(
        self,
        votes: list[AgentVote],
        options: list[str],
    ) -> tuple[str | None, bool]:
        tally: dict[str, int] = {opt: 0 for opt in options}
        for v in votes:
            if v.choice in tally:
                tally[v.choice] += 1

        if not tally:
            return None, True

        max_votes = max(tally.values())
        winners = [opt for opt, count in tally.items() if count == max_votes]

        # Tie: pick alphabetically first (deterministic)
        return sorted(winners)[0], False

    def _tally_weighted(
        self,
        votes: list[AgentVote],
        options: list[str],
    ) -> tuple[str | None, bool]:
        tally: dict[str, float] = {opt: 0.0 for opt in options}
        for v in votes:
            if v.choice in tally:
                tally[v.choice] += v.weight

        if not tally:
            return None, True

        max_weight = max(tally.values())
        winners = [opt for opt, w in tally.items() if w == max_weight]
        return (sorted(winners)[0], False) if winners else (None, True)

    def _tally_unanimous(
        self,
        votes: list[AgentVote],
        options: list[str],
    ) -> tuple[str | None, bool]:
        choices = {v.choice for v in votes if v.choice}
        if len(choices) == 1:
            winner = next(iter(choices))
            return winner, False
        # Disagreement → deadlock; caller must escalate to orchestrator
        return None, True

    def _tally_ranked(
        self,
        votes: list[AgentVote],
        options: list[str],
    ) -> tuple[str | None, bool]:
        """Instant-runoff voting on ranked ballots.

        Each round: count first-choice votes among remaining options.
        If any option gets >50%, it wins. Otherwise eliminate the option with
        the fewest first-choice votes and repeat.
        """
        remaining = list(options)
        ballots = [list(v.ranking) for v in votes]

        while len(remaining) > 1:
            first_pref_count: dict[str, int] = {opt: 0 for opt in remaining}
            for ballot in ballots:
                for choice in ballot:
                    if choice in first_pref_count:
                        first_pref_count[choice] += 1
                        break

            total_votes = sum(first_pref_count.values())
            if total_votes == 0:
                break

            # Check for majority winner
            for opt, count in first_pref_count.items():
                if count > total_votes / 2:
                    return opt, False

            # Eliminate lowest-scoring option (alphabetical tie-break)
            min_votes = min(first_pref_count.values())
            candidates_for_elimination = sorted(
                [opt for opt, c in first_pref_count.items() if c == min_votes]
            )
            eliminated = candidates_for_elimination[0]
            remaining.remove(eliminated)

        if remaining:
            return remaining[0], False
        return None, True

    # ── Internal: bus storage ──────────────────────────────────────────────────

    def _store_record_in_bus(self, record: VoteRecord) -> None:
        """Write vote record to the message bus _all directory for audit trail."""
        from .message_bus import Message

        try:
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            msg = Message(
                from_agent="consensus-manager",
                to_agent="_vote-archive",
                content=json.dumps(record.to_dict(), indent=2),
                timestamp=now,
                thread_id=f"vote-{record.vote_id}",
            )
            self.tm.bus.send(msg)
        except Exception as exc:
            logger.warning("Failed to store vote record in bus: %s", exc)


__all__ = [
    "VoteMode",
    "AgentVote",
    "VoteRecord",
    "ConsensusManager",
]
