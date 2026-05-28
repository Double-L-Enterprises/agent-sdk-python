"""Consensus voting for multi-agent decisions.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class VoteMode(str, Enum):
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"

@dataclass
class AgentVote:
    agent_name: str
    choice: str
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class VoteRecord:
    topic: str
    votes: list[AgentVote] = field(default_factory=list)
    result: str | None = None
    mode: VoteMode = VoteMode.MAJORITY

class ConsensusManager:
    def __init__(self, mode: VoteMode = VoteMode.MAJORITY) -> None:
        self._mode = mode
        self._records: list[VoteRecord] = []

    def create_vote(self, topic: str) -> VoteRecord:
        record = VoteRecord(topic=topic, mode=self._mode)
        self._records.append(record)
        return record

    def cast_vote(self, record: VoteRecord, vote: AgentVote) -> None:
        record.votes.append(vote)

    def tally(self, record: VoteRecord) -> str:
        if not record.votes:
            return "no_votes"
        counts: dict[str, float] = {}
        for v in record.votes:
            w = v.confidence if self._mode == VoteMode.WEIGHTED else 1.0
            counts[v.choice] = counts.get(v.choice, 0.0) + w
        winner = max(counts, key=counts.get)
        record.result = winner
        return winner

    def get_records(self) -> list[VoteRecord]:
        return list(self._records)
