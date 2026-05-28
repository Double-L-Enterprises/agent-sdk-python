"""Prompts for AutonomousRunner.
Created: 2026-05-27 23:00 CST
"""
AUTONOMOUS_AGENT_SYSTEM_PROMPT = (
    "You are an autonomous AI agent. Complete the given task thoroughly. "
    "When you are finished, say task complete."
)

def build_system_prompt(task: str) -> str:
    return AUTONOMOUS_AGENT_SYSTEM_PROMPT + chr(10) + chr(10) + "Your current task:" + chr(10) + task + chr(10) + chr(10) + "Begin working now."
