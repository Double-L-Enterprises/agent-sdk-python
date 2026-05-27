# Task: Locate claude-bridge Service and Venv

Created: 2026-05-20 (CST)

## What to Find
1. Full path to claude-bridge.service file
2. The ExecStart line from that service file
3. The Python executable path it uses
4. Verify the venv exists and list bin/python* files

## Context
- Running on Windows WSL2 Ubuntu
- claude-bridge is a systemd service (port :8020)
- Should be in user systemd location or /etc/systemd/system/

## Search Locations
- /home/larryloden/.config/systemd/user/
- ~/.config/systemd/user/
- /etc/systemd/system/
- Falls back to find command if above don't exist

## Output Required
- Full path to service file
- Exact ExecStart line
- Python interpreter location
- Venv status (exists/missing)
- Keep under 200 tokens

## Constraints
- Do NOT start any services
- Do NOT modify any files
- Do NOT install anything
