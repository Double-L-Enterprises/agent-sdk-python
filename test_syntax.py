import ast
import os

# Read the file content and parse it
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "src", "claude_agent_sdk", "tools", "file_ops.py")
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

try:
    ast.parse(content)
    print("Syntax is valid!")
except SyntaxError as e:
    print(f"Syntax error: {e}")