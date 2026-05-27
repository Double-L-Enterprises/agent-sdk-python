#!/usr/bin/env python3

import asyncio
import tempfile
import os
from pathlib import Path

# Import the file_ops module
import sys
sys.path.insert(0, 'src')
from claude_agent_sdk.tools.file_ops import _write

async def test_new_file():
    """Test writing a new file (should work normally)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "new_test.txt"
        result = await _write({"file_path": str(test_file), "content": "Hello, world!"})
        print(f"New file test: {result}")
        
        # Verify file exists and content is correct
        assert test_file.exists()
        assert test_file.read_text() == "Hello, world!"
        print("✓ New file test passed")

async def test_existing_file():
    """Test overwriting an existing file (should use safe-write)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "existing_test.py"
        # Create initial file
        test_file.write_text("print('original')")
        
        # Overwrite with valid Python
        result = await _write({"file_path": str(test_file), "content": "print('updated')"})
        print(f"Existing file test: {result}")
        
        # Verify file exists and content is correct
        assert test_file.exists()
        assert test_file.read_text() == "print('updated')"
        print("✓ Existing file test passed")
        
        # Verify no .bak or .new files remain
        assert not test_file.with_suffix(".py.bak").exists()
        assert not test_file.with_suffix(".py.new").exists()

async def test_invalid_python():
    """Test overwriting with invalid Python (should fail safely)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "invalid_test.py"
        # Create initial file
        test_file.write_text("print('original')")
        
        # Try to overwrite with invalid Python
        result = await _write({"file_path": str(test_file), "content": "print('invalid syntax'"))
        print(f"Invalid Python test: {result}")
        
        # Verify original file is intact
        assert test_file.exists()
        assert test_file.read_text() == "print('original')"
        print("✓ Invalid Python test passed - original preserved")
        
        # Verify backup exists and .new is cleaned up
        backup_file = test_file.with_suffix(".py.bak")
        new_file = test_file.with_suffix(".py.new")
        assert backup_file.exists()
        assert not new_file.exists()

async def main():
    await test_new_file()
    await test_existing_file()
    await test_invalid_python()
    print("\nAll tests passed! ✨")

if __name__ == "__main__":
    asyncio.run(main())