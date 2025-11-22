#!/usr/bin/env python3
"""
Test file
"""
import subprocess

result = subprocess.run(['./3-first_launch.py'], capture_output=True, text=True)
print(result.stdout.strip())
