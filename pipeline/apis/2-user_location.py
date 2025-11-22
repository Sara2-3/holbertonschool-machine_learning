#!/usr/bin/env python3
"""
A script that prints the location of a specific user.
"""

import requests
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit()

    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 200:
        location = response.json().get("location")
        if location:
            print(location)
        else:
            print("No location")

    elif response.status_code == 404:
        print("Not found")

    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-Ratelimit-Reset", 0))
        now = int(time.time())
        minutes = (reset_time - now) // 60
        if minutes < 0:
            minutes = 0
        print(f"Reset in {minutes} min")
