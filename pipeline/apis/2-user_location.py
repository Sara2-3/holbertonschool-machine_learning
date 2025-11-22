#!/usr/bin/env python3
"""
Një skript Python që printon lokacionin e një përdoruesi specifik në GitHub
duke përdorur GitHub API.
Trajton status kode 404 (Not found) dhe 403 (Rate limit exceeded).
"""
import requests
import sys
import time

def user_location():
    if len(sys.argv) < 2:
        sys.exit(1)

    url = sys.argv[1]

    try:
        response = requests.get(url)
        status_code = response.status_code

        if status_code == 404:
            print("Not found")
            return

        if status_code == 403:
            if 'X-Ratelimit-Reset' in response.headers:
                reset_time_unix = int(response.headers['X-Ratelimit-Reset'])
                current_time_unix = int(time.time())
                time_to_reset_sec = reset_time_unix - current_time_unix

                if time_to_reset_sec <= 0:
                    min_to_reset = 0
                else:
                    min_to_reset = (time_to_reset_sec + 59) // 60

                print(f"Reset in {min_to_reset} min")
                return
            else:
                print("Not found")
                return

        elif status_code == 200:
            user_data = response.json()
            location = user_data.get('location')

            if location is None or location.strip() == "":
                print("Not found")
            else:
                print(location)
            return

        else:
            print("Not found")
            return

    except requests.exceptions.RequestException as e:
        print("Not found")
        sys.exit(1)

if __name__ == '__main__':
    user_location()
