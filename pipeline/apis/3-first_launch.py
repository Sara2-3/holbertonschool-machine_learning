#!/usr/bin/env python3
"""
A script that displays the first launch.
"""

import requests

if __name__ == "__main__":
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"

    launches = requests.get(launches_url).json()

    first_launch_time = float('inf')
    first_launch = None

    for launch in launches:
        launch_time = launch.get('date_unix')
        if launch_time < first_launch_time:
            first_launch_time = launch_time
            first_launch = launch

    launch_name = first_launch.get('name')
    launch_date = first_launch.get('date_local')
    rocket_id = first_launch.get('rocket')
    launchpad_id = first_launch.get('launchpad')

    rocket_data = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    ).json()
    rocket_name = rocket_data.get('name')

    launchpad_data = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    ).json()
    launchpad_name = launchpad_data.get('name')
    launchpad_location = launchpad_data.get('locality')

    print(
        f"{launch_name} ({launch_date}) {rocket_name} - "
        f"{launchpad_name} ({launchpad_location})"
    )
