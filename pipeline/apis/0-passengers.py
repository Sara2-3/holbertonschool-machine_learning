#!/usr/bin/env python3
"""
A script that returns the list of ships that can hold
a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    A function that returns the list of ships that can hold
    a given number of passengers
    """
    data = requests.get("https://swapi-api.hbtn.io/api/starships").json()
    availableShips = []
    while data.get("next") is not None:
        starships = data.get("results")
        for starship in starships:
            passengers = starship.get("passengers")
            if passengers is None or passengers in ["n/a", "unknown"]:
                continue
            if int(passengers.replace(",", "")) < passengerCount:
                continue
            availableShips.append(starship.get("name"))
        data = requests.get(data.get("next")).json()

    return availableShips
