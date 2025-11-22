#!/usr/bin/env python3
"""
A script that returns the list of names of
the home planets of all sentient species
"""
import requests


def sentientPlanets():
    """
    A function that returns the list of names of the
    home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species"
    sentient_planets = []
    while url is not None:
        data = requests.get(url).json()
        species = data.get("results")
        for specie in species:
            if specie.get('designation') == 'sentient' or \
                    specie.get('classification') == 'sentient':
                homeworld = specie.get("homeworld")

                if homeworld is None:
                    continue
                sentient_planets.append(
                    requests.get(homeworld).json().get("name")
                )

        url = data.get("next")

    return sentient_planets
