# Pandas

Notes for Tasks

# Task 4
Short Version
#!/usr/bin/env python3

import pandas as pd
import numpy as np


def array(df):
    """
    Selects last 10 rows of High and Close columns and converts to numpy array
    """
    return df[['High', 'Close']].tail(10).to_numpy()
