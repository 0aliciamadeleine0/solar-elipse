from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
import numpy as np
import pandas as pd
from skyfield.api import Loader, wgs84

ECLIPSE_DATE = dt.date(2026, 8, 12)
CENTRAL_LINE = [
    (43.35, -8.55),   # Galicja
    (42.90, -6.50),
    (42.40, -4.00),   # okolice Palencia/Burgos
    (41.90, -1.80),
    (41.55, 0.10),    # okolice Huesca/Lleida
    (41.20, 1.90),    # południe od Barcelony
    (40.60, 3.10),    # Morze Balearskie
]
HALF_WIDTH_DEG = 0.55
SEARCH_WINDOW_UTC = (
    dt.datetime(2026, 8, 12, 20, 0, 0),
    dt.datetime(2026, 8, 12, 21, 0, 0),
)
TIME_STEP_SECONDS = 1.0
