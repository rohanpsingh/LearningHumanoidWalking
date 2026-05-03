from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def parse_mocap_csv(filepath, usecols):
    """Process a mocap CSV file exported from Motive/NatNet format.

    Returns a DataFrame with columns:
        Frame, Time, RotX, RotY, RotZ, RotW, PosX, PosY, PosZ
    """
    with open(filepath) as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("Frame"):
            header_index = i
            break
    else:
        raise ValueError("Could not find header row starting with 'Frame'.")

    df = pd.read_csv(filepath, skiprows=header_index, usecols=usecols)
    df.columns = ["Frame", "Time", "RotX", "RotY", "RotZ", "RotW", "PosX", "PosY", "PosZ"]
    return df
