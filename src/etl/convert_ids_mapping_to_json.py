#!/usr/bin/env python
# convert_ids_mapping_to_json.py
# 2025-08-24
# Roscoe

"""
`IDS_mapping.csv` is not in the best format for its purpose. Convert it from
CSV to JSON. We might end up not using this information anyway, but who knows
we might find something interesting?
"""

import csv
import json
import os
import stat


MAPPING_FILE: str = "data/raw/IDS_mapping.csv"
DEST: str = "data/IDS_mapping.json"


def read_ids_mappings(source: str=MAPPING_FILE) -> dict[str, dict[str, str]]:
    """
    File is CSV-like, containing 3 lookup tables separated by newlines. Each
    lookup table has a header (the first is the variable name, then other is
    literally 'description') and 2 columns (the numeric value and it's
    corresponding description).

    Parse into a dictionary of dictionaries, level 1 keys will be the variable
    names, level 1 values will be mappings for that variable.
    """
    blocks = {}
    header = True
    with open(source) as f:
        for k, v in csv.reader(f):
            if header:
                header = False
                name = k
                blocks[name] = {}
            elif k == "":  # and v == "":
                header = True
            else:
                blocks[name][k] = v.strip()
    return blocks


def write_json(
    col: dict[dict[str, str]],
    dest: str,
    readonly: bool=True,
    verbose: bool=False
) -> None:
    if os.path.isfile(dest):
        os.chmod(dest, stat.S_IWRITE)
    with open(dest, "w") as f:
        json.dump(col, f, indent=4)
    if readonly:
        os.chmod(dest, stat.S_IREAD)
    if verbose:
        msg = f"File written: '{dest}'"
        if readonly:
            msg += ", read-only"
        print(msg)



def main() -> None:
    dct_of_dcts = read_ids_mappings()
    write_json(dct_of_dcts, DEST, verbose=True)


if __name__ == "__main__":
    main()


