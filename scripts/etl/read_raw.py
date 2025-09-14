#!/usr/bin/env python
# read_raw.py
# 2025-09-14
# Roscoe

"""
Read raw data. Do not process the raw data, but process the metadata into a
useful form.
"""

import csv
import os
import sys
from collections.abc import Iterator
from io import TextIOWrapper
from inspect import currentframe, getframeinfo
from typing import Callable
from zipfile import ZipFile

# Just in case. Assume `download_data.py` is in the same directory.
_ = os.path.dirname(getframeinfo(currentframe()).filename)
if _ not in sys.path:
    sys.path.append(_)

from download_data import download_raw, md5sum


ZIP_SRC: str = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"
REF_MD5: str = "47100411c9c5ca5d97bd9f21afc02e69"
PATH1: str = "diabetic_data.csv"
PATH2: str = "IDS_mapping.csv"


def read_zip_csv(
    zip_src: str,
    path: str,
    func: Callable=csv.DictReader,
    ref_md5: str=REF_MD5
) -> Iterator[dict[str, str]]:
    """
    Read a CSV file within a ZIP archive. If file does not exist, or is
    incorrect, download directly from UCI ML Repository.
    """
    if not os.path.isfile(zip_src) or md5sum(zip_src) != ref_md5:
        download_data(zip_src)
    with ZipFile(zip_src) as z:
        with z.open(path) as f:
            for row in func(TextIOWrapper(f)):
                yield row


def read_data(zip_src: str=ZIP_SRC, path: str=PATH1) -> list[dict[str, str]]:
    """
    Read the raw data.
    """
    return [*read_zip_csv(zip_src, path)]


def read_maps(zip_src: str=ZIP_SRC, path: str=PATH2) -> list[dict[str, str]]:
    """
    Read the IDs mapping file.

    This one is a bit different: 3 tables in 1 file, separated by empty rows.
    Arrange it properly into a single flattened table.
    """
    rows2 = []
    next_tbl = True
    for row in read_zip_csv(zip_src, path, func=csv.reader):
        if all(not _ for _ in row):
            next_tbl = True
        elif next_tbl:
            next_tbl = False
            name, _description = row
        else:
            k, v = row
            v = v.strip(".").strip()
            rows2.append({"variable": name, "option": k, "description": v})
    return rows2


def list_to_dict(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """
    Convert flat table to nested dictionary.
    """
    dct = {}
    for row in rows:
        name, option, description = row.values()
        dct[name] = dct.get(name, {})
        dct[name][option] = description
    return dct


def get_maps(
    zip_src: str=ZIP_SRC,
    path: str=PATH2
) -> dict[str, dict[str, str]]:
    """
    Convenience function to get nested dictionary of ID mappings.
    """
    map_list = read_maps(zip_src, path)
    map_dict = list_to_dict(map_list)
    return map_dict



def main() -> None:
    assert "read_data" in globals()
    assert "get_maps" in globals()
    print("This module is not meant to be called directly. "
          "You would probably like to use the following functions:"
          "\n- `read_data`"
          "\n- `get_maps`")


if __name__ == "__main__":
    main()


