#!/usr/bin/env python
# download_data.py
# 2025-09-13
# Roscoe

"""
Download raw data directly from UCI ML Repository.
"""

import os
import stat
from hashlib import md5
from http.client import HTTPSConnection
from urllib.parse import urlparse


DEST: str = "data/diabetes+130-us+hospitals+for+years+1999-2008.zip"
URL: str = ("https://archive.ics.uci.edu/static/public/296/"
            "diabetes+130-us+hospitals+for+years+1999-2008.zip")
REF_MD5: str = "47100411c9c5ca5d97bd9f21afc02e69"


def md5sum(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return md5(f.read()).hexdigest()


def download_raw(dest: str=DEST, url: str=URL, ref_md5: str=REF_MD5) -> None:
    if os.path.isfile(dest) and md5sum(dest) == ref_md5:
        print(f"'{dest}' exists and the MD5 is correct. Skipping download.")
        return
    purl = urlparse(url)
    try:
        conn = HTTPSConnection(purl.netloc)
        conn.request("GET", purl.path)
        resp = conn.getresponse()
        if resp.status == 200:
            print("Downloading file ... ", end="")
            if os.path.isfile(dest):
                os.chmod(dest, stat.S_IWRITE)
            with open(dest, "wb") as f:
                f.write(resp.read())
            os.chmod(dest, stat.S_IREAD)
            print("\033[32mDone!\033[0m")
            print("Checking MD5 ... ", end="")
            if md5sum(dest) == ref_md5:
                print("\033[32mAll good!\033[0m")
            else:
                print("\033[31mFAILED! \033[33mThis "
                      "might not be the file you're looking for...\033[0m")
        else:
            print(f"Failed to download file from '{url}'. "
                  f"Status: {resp.status}, Reason: {resp.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "conn" in locals() and conn:
            conn.close()



def main() -> None:
    download_raw()


if __name__ == "__main__":
    main()


