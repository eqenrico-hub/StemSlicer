#!/usr/bin/env python3
"""
StemSlicer — License Key Generator

YOUR TOOL ONLY. Never distribute this file.

Usage:
    python generate_keys.py buyer@email.com
    python generate_keys.py buyer@email.com another@buyer.com
"""

import sys
import hmac
import hashlib
import base64


def _get_license_secret():
    _p1 = b"U3RlbVNsaWNlci1I"
    _p2 = b"TUFDLVBYRU1JVU0t"
    _p3 = b"UEFZUEFMLTVFVVIt"
    _p4 = b"MjAyNExJQ0VOU0U="
    return base64.b64decode(_p1 + _p2 + _p3 + _p4)


def generate_key(email):
    secret = _get_license_secret()
    digest = hmac.new(secret, email.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()
    return f"SS-{digest[:8].upper()}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_keys.py <email> [email2] [email3] ...")
        sys.exit(1)

    for email in sys.argv[1:]:
        key = generate_key(email)
        print(f"{email}  →  {key}")
