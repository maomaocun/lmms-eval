#!/usr/bin/env python3
"""Standalone worker for canonicalizing SMILES in an isolated subprocess."""
import json
import sys
import os

# Allow importing the package when run as a standalone script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smiles_canonicalization import canonicalize_molecule_smiles


def main():
    data = json.load(sys.stdin)
    items = data.get("items", [])
    return_none = data.get("return_none_for_error", True)
    fallback = data.get("fallback_to_original", False)

    results = []
    for item in items:
        try:
            if item == "" or item is None:
                results.append(None if return_none else (item or ""))
            else:
                results.append(
                    canonicalize_molecule_smiles(
                        item,
                        return_none_for_error=return_none,
                    )
                )
        except Exception:
            results.append(None if return_none else (item if fallback else ""))

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
