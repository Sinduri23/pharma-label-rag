"""Download public drug labels from openFDA and normalise them into named sections.

Only public, non-sensitive data is used (https://open.fda.gov/apis/drug/label/).
"""
from __future__ import annotations

import argparse
import json
import time

import requests

from . import config

OPENFDA_URL = "https://api.fda.gov/drug/label.json"

SECTION_FIELDS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "overdosage",
    "description",
    "clinical_pharmacology",
    "mechanism_of_action",
]

DEFAULT_DRUGS = [
    "metformin", "atorvastatin", "lisinopril", "amlodipine", "metoprolol", "omeprazole",
    "simvastatin", "losartan", "gabapentin", "hydrochlorothiazide", "sertraline", "furosemide",
    "levothyroxine", "warfarin", "ibuprofen", "naproxen", "amoxicillin", "azithromycin",
    "ciprofloxacin", "prednisone", "tramadol", "clopidogrel", "rosuvastatin", "escitalopram",
    "montelukast", "pantoprazole", "fluoxetine", "apixaban", "rivaroxaban", "empagliflozin",
    "sitagliptin", "denosumab", "adalimumab", "etanercept", "evolocumab", "filgrastim",
    "pembrolizumab", "nivolumab", "semaglutide", "dapagliflozin",
]


def _first(value) -> str:
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def normalize(raw: dict, drug_query: str) -> dict:
    """Turn one raw openFDA record into {set_id, drug, brand, sections{name: text}}."""
    openfda = raw.get("openfda", {})
    sections = {}
    for field in SECTION_FIELDS:
        value = raw.get(field)
        if value:
            sections[field] = " ".join(value) if isinstance(value, list) else str(value)
    return {
        "set_id": raw.get("set_id") or raw.get("id", ""),
        "drug": drug_query.lower().strip(),
        "generic_name_full": _first(openfda.get("generic_name")),
        "brand": _first(openfda.get("brand_name")),
        "effective_time": raw.get("effective_time", ""),
        "sections": sections,
    }


def fetch_label(drug: str, session=None) -> dict | None:
    http = session or requests
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 1}
    resp = http.get(OPENFDA_URL, params=params, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return normalize(results[0], drug) if results else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drugs", default=",".join(DEFAULT_DRUGS), help="comma-separated generic names")
    parser.add_argument("--out", default=str(config.RAW_LABELS_PATH))
    args = parser.parse_args()

    labels = []
    for drug in [d.strip() for d in args.drugs.split(",") if d.strip()]:
        label = fetch_label(drug)
        if label and label["sections"]:
            labels.append(label)
            print(f"ok   {drug}: {len(label['sections'])} sections ({label['generic_name_full']})")
        else:
            print(f"skip {drug}: no label found")
        time.sleep(0.3)  # be polite to the public API

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(labels, f, indent=1)
    print(f"saved {len(labels)} labels -> {args.out}")


if __name__ == "__main__":
    main()
