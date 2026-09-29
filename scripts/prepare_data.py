"""Valida uma fonte de churn e produz a cópia canônica local do projeto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_contract import dataset_metadata, validate_churn_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza e valida uma base de churn telecom.")
    parser.add_argument("input", type=Path, help="CSV bruto obtido da fonte documentada.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/churn.csv"),
        help="Destino local para o CSV canônico.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    prepared = validate_churn_dataframe(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)

    metadata_path = args.output.with_name("dataset_metadata.json")
    metadata_path.write_text(json.dumps(dataset_metadata(prepared), indent=2), encoding="utf-8")
    print(f"Dataset validado: {len(prepared)} linhas em {args.output}")
    print(f"Metadados: {metadata_path}")


if __name__ == "__main__":
    main()
