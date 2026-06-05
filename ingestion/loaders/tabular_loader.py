"""
Loads CSV, Excel, and Parquet files and records their metadata.
Tabular datasets are NOT pushed into Qdrant — they're stored for TABICLv2 prediction.
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TabularMeta:
    filename: str
    column_names: list[str]
    row_count: int


def load_tabular(path: str | Path) -> TabularMeta:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path, nrows=0)   # headers only for meta; full file loaded at predict time
        row_count = sum(1 for _ in open(path)) - 1
    elif suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path, nrows=0)
        full = pd.read_excel(path)
        row_count = len(full)
    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path).iloc[:0]
        import pyarrow.parquet as pq
        row_count = pq.read_metadata(path).num_rows
    else:
        raise ValueError(f"Unsupported tabular file format: {suffix}")

    return TabularMeta(
        filename=path.name,
        column_names=list(df.columns),
        row_count=row_count,
    )


TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
