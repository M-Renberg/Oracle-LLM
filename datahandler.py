import pandas as pd
import io
from typing import Optional

_df: Optional[pd.DataFrame] = None

def load_csv_to_memory(file_content: bytes):
    global _df
    _df = pd.read_csv(io.BytesIO(file_content))
    return _df

def get_dataframe() -> Optional[pd.DataFrame]:
    return _df

def get_stats() -> dict:
    if _df is None:
        raise ValueError("Inget dataset uppladdat")
    return _df.describe().to_dict()