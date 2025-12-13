from typing import Optional
import pandas as pd

class DataStore:
    def __init__(self):
        self._frames = {}

    def set_dataframe(self, sid: str, df: pd.DataFrame) -> None:
        self._frames[sid] = df.copy()

    def get_dataframe(self, sid: str) -> Optional[pd.DataFrame]:
        return self._frames.get(sid)

    def clear(self, sid: str) -> None:
        if sid in self._frames:
            del self._frames[sid]
