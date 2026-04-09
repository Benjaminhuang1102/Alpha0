"""Parquet-backed storage layer for Alpha0 data.

Provides a thin, consistent interface over pyarrow/pandas parquet I/O.
All reads and writes use pyarrow as the engine exclusively to avoid
subtle inconsistencies when mixing engines on timezone-aware DatetimeIndex.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetStore:
    """Thin wrapper around pyarrow/pandas parquet read/write.

    Organises files under a ``base_dir`` using a name-to-path mapping:
    ``name="prices/AAPL"`` → ``base_dir/prices/AAPL.parquet``.

    All DataFrames are expected to have a DatetimeIndex.  The index is
    always written to disk so it is faithfully round-tripped.

    Parameters
    ----------
    base_dir:
        Root directory for this store (e.g. ``"data/raw"``).
        Created on first write if it does not exist.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        df: pd.DataFrame,
        name: str,
        partition_cols: list[str] | None = None,
    ) -> Path:
        """Write a DataFrame to ``base_dir/<name>.parquet``.

        Parameters
        ----------
        df:
            DataFrame to persist.  Must have a DatetimeIndex.
        name:
            Logical dataset name (path relative to ``base_dir``, without
            ``.parquet`` extension).  Forward slashes create sub-directories.
        partition_cols:
            If provided, write a partitioned dataset using pyarrow's
            ``write_to_dataset``.  Otherwise a single ``.parquet`` file
            is written.

        Returns
        -------
        Path
            Absolute path to the written file or dataset directory.
        """
        dest = self._resolve(name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            table = pa.Table.from_pandas(df, preserve_index=True)
            pq.write_to_dataset(table, root_path=str(dest), partition_cols=partition_cols)
            logger.debug("Wrote partitioned dataset %s (%d rows)", dest, len(df))
        else:
            df.to_parquet(dest, engine="pyarrow", index=True)
            logger.debug("Wrote %s (%d rows, %d cols)", dest, len(df), len(df.columns))

        return dest

    def read(
        self,
        name: str,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        """Read a DataFrame from ``base_dir/<name>.parquet``.

        Parameters
        ----------
        name:
            Logical dataset name (same as used in :meth:`write`).
        columns:
            If provided, only these columns are loaded (column pruning).
        filters:
            PyArrow-style filter expressions for predicate pushdown.
            Example: ``[("date", ">=", "2020-01-01")]``.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex restored.

        Raises
        ------
        FileNotFoundError
            If the dataset does not exist at the expected path.
        """
        dest = self._resolve(name)
        if not dest.exists():
            raise FileNotFoundError(f"Dataset not found: {dest}")

        df = pd.read_parquet(dest, engine="pyarrow", columns=columns, filters=filters)
        logger.debug("Read %s (%d rows)", dest, len(df))
        return df

    def exists(self, name: str) -> bool:
        """Return True if the dataset exists on disk."""
        return self._resolve(name).exists()

    def list_datasets(self, prefix: str = "") -> list[str]:
        """List all dataset names (relative to ``base_dir``) matching ``prefix``.

        Parameters
        ----------
        prefix:
            Optional path prefix filter (e.g. ``"prices/"``).

        Returns
        -------
        list[str]
            Sorted list of logical dataset names (without ``.parquet`` extension).
        """
        search_dir = self.base_dir / prefix if prefix else self.base_dir
        if not search_dir.exists():
            return []

        results: list[str] = []
        for path in sorted(search_dir.rglob("*.parquet")):
            # Convert absolute path back to a logical name
            rel = path.relative_to(self.base_dir)
            # Strip .parquet suffix
            name = str(rel.with_suffix("")).replace("\\", "/")
            results.append(name)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> Path:
        """Convert a logical dataset name to an absolute filesystem path."""
        # Normalise separators and ensure .parquet extension
        clean = name.replace("\\", "/").strip("/")
        return self.base_dir / (clean + ".parquet")
