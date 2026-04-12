"""Audit trail for experience study runs.

Provides:
* :class:`StudyRun` — immutable record of a single study execution.
* :func:`audit_run` — decorator that wraps a study's ``.run()`` method to
  automatically capture metadata, timing, and a data checksum.
* :func:`compare_runs` — diff two :class:`StudyRun` instances.

Checksum design (Checkpoint S20)
---------------------------------
``data_checksum`` is the MD5 hex-digest of the *sorted*, newline-delimited
JSON representation of every :class:`~lifexp.core.data_model.PolicyRecord`
in the input :class:`~lifexp.core.data_model.PolicyDataset`.  Records are
sorted by ``policy_id`` before serialisation so the digest is independent
of iteration order and identical across machines with the same input data.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from typing import Any, Callable, Dict, Optional

import lifexp


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _default_serialiser(obj: Any) -> Any:
    """Extended JSON serialiser for dates and other non-standard types."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serialisable")


def _to_json_str(obj: Any) -> str:
    return json.dumps(obj, default=_default_serialiser, sort_keys=True)


# ---------------------------------------------------------------------------
# Data checksum
# ---------------------------------------------------------------------------

def checksum_dataset(dataset) -> str:
    """Return a deterministic MD5 hex-digest of *dataset*.

    Works with :class:`~lifexp.core.data_model.PolicyDataset` (the primary
    input type) and plain :class:`pandas.DataFrame` objects.  For any other
    type the ``repr()`` is hashed.

    The digest is deterministic across machines: records are sorted by
    ``policy_id`` (or index) before serialisation, and the JSON encoder uses
    ``sort_keys=True``.

    Parameters
    ----------
    dataset :
        Input data to fingerprint.

    Returns
    -------
    str
        32-character lowercase hex MD5 digest.
    """
    try:
        # PolicyDataset
        records = sorted(
            [asdict(r) for r in dataset._records],
            key=lambda d: str(d.get("policy_id", "")),
        )
    except AttributeError:
        try:
            # pandas DataFrame — sort by index for determinism
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                records = json.loads(
                    dataset.sort_index().to_json(orient="records",
                                                 date_format="iso")
                )
            else:
                records = repr(dataset)
        except Exception:
            records = repr(dataset)

    serialised = _to_json_str(records).encode("utf-8")
    return hashlib.md5(serialised).hexdigest()


# ---------------------------------------------------------------------------
# StudyRun dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudyRun:
    """Immutable audit record for one study execution.

    Attributes
    ----------
    run_id :
        UUID4 string; unique per execution.
    timestamp :
        UTC datetime when the run was created.
    lifexp_version :
        ``lifexp.__version__`` at execution time.
    python_version :
        ``sys.version`` string.
    study_type :
        Short label such as ``'mortality'``, ``'lapse'``, ``'reinsurance'``.
    parameters :
        All constructor keyword arguments passed to the study.
    data_checksum :
        MD5 hex-digest of the input dataset (see module docstring).
    execution_time_s :
        Wall-clock seconds consumed by ``.run()``.
    output_stats :
        Summary statistics extracted from the results object (optional).
    """

    run_id:           str
    timestamp:        datetime
    lifexp_version:   str
    python_version:   str
    study_type:       str
    parameters:       Dict[str, Any]
    data_checksum:    str
    execution_time_s: float = 0.0
    output_stats:     Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        d = {
            "run_id":           self.run_id,
            "timestamp":        self.timestamp.isoformat(),
            "lifexp_version":   self.lifexp_version,
            "python_version":   self.python_version,
            "study_type":       self.study_type,
            "parameters":       self.parameters,
            "data_checksum":    self.data_checksum,
            "execution_time_s": self.execution_time_s,
            "output_stats":     self.output_stats,
        }
        return d

    def to_json(self, path: str) -> None:
        """Write the audit record as a JSON file to *path*.

        Creates parent directories automatically.

        Parameters
        ----------
        path :
            Destination file path (e.g. ``'audit/run_abc123.json'``).
        """
        import os
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_default_serialiser)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyRun":
        """Reconstruct a :class:`StudyRun` from a dict (e.g. loaded from JSON).

        Parameters
        ----------
        d :
            Dict as produced by :meth:`to_dict`.
        """
        return cls(
            run_id=d["run_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            lifexp_version=d["lifexp_version"],
            python_version=d["python_version"],
            study_type=d["study_type"],
            parameters=d["parameters"],
            data_checksum=d["data_checksum"],
            execution_time_s=float(d.get("execution_time_s", 0.0)),
            output_stats=d.get("output_stats", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "StudyRun":
        """Load a :class:`StudyRun` from a JSON file.

        Parameters
        ----------
        path :
            File written by :meth:`to_json`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Output stats extractor
# ---------------------------------------------------------------------------

def _extract_output_stats(results: Any) -> Dict[str, Any]:
    """Pull a small set of scalar summary stats from a results object."""
    stats: Dict[str, Any] = {}
    for attr in (
        "overall_ae", "total_deaths", "total_etr",
        "overall_lapse_rate", "total_claims",
    ):
        val = getattr(results, attr, None)
        if val is not None:
            try:
                stats[attr] = float(val)
            except (TypeError, ValueError):
                stats[attr] = str(val)
    # Try summary_df shape
    summary = getattr(results, "summary_df", None)
    if summary is not None and hasattr(summary, "shape"):
        stats["summary_rows"] = int(summary.shape[0])
    return stats


# ---------------------------------------------------------------------------
# @audit_run decorator
# ---------------------------------------------------------------------------

def audit_run(
    study_type: Optional[str] = None,
    dataset_attr: str = "_dataset",
) -> Callable:
    """Decorator factory that wraps a study class's ``.run()`` method.

    Usage::

        @audit_run(study_type="mortality")
        class MortalityStudy:
            def run(self) -> MortalityResults: ...

    The decorator intercepts the constructor to capture parameters, and
    wraps ``.run()`` to:

    1. Compute the MD5 checksum of the input dataset.
    2. Time the execution.
    3. Extract summary stats from the returned results.
    4. Attach a :class:`StudyRun` instance as ``results.audit_run``.

    Parameters
    ----------
    study_type :
        Short label for the study type.  Defaults to the lower-cased class
        name with ``"Study"`` stripped (e.g. ``MortalityStudy`` → ``'mortality'``).
    dataset_attr :
        Name of the instance attribute holding the primary input dataset
        (default: ``'_dataset'``).

    Returns
    -------
    class decorator
    """
    def decorator(cls):
        _stype = study_type or cls.__name__.replace("Study", "").lower()
        original_init = cls.__init__
        original_run  = cls.run

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Capture constructor parameters from the call signature
            try:
                sig    = inspect.signature(original_init)
                params = list(sig.parameters.keys())[1:]  # skip 'self'
                bound  = dict(zip(params, args))
                bound.update(kwargs)
                self._audit_params: Dict[str, Any] = {
                    k: _serialise_param(v) for k, v in bound.items()
                }
            except Exception:
                self._audit_params = {}

        @functools.wraps(original_run)
        def new_run(self):
            # Checksum the primary dataset
            dataset = getattr(self, dataset_attr, None)
            if dataset is None:
                # Fallback: try common attribute names
                for attr in ("_policy_data", "_data", "_commission_data"):
                    dataset = getattr(self, attr, None)
                    if dataset is not None:
                        break
            checksum = checksum_dataset(dataset) if dataset is not None else ""

            t0 = time.perf_counter()
            results = original_run(self)
            elapsed = time.perf_counter() - t0

            run = StudyRun(
                run_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                lifexp_version=lifexp.__version__,
                python_version=sys.version,
                study_type=_stype,
                parameters=getattr(self, "_audit_params", {}),
                data_checksum=checksum,
                execution_time_s=round(elapsed, 6),
                output_stats=_extract_output_stats(results),
            )
            results.audit_run = run
            return results

        cls.__init__ = new_init
        cls.run      = new_run
        return cls

    return decorator


def _serialise_param(v: Any) -> Any:
    """Convert a constructor parameter to a JSON-safe value."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_serialise_param(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _serialise_param(val) for k, val in v.items()}
    # For complex objects (datasets, tables) store the type name
    return type(v).__name__


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

def compare_runs(run_a: StudyRun, run_b: StudyRun) -> Dict[str, Any]:
    """Compare two :class:`StudyRun` instances and report what changed.

    Parameters
    ----------
    run_a, run_b :
        The two audit records to compare.  Order matters for the ``'changed'``
        direction: values are shown as ``{'from': a_val, 'to': b_val}``.

    Returns
    -------
    dict with keys:

    ``'same_data'``
        ``True`` if both checksums are identical.
    ``'data_checksum'``
        Sub-dict ``{'run_a': ..., 'run_b': ...}`` when checksums differ;
        omitted when identical.
    ``'same_version'``
        ``True`` if ``lifexp_version`` matches.
    ``'version_diff'``
        Sub-dict when versions differ.
    ``'parameter_diffs'``
        Dict of parameter name → ``{'from': ..., 'to': ...}`` for every
        parameter that was added, removed, or changed between the two runs.
    ``'execution_time_delta_s'``
        Signed difference ``run_b.execution_time_s − run_a.execution_time_s``.
    """
    result: Dict[str, Any] = {}

    # --- checksum ---
    same_data = run_a.data_checksum == run_b.data_checksum
    result["same_data"] = same_data
    if not same_data:
        result["data_checksum"] = {
            "run_a": run_a.data_checksum,
            "run_b": run_b.data_checksum,
        }

    # --- version ---
    same_version = run_a.lifexp_version == run_b.lifexp_version
    result["same_version"] = same_version
    if not same_version:
        result["version_diff"] = {
            "run_a": run_a.lifexp_version,
            "run_b": run_b.lifexp_version,
        }

    # --- parameter diffs ---
    keys_a = set(run_a.parameters)
    keys_b = set(run_b.parameters)
    all_keys = keys_a | keys_b
    param_diffs: Dict[str, Any] = {}
    for k in sorted(all_keys):
        va = run_a.parameters.get(k, _MISSING)
        vb = run_b.parameters.get(k, _MISSING)
        if va != vb:
            param_diffs[k] = {
                "from": None if va is _MISSING else va,
                "to":   None if vb is _MISSING else vb,
            }
    result["parameter_diffs"] = param_diffs

    # --- timing ---
    result["execution_time_delta_s"] = round(
        run_b.execution_time_s - run_a.execution_time_s, 6
    )

    return result


class _MissingSentinel:
    """Sentinel for absent parameter keys."""
    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()
