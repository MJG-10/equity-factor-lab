"""Notebook pipeline helpers."""

from contextlib import redirect_stderr, redirect_stdout
import io
import warnings

from ..runner.pipeline import PipelineSettings, run_pipeline_core


def run_pipeline_core_quiet(settings: PipelineSettings):
    """Run the notebook-facing pipeline quietly with SimFin chatter suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*date_parser.*deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Missing SimFin sector classification.*",
            category=UserWarning,
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_pipeline_core(settings=settings)
