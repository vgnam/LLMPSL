from __future__ import annotations

from ..LLMPFG.profiler import EoHProfiler as _BaseEoHProfiler


class EoHProfiler(_BaseEoHProfiler):
    def __init__(
        self,
        log_dir: str | None = None,
        evaluation_name="Problem",
        method_name="LLMPSL",
        *,
        initial_num_samples=0,
        log_style="complex",
        **kwargs,
    ):
        super().__init__(
            log_dir=log_dir,
            evaluation_name=evaluation_name,
            method_name=method_name,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            **kwargs,
        )
