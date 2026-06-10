from __future__ import annotations

from ..baseline_resume import resume_population_method


def resume_nsga2(nsga2, path=None):
    resume_population_method(nsga2, path, label="NSGA2")
