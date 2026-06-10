from __future__ import annotations

from ..baseline_resume import resume_population_method


def resume_meoh(meoh, path=None):
    resume_population_method(meoh, path, label="MEoH")
