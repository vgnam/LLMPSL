from __future__ import annotations

from ..baseline_resume import resume_population_method


def resume_eoh(eoh, path=None):
    resume_population_method(eoh, path, label="EoH")
