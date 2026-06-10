from __future__ import annotations

from ..baseline_resume import resume_population_method


def resume_moead(moead, path=None):
    resume_population_method(moead, path, label="MOEAD")
