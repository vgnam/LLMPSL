from __future__ import annotations

from ..baseline_resume import resume_population_method


def resume_reevo(reevo, path=None):
    resume_population_method(reevo, path, label="ReEvo")
