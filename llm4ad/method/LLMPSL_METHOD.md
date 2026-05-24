# LLM-PSL: Pareto Set Learning for LLM-Driven Heuristic Design

## 1. Motivation

LLM-driven heuristic search usually evaluates each generated heuristic program as a black-box algorithm. In this repository, a generated function such as `select_neighbor` is run inside an inner multi-objective optimizer, and the evaluator returns a score such as:

```text
score(h) = (-HV(h), runtime(h))
```

The search objective is not a single best heuristic. A practical heuristic design system should return a set of heuristics with different trade-offs:

```text
high quality but slower
balanced quality and speed
fast but lower quality
structurally novel code that may open a new search direction
```

LLM-PSL adapts the idea of Pareto Set Learning (PSL) to heuristic program search. Instead of learning a continuous neural map from a preference vector to a solution, LLM-PSL maintains a preference-indexed memory of heuristic programs and uses LLM operators to synthesize new programs for under-covered trade-off regions.

The current implementation further uses the "Few for Many" idea: a memory cell does not store a single best heuristic, but a small representative set of heuristics that collaboratively covers the objectives. This replaces standard single-solution Tchebycheff scalarization with Smooth Tchebycheff Set scalarization.

Reference paper:

- Pareto Set Learning for Expensive Multi-Objective Optimization, NeurIPS 2022: https://papers.nips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract-Conference.html
- Paper PDF: https://papers.nips.cc/paper_files/paper/2022/file/7a583691ccfcf8945ab714b677ccbf0b-Paper-Conference.pdf
- Smooth Tchebycheff Scalarization for Multi-Objective Optimization, ICML 2024: https://proceedings.mlr.press/v235/lin24y.html
- OpenReview page: https://openreview.net/forum?id=m4dO5L6eCp
- Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization, ICLR 2025: https://openreview.net/forum?id=O4N9kWwV6R
- Paper PDF: https://openreview.net/pdf?id=O4N9kWwV6R

## 2. Connection To Original PSL

Original PSL considers expensive multi-objective optimization:

```text
minimize F(x) = (f_1(x), ..., f_m(x))
```

It learns a Pareto set model:

```text
x_lambda = h_theta(lambda)
```

where `lambda` is a trade-off preference vector on the simplex:

```text
lambda_i >= 0
sum_i lambda_i = 1
```

The model maps each preference vector to a candidate Pareto solution.

For LLM-driven heuristic search:

```text
solution x              -> heuristic program h
objective vector F(x)   -> heuristic score F(h)
h_theta(lambda)         -> LLM-guided generator conditioned on lambda and Pareto memory
Pareto set model        -> preference-indexed heuristic set memory M[lambda]
expensive evaluation    -> running the real evaluator to measure HV and runtime
```

Thus LLM-PSL learns a discrete program-level approximation:

```text
lambda -> heuristic program
```

This is different from only adding `lambda` to the prompt. The method changes the search state, selection policy, archive update, and generation operators.

## 3. Objective Formulation

LLM-PSL uses three minimization objectives:

```text
F(h) = (q(h), t(h), d(h))
```

where:

```text
q(h) = -HV(h)
t(h) = runtime(h)
d(h) = -Novelty_CodeBLEU(h)
```

So the final score stored in the population is:

```text
score(h) = (-HV(h), runtime(h), -Novelty_CodeBLEU(h))
```

All objectives are minimized:

```text
lower -HV       -> higher hypervolume
lower runtime   -> faster heuristic
lower -novelty  -> more novel code
```

Important distinction:

```text
HV  = absolute hypervolume obtained by the heuristic after evaluation
HVI = hypervolume improvement relative to the current archive
```

`HV` is stable enough to store as an objective. `HVI` depends on the current archive, so it should be used as an acquisition or tie-breaker, not as the permanent objective value in the population.

## 4. Evaluating -HV

For each benchmark instance:

1. Initialize an archive of feasible solutions.
2. Repeatedly call the generated heuristic `h` to propose a neighbor solution.
3. Compute the solution objective vector.
4. Update the inner non-dominated archive.
5. Compute hypervolume of the final archive.
6. Return `-HV` because the outer method minimizes scores.

For minimization tasks such as bi-objective TSP:

```python
objs = np.array([obj for _, obj in Archive])
hv_value = HV(ref_point=ref_point)(objs)
quality_loss = -hv_value
```

For maximization tasks such as bi-objective KP:

```python
objs = np.array([obj for _, obj in Archive]) * (-1)
hv_value = HV(ref_point=ref_point)(objs)
quality_loss = -hv_value
```

The multiplication by `-1` converts maximization objectives into minimization objectives before computing HV.

The reference point must be worse than all expected points in the minimization objective space.

## 5. CodeBLEU Novelty

CodeBLEU is used to measure code-level similarity. Since CodeBLEU can be asymmetric when one code is treated as the reference, LLM-PSL uses a symmetrized score:

```text
sim(a, b) = 0.5 * (CodeBLEU(a, b) + CodeBLEU(b, a))
dist(a, b) = 1 - sim(a, b)
```

The novelty of a heuristic program is computed against the current archive or population:

```text
Novelty(h) = mean distance to k nearest programs
```

That is:

```text
Novelty(h) = mean_k_smallest { 1 - sim(h, g) | g in archive }
```

For small populations, `k = 1` or `k = 2` is enough. If the archive is empty, novelty can be initialized to `1.0`.

The population objective is:

```text
d(h) = -Novelty(h)
```

This encourages the method to keep structurally different programs. However, novelty should not be allowed to dominate the search. A practical rule is:

```text
use novelty as the third Pareto objective
but reject candidates that are clearly invalid or extremely poor in HV/runtime
```

## 6. Preference Vector

LLM-PSL uses a 3-dimensional preference vector:

```text
lambda = (lambda_q, lambda_t, lambda_n)
```

where:

```text
lambda_q: preference weight for quality, objective -HV
lambda_t: preference weight for runtime
lambda_n: preference weight for code novelty, objective -CodeBLEU novelty
```

Examples:

```text
(0.80, 0.10, 0.10) -> quality-heavy heuristic
(0.45, 0.45, 0.10) -> balanced quality-speed heuristic
(0.20, 0.70, 0.10) -> speed-heavy heuristic
(0.35, 0.25, 0.40) -> novelty-seeking heuristic
```

The preference vectors form a finite grid:

```text
Lambda = {lambda_1, ..., lambda_K}
```

LLM-PSL maintains a small representative heuristic set for each preference vector.

## 7. Preference-Indexed Pareto Set Memory

LLM-PSL maintains two archives:

```text
A: global Pareto archive
M: preference-indexed set memory
```

The global archive stores non-dominated heuristic programs under the 3-objective score:

```text
(-HV, runtime, -Novelty_CodeBLEU)
```

The memory stores one small representative set per preference:

```text
M[lambda_k] = {h_1, ..., h_K}, a small set optimized for lambda_k
```

This turns the finite population into a discrete Pareto set model:

```text
lambda_k -> representative heuristic set M[lambda_k]
```

This is the main conceptual difference from LLM-PFG. LLM-PFG keeps a Pareto population. LLM-PSL additionally learns a preference-indexed mapping from trade-offs to small complementary heuristic sets.

## 8. Smooth Tchebycheff Set Scalarization For Memory Update

To compare a small heuristic set under a target preference, LLM-PSL normalizes each objective:

```text
s_i(h) = (f_i(h) - ideal_i) / (nadir_i - ideal_i + eps)
```

where:

```text
ideal_i = best observed value for objective i
nadir_i = worst observed value for objective i
eps     = small constant to avoid division by zero
```

Instead of standard Tchebycheff scalarization for one heuristic:

```text
g(h | lambda) =
    max_i lambda_i * s_i(h)
    + rho * sum_i lambda_i * s_i(h)
```

LLM-PSL uses Smooth Tchebycheff Set scalarization by default. The non-smooth set form is:

```text
g_TCH-Set(H_K | lambda) =
    max_i lambda_i * min_{h in H_K} s_i(h)
    + rho * sum_i lambda_i * min_{h in H_K} s_i(h)
```

where `H_K = {h_1, ..., h_K}` is a small set of heuristic programs. The inner `min` means objective `i` only needs to be well handled by at least one heuristic in the set. This follows the "Few for Many" view: a few complementary solutions can cover many objectives better than forcing one solution to compromise across all objectives.

The smooth form replaces `min` and `max` with log-sum-exp soft-min and soft-max:

```text
softmin_i(H_K) = -mu * log sum_{h in H_K} exp(-s_i(h) / mu)

g_STCH-Set(H_K | lambda) =
    mu * log sum_i exp(lambda_i * softmin_i(H_K) / mu)
    + rho * sum_i lambda_i * softmin_i(H_K)
```

The memory update becomes:

```text
M[lambda_k] = argmin_{H_K subset A} g_STCH-Set(H_K | lambda_k)
```

The implementation uses greedy forward selection to approximate this subset because exhaustive subset search is unnecessary for small populations and expensive for larger archives.

The code exposes this through `smooth_set_scalarization` and `smooth_mu`. In the current implementation, `smooth_set_scalarization` is enabled by default. Since heuristic code search is not differentiable, the smooth form is used as a stable ranking and selection criterion rather than as a gradient optimizer.

## 9. Adaptive Lambda Scheduler

At each iteration, LLM-PSL chooses a target preference region:

```text
lambda_t = SelectPreference(M, A)
```

The scheduler prioritizes regions that are:

```text
uncovered: no representative set exists
stale: representative set has not improved for many iterations
weak: STCH-Set scalarization score is poor compared with neighboring regions
sparse: set members are too similar in objective or CodeBLEU space
```

A practical priority score is:

```text
priority(lambda_k) =
    uncovered_bonus(lambda_k)
  + stale_bonus(lambda_k)
  + set_scalarization_regret(lambda_k)
  + diversity_gap(lambda_k)
```

Then sample `lambda_t` proportionally to this priority. This allocates evaluation budget to under-developed parts of the Pareto set.

## 10. Anchor Selection

After choosing `lambda_t`, LLM-PSL selects anchors from the representative set stored in memory and from the global archive.

Recommended anchor types:

```text
anchor_set: complementary heuristics from M[lambda_t]
anchor_quality: member or neighbor set representative with stronger -HV
anchor_speed: member or neighbor set representative with lower runtime
anchor_novel: member or neighbor set representative with higher CodeBLEU novelty
```

If a memory cell is empty, greedily build a small anchor set from the global archive by minimizing `g_STCH-Set(H_K | lambda_t)`.

Parent selection should also enforce code diversity:

```text
avoid choosing anchors with CodeBLEU similarity above a threshold
unless one anchor clearly dominates the other in objective space
```

## 11. LLM Pareto Operators

LLM-PSL uses LLM operators that are tied to the Pareto set structure.

### 11.1 Pareto Interpolation

Input:

```text
h_a at lambda_a
h_b at lambda_b
target lambda_t between lambda_a and lambda_b
```

Output:

```text
h_t = LLM_Interpolate(h_a, h_b, lambda_t)
```

The prompt asks the LLM to preserve the mechanism responsible for quality from one anchor, preserve the mechanism responsible for speed or novelty from another anchor, and synthesize a new heuristic for the target preference.

### 11.2 Pareto Extrapolation

Input:

```text
h_a at lambda_a
target lambda_t more extreme than lambda_a
```

Output:

```text
h_t = LLM_Extrapolate(h_a, lambda_t)
```

Examples:

```text
move a balanced heuristic toward higher quality
move a balanced heuristic toward lower runtime
move a common heuristic toward higher CodeBLEU novelty
```

### 11.3 Novelty Repair

If a candidate is too similar to existing programs:

```text
max CodeBLEU similarity > tau
```

the LLM is asked to rewrite the algorithm using a different control flow or search mechanism while preserving the target preference.

This is not just paraphrasing code. The prompt should require algorithmic changes such as:

```text
change archive sampling policy
change move operator
change repair mechanism
change scoring formula
change deterministic/randomized balance
```

## 12. Candidate Filtering

Before expensive evaluation, apply cheap checks:

```text
syntax check
template signature check
forbidden import check
duplicate string check
CodeBLEU near-duplicate check
simple infinite-loop risk check
```

Near-duplicate rule:

```text
if max_sim(candidate, archive) > tau:
    reject or send to novelty repair
```

Suggested threshold:

```text
tau = 0.85 to 0.90
```

Exception:

```text
if the candidate dominates the nearest similar program after true evaluation,
keep the candidate and remove the dominated one
```

## 13. Population Update

Each evaluated program receives:

```text
score(h) = (-HV(h), runtime(h), -Novelty_CodeBLEU(h))
```

The global archive is updated by 3-objective Pareto dominance.

If the population exceeds the target size:

1. Apply non-dominated sorting.
2. Keep full fronts while possible.
3. If the last front overflows, select by a combination of:

```text
objective crowding distance
CodeBLEU diversity to already selected programs
preference memory coverage
```

One practical score for tie-breaking:

```text
tie_score(h) =
    alpha * normalized_objective_crowding(h)
  + beta  * normalized_code_diversity(h)
  + gamma * preference_coverage_bonus(h)
```

The selected individuals should preserve both objective diversity and code-space diversity.

## 14. Full Algorithm

```text
Input:
  task description
  template function
  preference grid Lambda
  max evaluations T
  population size N

Initialize:
  A = empty global archive
  M[lambda_k] = empty for each lambda_k in Lambda

Initial sampling:
  generate initial heuristic programs with diverse preferences
  evaluate each program by true evaluator
  compute CodeBLEU novelty
  update A and M

For t = 1 ... T:

  1. Select target preference
       lambda_t = adaptive scheduler over Lambda

  2. Select anchors
       anchors = preference-aware, CodeBLEU-diverse heuristics from M and A

  3. Generate candidate
       h_new = LLM Pareto interpolation/extrapolation/repair operator

  4. Cheap validation
       reject syntax-invalid, unsafe, duplicate, or near-duplicate code

  5. True evaluation
       q = -HV(h_new)
       r = runtime(h_new)
       n = CodeBLEU novelty of h_new
       score(h_new) = (q, r, -n)

  6. Update archives
       insert h_new into A
       prune A by Pareto dominance and diversity-aware truncation
       update M[lambda_k] = argmin_h g(h | lambda_k)

Return:
  global Pareto archive A
  preference-indexed heuristic set M
```

## 15. Difference From LLM-PFG

LLM-PFG:

```text
maintains one Pareto population
uses Pareto front grid parent selection
generates heuristics through generic EoH-style operators
does not store a mapping from preferences to programs
does not explicitly preserve code diversity
```

LLM-PSL:

```text
maintains global archive plus preference-indexed memory
uses Smooth Tchebycheff Set scalarization for each preference
allocates budget to under-covered preference regions
uses LLM Pareto interpolation and extrapolation operators
uses 3 objectives: -HV, runtime, -CodeBLEU novelty
preserves diversity in both objective space and code space
```

The novelty is not the prompt text alone. The novelty is the search mechanism:

```text
preference-indexed Pareto set memory
adaptive lambda scheduling
Smooth Tchebycheff Set-based memory update
LLM-based Pareto interpolation/extrapolation in program space
CodeBLEU novelty as a third objective
```

## 16. Practical Implementation Plan For This Repository

### Step 1: Extend score to 3 objectives

Current score:

```text
(-HV, runtime)
```

New score:

```text
(-HV, runtime, -CodeBLEU_novelty)
```

The novelty value should be computed after evaluation and before registering the function into the population.

### Step 2: Add CodeBLEU novelty utilities

Add helper functions:

```text
code_similarity(func_a, func_b)
code_novelty(func, archive, k)
```

If CodeBLEU dependency is unavailable, use a fallback such as token n-gram similarity or AST node similarity, but report it as fallback rather than true CodeBLEU.

### Step 3: Add preference memory

Extend population management with:

```text
preference_grid
memory_by_preference
normalize_scores()
smooth_tchebycheff_scalarization()
smooth_tchebycheff_set_scalarization()
tchebycheff_set_scalarization()
greedy_tchebycheff_set_selection()
update_preference_memory()
```

### Step 4: Add adaptive lambda scheduler

Choose `lambda_t` based on:

```text
empty memory cells
staleness
poor STCH-Set scalarization score
low CodeBLEU/objective diversity near the cell
```

### Step 5: Modify prompt operators

Add prompt templates for:

```text
Pareto interpolation
Pareto extrapolation
novelty repair
```

Each prompt should include:

```text
target lambda
objective meanings
anchor set scores
anchor algorithms and code
explicit requirement for CodeBLEU-level algorithmic difference
```

### Step 6: Keep true evaluation as final authority

The LLM can generate and revise programs, but final fitness must always come from the evaluator:

```text
true HV
true runtime
computed CodeBLEU novelty
```

Do not let LLM-estimated quality replace real evaluation.

## 17. Suggested Claim

```text
LLM-PSL extends Pareto Set Learning to programmatic heuristic design by replacing the continuous Pareto set model with a preference-indexed memory of small complementary heuristic sets and LLM-based Pareto interpolation/extrapolation operators. Inspired by Smooth Tchebycheff and Tchebycheff Set scalarization, it learns a discrete mapping from trade-off preferences to representative heuristic sets, while preserving diversity in both objective space and code space through a three-objective formulation over hypervolume, runtime, and CodeBLEU novelty.
```
