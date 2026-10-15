# GUARDIAN

Automated security & bug guardian ledger for `data-palette`. Branch: `claude/guardian`.

## Checks

| Area | Command |
|---|---|
| Dependency vulns | `pip-audit -r <deps from pyproject incl. torch extra>` (numpy, opencv-python, tqdm, pyyaml, scikit-learn, torch, torchvision) |
| SAST | `bandit -r datapalette -q` |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|yaml\.load\|shell=True\|verify=False\|0\.0\.0\.0\|np\.load" datapalette` |
| Secrets (history) | `gitleaks detect --source . --redact -v` |
| Secrets (tree) | `gitleaks detect --no-git --source . --redact` ; `git ls-files \| grep -i env` |
| CI | manual review of `.github/workflows/ci.yml` (pinning, `permissions:`, untrusted `${{ github.event.* }}`) |
| Docker | none present |
| Tests | `pytest --cov=datapalette` (venv with `pip install -e ".[dev]"`, Python 3.11) |
| Lint / types | `ruff check .` ; `mypy datapalette/` (as configured in CI) |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug (5 failing tests) | datapalette/transforms/base.py | fixed | 2026-10-15 |
| G-002 | low | ci-hardening: no `permissions:` block (default token scope) | .github/workflows/ci.yml | open | 2026-10-15 |
| G-003 | low | ci-hardening: actions pinned by tag, not commit SHA | .github/workflows/ci.yml | open | 2026-10-15 |
| G-004 | low | lint: 4 ruff F401 errors in tests | tests/ | fixed | 2026-10-15 |
| G-005 | low | types: mypy errors (np.uint8 list ctor) | datapalette/transforms/dataset.py | fixed | 2026-10-15 |
| G-006 | low | types: 5 remaining mypy errors (opencv-python 5 stubs overloads, `transform` override return type) | transforms/{edges,frequency,color,dataset}.py | open | 2026-10-15 |
| G-007 | low | bandit B101 `assert` for narrowing (4x) | datapalette/transforms/color.py | wontfix (type narrowing only) | 2026-10-15 |

G-001 detail: with scikit-learn >= 1.6, `Pipeline.transform` calls `check_is_fitted`
on the last step; stateless `ImageTransform`s had no fitted attributes, so every unfitted
pipeline (including the library's own Diffusion/GAN/Segmentation pipelines) raised
`NotFittedError`. Fix: `ImageTransform.__sklearn_is_fitted__` returns True (transforms are
stateless or self-fit on first use). Proven by the 5 previously failing tests plus new
`tests/test_transforms/test_spatial.py::TestSklearnPipelineComposition::test_stateless_transforms_report_fitted`.

Secrets: gitleaks found nothing in history (18 commits) or the working tree. Tracked env files
are `.env.sample` (keys only, placeholder paths) and `.envrc` (`dotenv` only) — no secrets.

## Metrics

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low | 0/0/0/7 (dep vulns: 0; bandit low: 4; CI low: 2; mypy-stub/type low counted separately; secrets: 0) | 0/0/0/6 |
| failing tests | 5 | 0 |
| tests passed/skipped | 73 / 5 | 79 / 5 (+1 new test) |
| ruff errors | 4 | 0 |
| mypy errors | 7 | 5 |
| coverage | n/a (suite failing) | 71% |

## Backlog

- G-002: add top-level `permissions:\n  contents: read` to ci.yml (guardian token lacks `workflow` scope, so it could not push this).
- G-003: pin `actions/checkout` and `actions/setup-python` to full commit SHAs.
- G-006: resolve opencv-python 5 stub overload errors (pass `dst` arrays / casts) and the
  `TextLabel.transform`-style override return type in `transforms/dataset.py`.
- Core deps are unpinned (`numpy`, `opencv-python`, `scikit-learn`, ...); consider lower bounds
  (e.g. `scikit-learn>=1.3`) so behaviour changes like G-001 are caught deliberately.

## Learnings

- sklearn >= 1.6 enforces fittedness in `Pipeline.transform`; stateless custom transformers
  need `__sklearn_is_fitted__` (or `requires_fit=False` tags).
- The guardian push token cannot modify `.github/workflows/*`; CI hardening must be done by the owner.
- No network, pickle, or subprocess surface in the package; YAML uses `safe_load`.

## Ideas

1. Add a CI job against the minimum supported versions (numpy/sklearn lower bounds) to catch API drift.
2. Add `pip-audit` + `gitleaks` steps to CI.
3. Add Dependabot for GitHub Actions to keep SHA pins fresh.

## Log

- 2026-10-15 run #1: first run. Fixed G-001 (sklearn NotFittedError breaking 5 tests and all pipelines), G-004, G-005; G-002 fix reverted (push token lacks workflow scope); no secrets, no dependency vulns.
