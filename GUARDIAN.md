# GUARDIAN

Automated security & bug guardian ledger for `data-palette`. Branch: `claude/guardian`.

## Checks

| Area | Command |
|---|---|
| Dependency vulns | `pip-audit -r <deps from pyproject incl. torch extra>` (numpy, opencv-python, tqdm, pyyaml, scikit-learn, torch, torchvision) |
| SAST | `bandit -c pyproject.toml -r datapalette -q` (B101 skipped in `[tool.bandit]`, see G-007) |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|yaml\.load\|shell=True\|verify=False\|0\.0\.0\.0\|np\.load" datapalette` |
| Secrets (history) | `gitleaks detect --source . --redact -v` |
| Secrets (tree) | `gitleaks detect --no-git --source . --redact` ; `git ls-files \| grep -i env` |
| CI | manual review of `.github/workflows/ci.yml` (pinning, `permissions:`, untrusted `${{ github.event.* }}`) |
| Docker | none present |
| Mask/label geometry (added run #2) | `grep -rnE "cv2\.(resize\|warpAffine\|remap)" datapalette` — every path that can receive label masks must use `INTER_NEAREST` |
| Tests | `pytest --cov=datapalette` in uv venvs for 3.9/3.10/3.11/3.12 (`.[dev]`); the 3.11 venv also gets CPU `torch torchvision` so `tests/test_adapters` actually runs |
| Lint / types | `ruff check .` ; `mypy datapalette/` (as configured in CI) |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | high | bug (5 failing tests) | datapalette/transforms/base.py | fixed | 2026-10-15 |
| G-002 | low | ci-hardening: no `permissions:` block (default token scope) | .github/workflows/ci.yml | open | 2026-10-15 |
| G-003 | low | ci-hardening: actions pinned by tag, not commit SHA | .github/workflows/ci.yml | open | 2026-10-15 |
| G-004 | low | lint: 4 ruff F401 errors in tests | tests/ | fixed | 2026-10-15 |
| G-005 | low | types: mypy errors (np.uint8 list ctor) | datapalette/transforms/dataset.py | fixed | 2026-10-15 |
| G-006 | low | types: 5 remaining mypy errors (opencv-python 5 stubs overloads, `transform` override return type) | transforms/{edges,frequency,color,dataset}.py | fixed | 2026-10-15 |
| G-007 | low | bandit B101 `assert` for narrowing (4x) | datapalette/transforms/color.py | wontfix (type narrowing only; skipped in bandit config 2026-10-17) | 2026-10-15 |
| G-008 | medium | bug: SegmentationPipeline resized/rotated label masks bilinearly, inventing invalid class ids (e.g. 129 between 3 and 255) | datapalette/pipelines/segmentation.py, transforms/spatial.py | fixed | 2026-10-17 |
| G-009 | low | bug: a single 2-D grayscale image passed to `transform` was iterated as a batch of rows | transforms/base.py, spatial.py (Tile), dataset.py (DoubleCrop) | fixed | 2026-10-17 |
| G-010 | low | bug: `extract_frames` silently returned 0 for an unopenable video; `fps=0` -> ZeroDivisionError | datapalette/io/video.py | fixed | 2026-10-17 |
| G-011 | medium | bug: torch `DataPaletteDataset` transformed images through the pipeline but returned masks untouched (shape/geometry mismatch) | datapalette/adapters/torch.py | fixed | 2026-10-17 |
| G-012 | low | quality: `Rotate` border fill is 0, which is a valid class id for multiclass masks (non-90-degree angles) | transforms/spatial.py | open | 2026-10-17 |

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
| mypy errors | 7 | 0 |
| coverage | n/a (suite failing) | 71% |

### Metrics run #2 (2026-10-17)

| metric | before | after |
|---|---|---|
| open vulns crit/high/med/low | 0/0/0/6 (dep vulns 0, secrets 0) | 0/0/0/3 (G-002, G-003, G-012; bandit B101 now configured-out) |
| open bugs found this run | 4 (G-008..G-011) | 0 |
| failing tests | 0 | 0 |
| tests passed/skipped (no torch) | 79 / 5 | 88 / 6 (+10 tests; 1 new torch test) |
| tests passed/skipped (with CPU torch, 3.11) | n/a | 93 / 1 |
| ruff / mypy errors (3.9-3.12) | 0 / 0 | 0 / 0 |
| coverage (3.11 + torch) | 71% | 75% |

## Backlog

- G-002: add top-level `permissions:\n  contents: read` to ci.yml (guardian token lacks `workflow` scope, so it could not push this).
- G-003: pin `actions/checkout` and `actions/setup-python` to full commit SHAs.
- Replace the G-006 `type: ignore[call-overload]` on `cv2.normalize` once opencv stubs accept `dst=None`.
- Core deps are unpinned (`numpy`, `opencv-python`, `scikit-learn`, ...); consider lower bounds
  (e.g. `scikit-learn>=1.3`) so behaviour changes like G-001 are caught deliberately.

- G-012: let `Rotate` take a `border_value` (masks: ignore id such as 255) and use it in `SegmentationPipeline.mask_pipeline`.
- `DataPaletteDataset` returns `label=None` without masks, which breaks torch's default `collate_fn`; consider an empty tensor or documenting a custom collate.
- `SegmentationPipeline.mask_pipeline` duplicates the image steps by hand; if steps become configurable, derive it via `sklearn.base.clone` + `set_params(interpolation=INTER_NEAREST)`.
- CLI (`datapalette/cli.py`) has 13% coverage; add smoke tests for `_get_pipeline_steps` / `_process_input`.

## Learnings

- sklearn >= 1.6 enforces fittedness in `Pipeline.transform`; stateless custom transformers
  need `__sklearn_is_fitted__` (or `requires_fit=False` tags).
- The guardian push token cannot modify `.github/workflows/*`; CI hardening must be done by the owner.
- `cv2.normalize(x, np.empty_like(x), ...)` is NOT equivalent to `dst=None` for float64 input (broke FourierTransform tests); prefer a targeted `type: ignore`.
- CI runs a 3.9-3.12 matrix with different numpy/mypy versions; verify type checks on every matrix version (uv venvs) before pushing.
- No network, pickle, or subprocess surface in the package; YAML uses `safe_load`.
- The torch adapter tests are skipped without torch, so they hid G-011; installing CPU torch (`--index-url https://download.pytorch.org/whl/cpu`, ~3.5 min with uv) is worth it.
- For a segmentation toolkit the highest-yield bug class is label corruption (interpolation, border fill, image/mask misalignment) rather than crashes.

## Ideas

1. Add a CI job against the minimum supported versions (numpy/sklearn lower bounds) to catch API drift.
2. Add `pip-audit` + `gitleaks` steps to CI.
3. Add Dependabot for GitHub Actions to keep SHA pins fresh.
4. Property test: for every spatial transform, `mask_pipeline` output values are a subset of the input mask values (hypothesis-style, random masks/sizes).
5. Optional CI job that installs CPU torch so `tests/test_adapters` never silently skips.
6. Add `bandit -c pyproject.toml` to pre-commit (config now lives in pyproject).

## Log

- 2026-10-15 run #1: first run. Fixed G-001 (sklearn NotFittedError breaking 5 tests and all pipelines), G-004, G-005; G-002 fix reverted (push token lacks workflow scope); no secrets, no dependency vulns.
- 2026-10-15 run #1 follow-up: CI `test (3.12)` was red on the mypy step (G-006, pre-existing). Fixed: targeted ignores on cv2.normalize/override, saturating `cv2.add` with an array operand; mypy 5 -> 0, ruff 0, 82 passed / 2 skipped locally.
- 2026-10-15 run #1 follow-up 2: CI matrix reproduced locally with uv (3.9-3.12). Fixed a numpy-2.2 stub error (py3.10, color.py) and missing yaml stubs on py3.9 (added `types-PyYAML` to dev extra). mypy clean and 79 passed / 5 skipped on all four versions.
- 2026-10-17 run #2: PR #3 CI green, no review threads; merged origin/main (no-op). Scans clean: pip-audit 0 (incl. torch/torchvision), gitleaks 0 (21 commits + tree), bandit 0 after config. Fixed 4 bugs test-first: G-008 bilinear mask interpolation in SegmentationPipeline, G-009 grayscale single-image dispatch, G-010 extract_frames silent failure / fps=0, G-011 torch adapter image/mask misalignment. Tests 79 -> 88 passed (93 with torch), coverage 71% -> 75%, mypy/ruff clean on 3.9-3.12. META-REVIEW (4th Log entry): (a) new check: mask/label geometry grep (every resize/warp path reachable by masks must be INTER_NEAREST) - motivated by G-008/G-011; (b) cut false positive: bandit B101 skipped via `[tool.bandit]` in pyproject (only type-narrowing asserts, G-007 wontfix twice); (c) tightened config: tests now run with CPU torch in one venv so adapter tests stop skipping, and the matrix check (uv 3.9-3.12) is part of Checks. G-002/G-003 still need the owner (workflow scope).
