# Operations — live Space, Hugging Face, and DGX Spark

Runbook for deploying and maintaining the **live** Gradio Space and for working on this repository from the
**DGX Spark** control host (`spark-1240`). Written from a Cursor Cloud Agent session on that machine (2026-11-09).

## Repositories and artifacts

| Artifact | Location |
|---|---|
| Source (this repo) | [github.com/Ryukijano/world-model-rankability](https://github.com/Ryukijano/world-model-rankability) |
| Live Space (Gradio) | [huggingface.co/spaces/Ryukijano/world-model-rankability-live](https://huggingface.co/spaces/Ryukijano/world-model-rankability-live) |
| Static leaderboard Space | [huggingface.co/spaces/Ryukijano/world-model-rankability](https://huggingface.co/spaces/Ryukijano/world-model-rankability) |
| Private weights (all sizes) | [huggingface.co/Ryukijano/robojepa-22m-rankability-live](https://huggingface.co/Ryukijano/robojepa-22m-rankability-live) |
| Space app source in git | [`demo/live/space/`](../demo/live/space/) |

The live Space downloads weights with `huggingface_hub.snapshot_download` from the **private** model repo above.
Space runtime must have a read token (see [HF_TOKEN](#hf_token-space-secret)).

## DGX Spark (spark-1240) checkout

On the lab Spark box, a working clone used in this session lives at:

```text
/home/aimsgroupuol/world-model-rankability
```

Clone elsewhere:

```bash
git clone https://github.com/Ryukijano/world-model-rankability.git
cd world-model-rankability
```

**Git credentials:** `gh auth status` was configured as user `Ryukijano`; pushes use HTTPS via `gh`.

**Hugging Face CLI:** `hf auth whoami` was available as `Ryukijano` on this host. Do not commit tokens; use
`hf auth login` or Space secrets.

Optional: this host also runs a **Cursor private worker** (`~ @ spark-1240`) for Cloud Agents. For an integrated
terminal and desktop share from agent runs, the worker must start with `--connect-workspace` and `--share-desktop`
(see Cursor self-hosted docs). A helper script on that machine:
`/home/aimsgroupuol/.local/bin/cursor-worker-spark-live.sh`.

## Live Space startup failure (401 on weights)

**Symptom:** Space crashes at import time with HTTP 401 when calling `snapshot_download` for
`Ryukijano/robojepa-22m-rankability-live`.

**Cause:** No Hugging Face token with access to the private weights repo (missing or invalid `HF_TOKEN` Space
secret).

**Fix in code** (`demo/live/space/app.py`):

1. Resolve token: `os.environ.get("HF_TOKEN") or huggingface_hub.get_token()`.
2. If still missing, raise a clear `RuntimeError` pointing operators to Space **Settings → Secrets → `HF_TOKEN`**.
3. Pass the resolved token into `snapshot_download(WEIGHTS_REPO, token=...)`.

**Fix in docs** (`demo/live/space/README.md`):

- YAML frontmatter `models:` entry for the weights repo.
- Short note that operators must set the `HF_TOKEN` repository secret.

## HF_TOKEN (Space secret)

1. Open the live Space → **Settings** → **Repository secrets**.
2. Add **`HF_TOKEN`** = a Hugging Face **read** token that can access `Ryukijano/robojepa-22m-rankability-live`.
3. Rebuild or restart the Space if it was already failing at startup.

Check that the secret **key** exists (values are write-only):

```bash
hf spaces secrets list Ryukijano/world-model-rankability-live
```

On 2026-11-09 the key `HF_TOKEN` was already present on the Space; only app/README changes were uploaded.

Never paste tokens into git, issues, or chat logs.

## Deploy Space files from git

From the repository root, after editing `demo/live/space/app.py` or `demo/live/space/README.md`:

```bash
cd demo/live/space
hf upload Ryukijano/world-model-rankability-live app.py README.md \
  --repo-type space \
  --commit-message "fix: HF_TOKEN error and document private weights secret"
```

Requires `hf` authenticated as a user with commit access to the Space.

**Verification on host:**

```bash
python3 -m py_compile demo/live/space/app.py
```

**Hub reference deploy** from this session: Space commit
[`7a3497e`](https://huggingface.co/spaces/Ryukijano/world-model-rankability-live/commit/7a3497ea4e4b96494ab5fb6e653a06f781b680ad).

## GitHub change (2026-11-09)

| Item | Value |
|---|---|
| Branch | `fix/live-space-hf-token` (from `main`) |
| Commit | `c26c926` — fix(space): document HF_TOKEN and fail clearly for private weights |
| Pull request | [PR #3](https://github.com/Ryukijano/world-model-rankability/pull/3) |

After merge, keep the Space in sync by uploading from `main` or re-running `hf upload` from the same paths.

### Open PR locally

```bash
cd /path/to/world-model-rankability
git fetch origin
git checkout fix/live-space-hf-token
git pull origin fix/live-space-hf-token
```

### After merge

```bash
git checkout main
git pull origin main
```

## Related scripts in this repo

| Script | Purpose |
|---|---|
| [`demo/live/verify_gpu.py`](../demo/live/verify_gpu.py) | Score parity on GPU vs stored benchmark |
| [`demo/live/verify_all_sizes.py`](../demo/live/verify_all_sizes.py) | All eight sizes vs stored scores |
| [`demo/live/export_weights.py`](../demo/live/export_weights.py) | Repackage weights for the live Hub repo |
| [`slurm/40_export_live_weights.sbatch`](../slurm/40_export_live_weights.sbatch) | Isambard export jobs |
| [`slurm/41_verify_live.sbatch`](../slurm/41_verify_live.sbatch) | Isambard verification |

Heavy training and export still target Isambard (`/projects/u6xn/rankability`); Spark is used here for git, Hub
uploads, and agent-driven edits.

## Checklist (new machine or fresh Space)

- [ ] `git clone` and `gh auth login` / `HF_TOKEN` in environment for CLI (optional)
- [ ] Space secret `HF_TOKEN` set with access to private weights repo
- [ ] `hf upload` of `app.py` + `README.md` after merges touching live demo
- [ ] Space builds and loads without 401; smoke-test one state in the UI
- [ ] PR merged on GitHub; `main` matches deployed Space intent
