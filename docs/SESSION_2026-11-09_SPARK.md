# Session log — DGX Spark (2026-11-09)

Chronological record of work on **world-model rankability** from a **Cursor Cloud Agent** run on the lab
**spark-1240** private worker (`bc-8e77c559-5774-5ce4-9afa-adccf3913697`, Project *AIRR-Open_source_AI*).
Operations runbook: [OPERATIONS.md](OPERATIONS.md).

## Environment

| Item | Detail |
|---|---|
| Host | `spark-1240` (DGX Spark / GB10), Linux |
| Worker ID | `be33a135-df69-5337-b2ae-f3848065d817` (display `~ @ spark-1240`) |
| Agent run | [Live Space and PR push](https://cursor.com/agents/bc-8e77c559-5774-5ce4-9afa-adccf3913697) |
| Git user (local clone) | Gyanateet Dutta / `gyanateet@gmail.com` |
| `gh` | Authenticated as **Ryukijano** |
| `hf` | `hf auth whoami` → **Ryukijano** |

## Problem we fixed

The Hugging Face Space **[Ryukijano/world-model-rankability-live](https://huggingface.co/spaces/Ryukijano/world-model-rankability-live)**
failed at startup with **HTTP 401** on `snapshot_download` for the **private** weights repo
`Ryukijano/robojepa-22m-rankability-live`. The Space had no usable token at runtime (or only CLI login on a dev
machine, not the Space secret).

An equivalent fix had existed only on a cloud VM as local commit `ed85e91` (never on GitHub `main`).

## Code and documentation changes

### `demo/live/space/app.py`

- Import `get_token` from `huggingface_hub`.
- Before download: `_hf_token = os.environ.get("HF_TOKEN") or get_token()`.
- If missing: raise `RuntimeError` with instructions to set Space secret **HF_TOKEN**.
- `snapshot_download(..., token=_hf_token)`.

### `demo/live/space/README.md`

- Frontmatter: `models: [Ryukijano/robojepa-22m-rankability-live]`.
- Operator note: Space repository secret **HF_TOKEN** required for private weights.

### `docs/OPERATIONS.md` (new)

Deploy runbook: artifact map, spark clone path, `HF_TOKEN`, `hf upload`, PR links, verification, Isambard vs Spark roles.

### `README.md`

- Live demo line: weights are private; link to **OPERATIONS.md**.
- Repository layout: `docs/` includes operations runbook.

## Git and GitHub

```text
Clone:  /home/aimsgroupuol/world-model-rankability
Branch: fix/live-space-hf-token  (from main @ 2c6415c)
```

| Commit | Message |
|---|---|
| `c26c926` | fix(space): document HF_TOKEN and fail clearly for private weights |
| `c2e486e` | docs: add Spark and live Space operations runbook |

**Pull request (open):** [PR #3](https://github.com/Ryukijano/world-model-rankability/pull/3) → `main`.

**Verification:** `python3 -m py_compile demo/live/space/app.py` passed after the code change.

## Hugging Face Hub (same session)

**Space upload** (from `demo/live/space/`):

```bash
hf upload Ryukijano/world-model-rankability-live app.py README.md \
  --repo-type space \
  --commit-message "fix: HF_TOKEN error and document private weights secret"
```

**Hub Space commit:** [`7a3497e`](https://huggingface.co/spaces/Ryukijano/world-model-rankability-live/commit/7a3497ea4e4b96494ab5fb6e653a06f781b680ad).

**Secrets:** `hf spaces secrets list Ryukijano/world-model-rankability-live` showed key **`HF_TOKEN`** already present
(updated 2026-10-28). Value was **not** rotated in this session (write-only; no token pasted in chat).

## Cursor worker: desktop share and terminal (investigated, not fully enabled)

The spark worker process was running **without** `--share-desktop` or `--connect-workspace`. Logs showed
`computerUseEnabled: false` and `recordScreenEnabled: false`, so **Show desktop** and the agent **Terminal** tab stay off
until the worker restarts with those flags.

**Created locally on spark:**

| Path | Purpose |
|---|---|
| `/home/aimsgroupuol/.local/bin/cursor-worker-spark-live.sh` | Starts worker with `--share-desktop view_and_control` and `--connect-workspace` |

**Not completed on spark (needs your sudo):** install Linux desktop deps for share-desktop:

```bash
sudo apt-get install -y --no-install-recommends \
  dbus-x11 tigervnc-standalone-server x11-utils x11-xserver-utils xdotool xfce4
```

**Project store note (Cursor Context):**
`/home/aimsgroupuol/.local/state/cursor/agent-stores/cursor_agent_stores/bc-82ee85fc-9958-4868-be89-524f764a1b52/files/internal/spark-1240-worker-desktop-terminal.md`

Original task spec (Coordinator):
`.../internal/live-space-hf-token-fix.md`

## What to do next

1. **Merge [PR #3](https://github.com/Ryukijano/world-model-rankability/pull/3)** when reviewed.
2. Confirm the **live Space** boots (no 401); smoke-test one benchmark state in the UI.
3. If Hub README lags `main`, re-run `hf upload` from `demo/live/space/` per [OPERATIONS.md](OPERATIONS.md).
4. For Cursor **terminal + desktop** on spark: restart worker with flags (see OPERATIONS or the worker script above).

## Quick paths (copy-paste)

```bash
cd /home/aimsgroupuol/world-model-rankability
git checkout fix/live-space-hf-token
git pull origin fix/live-space-hf-token
```

```bash
cd /home/aimsgroupuol/world-model-rankability/demo/live/space
hf spaces secrets list Ryukijano/world-model-rankability-live
python3 -m py_compile app.py
```

---

*This file is the narrative session log; [OPERATIONS.md](OPERATIONS.md) is the maintained operator runbook.*
