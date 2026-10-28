# Our model: pixel World-Action Model on real video (draft v1)

## What it must fix (measured)
1. Resolution floor: outcome differences below ~4 mm/step are invisible (coin flip on sim ranking AND real video).
2. Action signal is diluted in whole-frame latents (change-weighted scoring hurt at every size — the signal is real but spread thin).
3. Imagination and choosing pull against each other (rank-loss up → imagination error up).
4. No "which action was better" labels on real video (rank-loss needs sim labels; fine-tuning Meta's weights inherits NC-SA licence + frozen-encoder limits).

## Design
- **Encoder: from pixels.** LeWM-style ViT (~15-100M), trained end-to-end on real frames. No frozen V-JEPA input — Meta's encoder is the licence-and-resolution problem, not the foundation.
- **Base: LeWM + AC-MTM.** LeWM trains in hours/GPU (measured 0.7 s/step, 13.8 GB peak on GH200 on our 1.67M-frame DROID set). AC-MTM's action-contrastive loss forces latents to encode WHICH action caused the change — directly attacks floors 1+2.
- **Anchoring (optional ablation):** distill/regress toward frozen V-JEPA 2.1 features on the same frames (teacher anchoring) — keeps generality while staying pixel-native.
- **Outcome-matching preference head (attacks floor 4):** sample K candidate action sequences per frame; score imagined futures vs observed continuation (ThinkWorld-style reinforcement, but on latent distance), plus DROID success/failure episodes (1,574 available in raw copy) as preference pairs where label exists. No simulator needed.
- **Separate action head, gradient-blocked (attacks floor 3):** action/preference head reads the encoder but its loss does not flow into it (π0.7-style insulation) — ablated.

## Data (staged, verified)
- DROID robojepa copy: 5,070 episodes for world-model training (already converted for LeWM: 4,999 train + 102 holdout).
- + community-v3 (~961 GB), AgiBot core (~337 GB), Open-AoE (~385 GB) as the "good chunk of data" — convert incrementally.
- Preference split: droid-raw-1.0.1 failure episodes (1,574) + outcome-matching negatives.

## Training budget (1 GH200 per job, per our rule)
- Stage A (world model): LeWM+AC-MTM on DROID, ~24 GPU-h (~2 days chained). Baseline already measured.
- Stage B (preference head): on frozen encoder, ~6 GPU-h.
- Stage C (ablations): anchor on/off, gradient-block on/off, AC-MTM on/off — 4 × ~6 GPU-h.
- Total ~50 GPU-h for the first full result; AgiBot/community-v3 scale-up only after Stage A passes the eval gates.

## Eval gates (no robot needed)
1. DROID graded-edit test (running): beat Meta 1B on ±10%/±25% magnitude — the floor.
2. RoboCasa rankability bench: beat frozen-1B fine ρ .418.
3. Success/failure prediction on held-out DROID episodes (build the split from the 1,574 failure eps).
4. Closed-loop RoboCasa planning (existing harness) — headline number.
