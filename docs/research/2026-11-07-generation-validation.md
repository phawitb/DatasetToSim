# Generate data validation — 2026-11-07

## Delivered workflow

Open Studio → **Generate data**. Choose one or more tasks, episodes per task, seed, motion technique/speed and independent variation toggles. **Generate dataset** starts a cancellable simulation-only worker. Review accepted/rejected attempts, camera previews, persisted batch settings and history. **Download LeRobot ZIP** appears only after the complete requested batch passes export validation.

Generated files remain under `output/generated/<job-id>/`. No robot commands, VLA training, image-translator training or Hub uploads occur in this workflow. Original recordings remain unchanged.

## Measured integration result

- Browser-created job: `e2c3506461614fd98ff183607160708c`, `so101-five-task-smoke`.
- All five requested tasks, one episode each, seed 0, Natural 100%.
- Layout, appearance, camera, sensor and physics variation all enabled.
- **5 accepted / 5 attempts / 0 rejected; 4,119 training rows.**
- Top + Wrist: **8,238 decoded frames**, 640×480, 30 Hz.
- Official installed LeRobot **0.5.2** reader audit: passed, including numerical temporal action windows at episode boundaries and both camera decodes.
- All tasks retained in train. No independent validation split for this singleton-per-task smoke batch.
- ZIP downloaded through the browser to `~/Downloads/so101-five-task-smoke.zip`; byte hash equals server artifact and ZIP integrity passes. 83 entries, including portable provenance. Model mesh assets referenced externally remain dependencies for re-rendering; videos and training rows are standalone.
- Physical task checks pass in MuJoCo. This is **not evidence of a trained policy succeeding on real hardware**.

## Additional checks

- **199 Python tests passed**, including prior planner/Natural/stack behavior, five-task physical integration, exporter integrity, API protection and worker lifecycle checks.
- **3 playback JavaScript tests passed**; application syntax and whitespace checks passed.
- Extra physical integrations: all five tasks with seeded layout and ±10% mass/friction variation passed. Unstack also passed with Classic, Gentle, Flow, and Natural with a green upper cube. Lower cube maximum disturbance in these extra checks was 0.22–1.75 mm.
- Browser: navigation into Generate data, all five task selections, start/progress, preview selection and actual playback, persisted history after server restart, saved settings, cancellation, complete-export gating and real download verified.
- Cancelled test job `340fbbfd5ea741fea88407877f22fdfc` stopped and exposes no completed export.
- Code review caught and fixed a Home local-variable shadow, a completion race, import-time recovery side effects, appearance-group leakage and singleton-task split exclusion.

## Reproduce the exported artifact audit

```sh
HF_DATASETS_CACHE=/tmp/studio-generation-hf-cache \
HF_HUB_OFFLINE=1 PYTHONPATH=src \
/opt/miniconda3/envs/lerobot/bin/python scripts/studio_generation_export.py audit \
  output/generated/e2c3506461614fd98ff183607160708c/dataset --official-reader
```

Expected result: `valid: true`, `total_frames: 4119`, `total_episodes: 5`, `video_frames_decoded: 8238`, `official_reader_checked: true`, `physical_transfer_validated: false`.

See [research and transfer limitations](2026-11-07-synthetic-vla-transfer.md) before scaling data or evaluating a learned policy. Layout coverage and visual variation remain conservative; the generated validation split cannot replace new held-out physical sessions.
