# Generate data: build and visualize

User requests two tabs, Build a dataset and Visualize data, with synchronized Top/Wrist/state/action playback, following VLA-Benchmark. Existing design autonomy remains applicable.

Build retains generation controls, progress, saved settings, cancellation/download and history. Visualize offers batch and episode selection, two camera canvases, a single Play/Pause + frame scrubber/step/speed control, and six per-joint observed-state/commanded-action graphs with values and a common cursor. All UI English. Navigation pauses playback, polling must not reload the selected episode or jump to a newly accepted episode. Deep links retain tab/job/episode.

Reference inspected: VLA-Benchmark/static/index.html visualizer around lines 775–843, 5605–5708, 6034–6083. Adopt its dataset/episode controls and linked chart cursor. Its multiple native video clocks require drift correction; instead prepare a disposable side-by-side camera video and paint two canvases from one decoded frame. Training videos/rows remain unchanged. Playback callback mediaTime chooses floor(t*30), clamped to N−1. One clock also handles seeking, pause, speed and replay. State is integrated pre-action native state[i], action is planned native command[i]; terminal N+1 sample never becomes a row. Five arm channels in degrees, gripper 0–100.

Backend read-only source API validates accepted episode, provenance hash, finite states, dimensions/timestamps and camera frame counts. On-demand background preview encoding caches by source fingerprint; preparation/error UI explicit. No robot commands. Verify real existing five-task artifact plus regression tests for causal pairing, camera merge, frame boundaries, source changes and refresh during playback.

## Verification

Implemented Build and Visualize tabs with persistent tab/job/episode links. Visualizer loads actual saved native trace states and unchanged commands, then presents Top/Wrist from one paired decoder. On-demand previews are disposable cache files, not training-video replacements.

- 202 Python regression tests and 9 JavaScript playback tests passed.
- Actual five-task batch: all 4,119 causal state/action rows checked against saved N+1 traces; no terminal-row inclusion.
- Browser tested play/pause, 2× speed, seeking to first/last frames, one-frame advance, episode changes, tab pausing, deep-link reload and existing New Run/Datasets navigation.
- Playing episode 001 continued across background status polling: both canvases and plots reported frame 209 together.
- Unstack seek to frame 420: Top/Wrist/plots all 420; displayed state `[-33.7,40.9,-43.3,101.1,-21.6,11.6]` and action `[-32.2,36.4,-43.7,101.1,-20.3,0.0]` match the saved artifacts (rounded to one decimal).
- Screenshot: `output/generated/visualize-data-ui.png`.
- No physical robot commands, dataset regeneration or GitHub push performed for this UI change.
