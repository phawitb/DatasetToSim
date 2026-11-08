# Synthetic demonstrations for SO-101 VLA training

## Decision

Start with physically validated demonstrations, calibrated two-camera rendering, bounded appearance/camera/sensor variation, and a small amount of task-matched real fine-tuning. Keep native proprioception and the exact commanded joint targets. A learned image translator is an optional experiment, not a prerequisite or a guarantee of real-world transfer. The new Generate data workflow implements generation/export, not VLA training or a deployed image translator.

## Evidence and design implications

**Robust Visual Sim-to-Real Transfer for Robotic Manipulation (Garcia et al., IROS 2023).** The study tests texture, lighting, object-color and camera variation using an offline cube-localization proxy, then manipulation on hardware. Image augmentation alone was insufficient in its setting; scene randomization and real fine-tuning helped. Object colors were varied only near their nominal colors so that color-conditioned tasks remained distinguishable. This supports treating geometry/calibration and scene diversity separately from sensor augmentation. Our initial implementation uses conservative scene-color/light variation, not the paper's full realistic texture library. These bounds are engineering starting points and have not been optimized on SO-101 held-out trials. [Paper](https://arxiv.org/abs/2307.15320), [project](https://www.di.ens.fr/willow/research/robust_s2r/).

**ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis (2025).** ReBot replays demonstrations in a digital twin and combines virtual robot/object motion with real backgrounds. Its pipeline removes the original robot and manipulated objects using segmentation and video inpainting before composition; dynamic real video backgrounds matter for moving cameras. This is a useful next experiment for our paired recordings, but pasting a simulated robot onto a fixed real wrist image would leave inconsistent camera motion and occlusion. We have not implemented or trained ReBot. [Paper](https://arxiv.org/abs/2503.14526), [project](https://yuffish.github.io/rebot/).

**Grounding Sim-to-Real Generalization in Robotic Manipulation: An Empirical Study with VLA models (2026).** The reported experiments emphasize spatial/camera/table variation, simulation fidelity and combined training interventions. Their findings are platform/protocol dependent; they do not establish that making our screenshots photorealistic will make an SO-101 policy reliable. We use episode-coherent camera and scene variation to avoid unphysical camera jumps, plus small independent sensor noise. This is a conservative design choice, not a replication of every augmentation protocol in the paper. [Paper, version 2](https://arxiv.org/html/2603.22876v2).

**RialTo / Reconciling Reality through Simulation (RSS 2024).** Its real-to-sim-to-real approach uses scene reconstruction and simulation-based policy improvement. This motivates keeping a calibrated digital twin and validating task dynamics as well as images. Our planned trajectories are not a RialTo-trained policy. [Paper](https://www.roboticsproceedings.org/rss20/p015.pdf), [project](https://real-to-sim-to-real.github.io/).

**LeRobot dataset v3.** Export uses tabular state/action rows, synchronized camera video, task strings, episode offsets, statistics and metadata. The local official-reader compatibility check is against installed LeRobot **0.5.2**, not an assertion about every future release. [Dataset format](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3), [dataset API](https://huggingface.co/docs/lerobot/api/datasets).

## What existing paired data can and cannot do

Audit on 2026-11-07:

- Original `SIMCUBE_1O1B_VF`: 72 episodes / 11,661 frames, 30 Hz, Top and Wrist 640×480, only red cube → green bowl. This is roughly 6.5 minutes and does not cover the five requested tasks.
- 54/72 original episode metadata ranges are stale. The original data is left unchanged. New export regenerates indices and validates temporal windows with the official reader instead of copying old metadata.
- Benchmark camera alignment uses a fitted two-frame observation offset; this is not measured exposure synchronization. Simulated/real motion and contacts differ. The pairings are weak supervision, not pixel-perfect paired targets.
- Active camera settings include a wrist override; saved model plus active config are authoritative, rather than stale per-episode camera metadata alone.
- Reviewed Studio manipulation runs include both real success and real failure despite simulated success. Command completion is not physical success. No recorded unstack examples were found.
- Camera receipt timing is not exposure timing; stale/repeated frames and cross-camera skew require causal matching and quality flags before real-video training conversion.
- All original episodes influenced calibration. A random split of those frames is not a new physical generalization test.

## Implemented controls

1. **Calibrated base scene:** fixed 19.5 mm cubes, calibrated robot and cameras, existing actuator/contact model. Both cube colors appear even when one is a distractor. Green bowl is green in the synthetic task. Semantic colors are never swapped by augmentation.
2. **Layout variation:** seeded XY offsets within 10 mm and yaw within 0.10 radians; intentionally narrow first coverage around reachable, validated poses.
3. **Appearance variation:** bounded table/mat colors and light intensity, fixed throughout an episode. This is not photorealistic background synthesis.
4. **Camera variation:** optional local extrinsic offsets within 3 mm, rotations within 0.015 radians and field-of-view offsets within 0.5 degrees; same camera draw throughout the trajectory. These are proposed uncertainty bounds, not measured confidence intervals.
5. **Sensor augmentation:** modest gain, offset, gamma, blur and achromatic noise. No flips, crops, hue swaps or geometry-changing image warps. Parameters are coherent over an episode; only small noise samples vary per frame.
6. **Physics variation:** optional ±10% cube mass/contact-friction variation, with full physical validation after planning. Joint limits, gains and collision masks are not relaxed.
7. **Acceptance:** actual contact integration and task-specific success, collision and tracking gates; unstack additionally requires the lower cube to remain in place and the released upper cube to be supported on the table. Failed attempts remain diagnostics, never training rows. Retries are bounded; an incomplete requested batch is not exported as complete.
8. **Traceability:** seed, model/config, actual actions, N+1 integrated traces and visual draw provenance retained. Training observations use only the first N states paired with N actions, never the terminal observation as an extra labeled row.
9. **Splits:** whole physical-trajectory groups stay together. The generated validation split is useful for software/model checks, but is not a real-world transfer test. If multiple batches are combined later, regroup globally before splitting to avoid duplicate-trajectory leakage.

## Should an image bridge run at inference?

Possible, but first compare against the raw-camera baseline:

- **Sim-to-real at training time:** geometry/mask/depth-conditioned image or video translation; real cameras remain raw at inference. Require preserved object identity/color, gripper contact, occlusion and temporal consistency. A visually attractive hallucinated cube position creates incorrect action supervision.
- **Real-to-canonical at inference:** transform both training and deployment observations into the same domain. Measure processing latency and failure behavior, use identical resizing/normalization/camera ordering, and verify two-camera consistency. Never introduce a translator only at deployment without training the policy with it.
- **Hybrid real background composition:** curate successful aligned pairs, remove real foreground, retain dynamic wrist backgrounds and use depth-aware composition. This is more controllable than unconstrained style transfer but still requires quality checks.

Before training a translator, curate aligned source episodes, reject divergent contact segments, preserve source masks and clocks, split by original episode/session, and establish a held-out localization/color-consistency test. Existing pairs can help this preparation; they do not currently justify a reliable five-task translator. No pretrained translator is installed or advertised as ready in this implementation.

## Evaluation sequence

1. Generate a small balanced batch and inspect both views, state/action conventions, task outcomes and decoder alignment.
2. Expand task/layout coverage incrementally, inspect rejection rates, then compare calibrated-only vs appearance vs camera/sensor vs physics variation with the same grouped split.
3. Fine-tune a chosen VLA using matched native action conventions and Top/Wrist inputs; compare sim-only, real-only and mixed-data baselines. Dataset export alone does not perform this training.
4. Validate the policy in simulation on held-out layouts, then collect **new physical sessions** for all five tasks with physical success review. Report task success, lower-cube disturbance, collisions and latency, not just visual similarity.
5. Add a learned bridge only if it improves these held-out measures over the simpler baseline.

The initial generator does not prove broad real-world robustness: layouts are narrow, appearance variation is modest, dynamics remain estimated, and no VLA was trained or deployed by this change.
