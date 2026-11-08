# Generated motion timing compared with real recordings

## Measurement

Source: `VLA-Benchmark/data/SIMCUBE_1O1B_VF`, 72 episodes, 11,661 frames, 30 Hz. Rows selected by `episode_index`, not stale metadata-global ranges. Arm commands are native absolute degrees; gripper uses its original 0–100 scale.

Real episode duration: median 5.00 s; 10th–90th percentile 4.37–7.31 s; full range 3.97–9.57 s. The reference contains red-cube-to-green-bowl demonstrations only. It cannot establish human timing for stacking/unstacking.

The previous generator inherited Studio's conservative live timing: a 32.2 deg/s effective Natural cap, long travel floors, and grasp/release holds. Initial Home-to-Pregrasp alone took 9.2–9.7 s. Its bowl episodes lasted about 20 s and stack/unstack episodes about 32 s.

The reproducible measurement command is:

```sh
PYTHONPATH=src .venv/bin/python scripts/studio_generation_timing.py /Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF
```

`calibration/generation-recorded-pace.json` stores source shard hashes, per-joint 95th-percentile derivative envelopes, and measurement details. A nine-frame cubic Savitzky–Golay fit suppresses native encoder quantization when calculating derivatives; episode boundaries are excluded. Source observations/actions are never rewritten.

## New timing

Build offers **Recorded robot pace** (default) and **Conservative studio pace** (previous behavior). Recorded timing shortens minimum travel/hold durations and replans at 30 Hz under measured derivative limits. Natural, Gentle and Flow enforce velocity, acceleration and jerk; Classic preserves its original interpolation and enforces velocity only. Smooth techniques retain their existing continuity properties.

Uncapped measured timing achieved physical task success at 7.1 s for the fixed bowl scene, but produced 13.4 degrees peak simulated elbow tracking error. The existing 10-degree arm / 25-unit gripper tracking gate was retained. Empirically checked generation-only arm ceilings are 100/90/75/100/100 deg/s before the interpolator's safety factor. These may cap a requested speed increase. All contact, penetration, grasp, placement and full integrated tracking checks remain enabled. No controller gains, collision masks, physical geometry, or recorded benchmark actions are changed.

Natural randomized/physics-varied examples at 100% now last about 8.8–8.9 s for bowl tasks and 14.2–14.3 s for stacking/unstacking. This is closer to real timing, but is still slower than the real median and is a timing heuristic rather than learned human motion. Exact timing depends on scene, technique and requested speed. A 20-case offline matrix covers five tasks, four techniques, speed endpoints, both upper cube colors, and randomized layout/physics. Results are in `output/generation-recorded-validation/results.json`.

Every accelerated trajectory is integrated afresh. Renderers export its actual 30 Hz observations and native state/action rows; there is no video-speed trick or timestamp-only retiming. Existing datasets retain their original videos/timing and display **Conservative studio pace** when no profile was saved. Generate a new dataset to use the new timing.

Hardware planning retains its original limits. Synthetic captures cannot pass the hardware execution gate. This work does not establish real-robot transfer.

## Workspace

Build contains one form card. Visualize owns the dataset selector, basic dataset information, progress, saved settings, export download and episode list. Task instructions are identical across both tabs, including the selected upper cube for unstacking. The redundant episode selector and Attempt outcomes section are removed. Top/Wrist canvases and all six state/action charts still use one paired-video media timestamp.

## Final verification

- Final offline timing matrix: 20/20 cases passed physical task success and the unchanged collision/tracking checks.
- UI-created batch `f20814dc81ed4bb79e553ed73e7b030d` (`so101-recorded-pace`): five requested, five accepted, zero rejected, 1,813 frames. Native 30 Hz durations: 8.90, 8.77, 14.20, 14.30, 14.27 s in task-list order.
- Official installed LeRobot reader audit passed for all five episodes and decoded all 3,626 camera frames. No real-robot-transfer claim.
- Browser Play at 1× advanced all three camera/chart frame markers together (7 → 272). Seeking to frame 215 showed Top, Wrist, robot state/action and slider all at frame 215 / 7.17 s. Episode selection uses the list; Build submission switched automatically to Visualize.
- UI evidence: `output/generated/generation-recorded-pace-ui.png`.
