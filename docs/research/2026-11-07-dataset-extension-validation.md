# Dataset continuation and visual controls validation

Implemented an immutable parent-linked dataset revision workflow. Add episodes opens Build with the parent's settings, editable options and the first unused seed in the lineage. The revision includes inherited accepted episodes and newly validated episodes. Inherited trajectory-group train/validation assignments remain fixed; new generation settings apply only to the added episodes.

## Appearance controls

Lighting controls include intensity, azimuth, elevation, shadow strength and seeded variation around these settings. Exposure/color matching is performed under reference lighting, then the configured illumination ratio is reapplied so appearance matching does not erase lighting changes. An uploaded background receives the estimated real Top-to-Wrist camera response in the Wrist view.

The background library accepts PNG/JPEG/WebP still images up to 12 MB and 20 million pixels, normalizes orientation and dimensions, strips metadata, and stores immutable content-addressed local assets. Each generation/preview snapshots its selected image. Uploaded imagery is mapped to the background plane and is explicitly marked uncalibrated.

Preview 5 scenes displays five vertically arranged Top/Wrist still pairs, sharing the generation renderer and visual settings. These show appearance and initial layouts; they do not establish IK reachability or physical task success. New requests cancel obsolete previews; generation can cancel an unfinished preview before starting.

## Automated checks

- Full Python suite: 252 passed; two dependency deprecation warnings.
- JavaScript suite: 29 passed.
- `git diff --check`: clean.
- Focused coverage includes invalid uploads/options, protected media routes, asset snapshots, obsolete-preview cancellation, continuation failures/seed advancement, changed settings, inherited task labels and fixed split assignments.

## Interactive checks

- Add episodes opened the 10-episode stacking parent with its original settings and seed 25.
- Uploaded the existing clean calibrated background through the actual file input and selected it from the saved library after restart.
- Rendered five preview pairs with intensity 1.2, azimuth -60 degrees, elevation 55 degrees, spread 45 degrees and shadow strength 0.7, with lighting randomization enabled.
- Preview ID: `626cd7836bf24db1a8fe946c4d0f9dab`.
- Extension test ID: `fcbc74ef236f49efb5bafdeec0827769`, parent `aa35f96251e6460ab3bf6ea1614d2458`.
- Test requested one new red-on-green episode; four proposed layouts failed planning/physical checks before seed 29 passed and produced 423 frames.
- Existing episode playback was exercised during generation; Top/Wrist and joint state/action cursor advanced together.
- No physical robot commands were issued.

Evidence is saved locally in `output/dataset-extension-validation/`.

## Combined export result

- Completed revision: 11 episodes, 4,952 frames, 9 training / 2 validation.
- Parent integrity: all 439 snapshotted files unchanged.
- Inherited data integrity: 188 source files copied identically; all 10 inherited episode action/state hashes, trace/plan hashes, camera video hashes and trajectory-group split assignments match the parent.
- New episode: seed 29; the next available lineage seed is 30.
- Selected background snapshot SHA-256 verified; ZIP integrity passed.
- Official installed LeRobot reader audit passed: 11 episodes, 4,952 frames and 9,904 decoded camera frames. Physical transfer is not validated by this simulated export.
- macOS emitted an existing duplicate AVFoundation library warning during the official-reader audit; the audit exited successfully.
