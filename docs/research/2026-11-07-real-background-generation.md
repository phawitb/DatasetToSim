# Full-workspace stacking with real camera appearance

The Generate data builder now exposes independent **Real-image background** and
**Match real camera appearance** options. Randomize object layout proposes across
the full calibrated 405 × 405 mm mat, with footprint margins, robot-base exclusion,
and object separation. IK, collision/tracking checks and contact-driven task
validation still decide which proposals enter the dataset. This does not mean the
entire mat is reachable or that ten accepted layouts uniformly cover every corner.

## Requested batch

- Name: `so101-stack-real-background-10`
- Five `Stack the red cube on the green cube.` demonstrations.
- Five `Stack the green cube on the red cube.` demonstrations.
- Natural motion, Recorded robot pace, speed scale 1.0, seed 0.
- Enabled: layout randomization, real background, real appearance matching.
- Disabled: material/light randomization, camera jitter, sensor noise augmentation,
  physics randomization.
- Validation fraction 0.2; split by physical trajectory group and task.
- Planned synthetic demonstrations, not unchanged-action replay or hardware trials.

## Reference and rendering

`calibration/real-appearance-v1/profile.json` records reference timestamps, image
and source-video hashes, camera calibration, per-camera semantic color statistics,
and the foreground fill fraction. The source is `SIMCUBE_1O1B_VF`: twelve episodes
sampled at six timestamps each, giving 72 Top and 72 Wrist observations. Actual
recorded robot states align the geometry masks. Original recordings are unchanged.

A masked temporal median creates the Top background. Robot/object geometry masks,
chroma exclusion and expanded shadow margins remove moving foreground. About
23.8% of the reference plate is filled from nearby observed illumination on the
same surface, with feathered boundaries; those pixels are explicitly marked in
`valid-mask.png`. This is a reconstructed background, not a newly captured empty
workcell. Missing texture cannot be recovered from unseen pixels.

At every simulated frame the plate is projected onto calibrated static planes.
Wrist projection follows the actual simulated camera pose. MuJoCo segmentation
preserves foreground robot/object geometry and occlusion; rendered contact shadows
modulate the plate. Pixels outside the reference camera coverage use a surface
color fallback. Planar projection is not a 3D reconstruction of cables or distant
objects outside the work mat.

Appearance processing uses fixed per-camera and semantic-surface Lab mean/contrast
transfers plus mild smoothing. Wrist background brightness/color is matched to the
actual Wrist reference, not copied unchanged from the brighter Top camera. Actual
red/green cube colors use the SAM masks of captured run
`dc7fe3a0b44040d6838eb254eb28e2b0`. The Wrist cube appearance is an estimate using
the measured Top-to-Wrist mat color shift, because that capture only supplies
segmented Top cube references. Cube geometry and labels remain unchanged; no neural
translation or learned texture reconstruction is used.

The renderer measures simulated cube statistics at twelve trajectory poses and
freezes the transformation for the episode. It does not independently vary color
frame by frame. Both cameras render the integrated pre-action state corresponding
to state/action row i; there are N exported observations for N commands, with the
extra terminal integrated state retained only in provenance.

## Reproduction and limits

Build the versioned reference assets using the local graphics-enabled environment:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/studio_real_appearance.py /Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF
```

Build through Generate data using the settings above. Each exported episode keeps
the appearance profile, background, observed-pixel mask, visual settings, scene,
plan and physics trace. The exporter hashes these files. Preserve the versioned
reference assets when reproducing: new generation/rerender loads this reference
version, while the dataset contains snapshots for audit.

This reduces selected appearance differences but does not validate sim-to-real
policy transfer. Robot mesh detail, cube surface markings, lens effects, unseen
background geometry and estimated contact parameters still differ from reality.
The real source informed calibration; synthetic validation is not a blind real
robot evaluation. A separate real rollout evaluation is required for any trained
policy.

## Verified result

Completed job `aa35f96251e6460ab3bf6ea1614d2458` contains 10 accepted episodes
from 25 full-workspace proposals (15 rejected before export), exactly 5 per task.
There are 4,529 synchronized rows at 30 Hz, 150.967 seconds total, with 8 training
and 2 validation episodes. All ten layouts are distinct. Accepted seeds are
red-on-green `[2,4,10,11,14]`, green-on-red `[15,20,22,23,24]`.

The independent `batch-audit.json` verifies successful released, settled stacks,
zero initial overlap, no post-initial state writes or target clipping, and exact
array equality of every command and integrated state against the pre-render
physical validation baseline. All ten first Top images show both cubes (minimum
254 visible pixels for either cube). Image processing does not alter those traces.

The official LeRobot reader read all 4,529 rows and decoded 9,058 camera frames.
The ZIP integrity check passed. Browser checks confirmed switching episodes,
playing a complete episode, and identical Top/Wrist/chart/timeline indices at
multiple playback positions. Evidence lives beside the job in `batch-audit.json`,
`initial-visibility-audit.json`, `official-reader-audit.json`, `playback-audit.json`,
`visualize-proof.png` and `completed-dataset-proof.png`.

Regression verification: 226 Python tests and 19 JavaScript tests passed. Review
found and fixed the missing Top floor-reference fallback and incorrect use of
plane grid spacing as surface height. The latter was corrected during rendering;
an all-frame audit of every accepted moving Wrist trajectory confirms zero floor
pixels with reference-camera coverage, so both plane heights use the identical
constant fallback and the corrected code does not change any batch image.
`floor-projection-audit.json` records this check over all 4,529 frames.

No hardware rollout or trained-policy evaluation was performed for this batch.
