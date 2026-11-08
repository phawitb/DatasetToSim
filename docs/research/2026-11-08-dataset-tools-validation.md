# Dataset tools validation — 2026-11-08

## Delivered
- Per-task accepted counts for all synthetic and replay datasets, including inherited tasks.
- Initial pick/destination XY maps with 3×3 zone counts, task and origin filters, and clickable episode points.
- Datasets > Build Simulation dataset uses original real native actions and archived physical traces, with shared background, camera appearance, lighting, camera jitter and sensor controls. The current supported calibrated source is SIMCUBE_1O1B_VF.
- Seven actual scene models with interactive local 3D geometry and task-role mappings.

## Evidence
- Python suite: 296 passed (2 dependency deprecation warnings).
- JavaScript suite: 39 passed.
- Root integration API checks: output/dataset-tools-check/api-checks.json.
- Browser verified source selection (68 successful, 4 unresolved), appearance preview, submission, completed export, Play and seek. Top/Wrist/charts agreed at frame7 during playback and frame121 at final seek.
- Website-created dataset: output/generated/351f01e298394b8bafecc27b0c8561fd, so101-real-replay-ui-check, source episode1, 122frames, 640×480 Top/Wrist,30Hz; successful validated LeRobot v3 ZIP.
- Independent one-episode export: output/replay-dataset-smoke/2fd3fa50c9cf4713b782c9f6aa4e751d; unchanged source actions, N+1 integrated states,244decoded video frames, valid ZIP.
- 600-episode job28808b6f1afb4044bf978d23e994f766 completed:300 each stack task. Each task has pick and destination zone counts[33,34,34,33,33,34,33,33,33]. All600 positions present and within saved mat bounds. 480training/120validation,275374frames.

## Interpretation
The real replay builder re-renders frozen calibrated physics, not recorded-state animation. Source camera offsets are preserved as provenance; exported cameras render the same pre-action simulation state as exported observations. Source metadata ranges and statistics are not trusted for export; actual episode rows and regenerated metadata/statistics are used. Source datasets remain unchanged. Physical task failures are labeled and excluded from training export. Appearance matching and valid LeRobot format do not establish policy transfer to physical hardware. No robot movement was performed.

## Follow-up: robot origins and model color variants
- Coverage uses each episode's saved robot_base position/quaternion. Both maps expand their display bounds to include origins outside the mat; mat zones/counts remain unchanged. Base marker is explicitly an origin marker, not an arm silhouette or footprint.
- Cube red/green share one catalog model with nested variants and per-variant task roles. The catalog now has six model families; original mesh URLs stay available.
- Catalog swatches use compiled simulation RGBA rather than independent CSS colors. Material color mode is unlit; Studio lighting retains an orbit-friendly shaded view. Neither claims to reproduce per-dataset real-camera color matching or randomized lighting.
- Targeted Python15/15 and JavaScript8/8 passed; full JavaScript41/41 passed. Browser verified color switching, per-color task roles and both robot-base markers for the600episode dataset.

## Follow-up: dataset deletion, model thumbnails and project material palette
- Visualize Data has a named inline Delete dataset confirmation, recoverable Trash and Restore controls. Active builds and in-flight episode preview creation block deletion. Folder moves preserve all export, episode and provenance files.
- Removed nine other generated/replayed datasets from the active list, retaining only so101-stack-600 (28808b6f1afb4044bf978d23e994f766). The 72 original real recordings are unaffected. All 21,954 original files of the retained dataset have unchanged size and modification time.
- Browser restored cancellation-smoke, selected it, confirmed its name, and deleted it again. Final list contains only the 600-episode dataset and Trash contains nine datasets.
- Six model cards use cached PNG projections of actual geometry. All six thumbnail images decoded successfully in the browser. Color variants and preview lighting use square buttons; four cube material variants include Generate Data, real-dataset Simulation and New Run provenance. Multi-material robot keeps original orange/charcoal part colors.
- Fixed generation bowl material override: the old code replaced calibrated sage RGBA [0.5, 0.65, 0.6, 1] with cube green [0.045, 0.65, 0.09, 1]. Future generation preserves calibrated sage matching archived real-dataset Simulation. Existing exported videos and episodes are unchanged. Lighting and camera color processing can still change rendered pixel colors.
- Full verification: 310 Python tests passed (two dependency deprecation warnings), 46 JavaScript tests passed; git diff --check clean. No hardware movement.
- Evidence: output/dataset-tools-check/final-verification.json, final-python-tests.txt, final-js-tests.txt, dataset-delete-final.png, objects-square-colors.png, bowl-matched-material.png.
