# Calibration and failure diagnosis

## Data and units

Build a channel contract: feature name/order, absolute/delta/velocity/torque semantics, native unit, hardware normalization, mechanical reference, sign and inverse observation mapping. Confirm from recording-time code/configuration; numerical quantization is corroborating evidence. A current cached calibration does not prove historical identity.

Distinguish stored policy actions from commands sent after processors or clipping. If that history is missing, retain the uncertainty rather than calling stored values verified hardware commands.

For a verified normalized 0–100 gripper with encoder endpoints, one possible mapping is:

```
degrees_per_native_unit = (range_max - range_min) * 360 / encoder_max_tick / 100
model_angle_deg = closed_reference_deg + sign * native_value * degrees_per_native_unit
```

This is conditional on that hardware convention. Do not apply it to radians, a different normalization or an unverified robot. Mechanical closure and encoder scale are separate quantities. A first observed opening of 0.8 is not a zero reference.

Join trajectories by episode identity, not by assuming compacted state-table indices equal video-global indices. Validate finite values, shape/order, episode/frame counts, increasing time and video offsets. Reject incompatible mappings rather than silently reshuffling channels or clipping values.

## Time and integration

Use recorded timestamps and causal holds, including variable intervals. If the final action has no following timestamp, document its terminal interval, usually one nominal period from this dataset's FPS. Store N applied actions and N+1 physical state samples, including the state after executing the last action.

Pre-recording motor history is usually unknown. Initializing history to the observed initial pose is an explicit assumption. Do not interpolate toward future commands to smooth replay.

Camera delay is an observation parameter. Two frames at 30 Hz is not two frames at 25 Hz. Check video slice offsets and timestamp labels; do not change action boundaries to match images.

## Geometry and cameras

- Separate measurements from estimates: object size, outside versus inside bowl diameters, height, wall/floor thickness, mat thickness and base support. An outer diameter does not define the inner contact surface.
- Check mesh underside/body-frame offset. Putting a body origin on the table can bury the base or create a fictitious riser.
- Use consistent intrinsic/extrinsic transforms for rendering and scene estimation, including crop, resolution and aspect ratio. Attach wrist cameras to the correct moving body.
- Initialize scenes from initial RGB or independent measurements. Account for object height in projection. Inspect occlusion, segmentation, symmetric yaw ambiguity and rounded mat corners; cables can masquerade as edges.
- After changing a camera, re-estimate image-derived initial poses with that same camera. Stale world coordinates invalidate the experiment.
- Check empty closure and open gripper poses. Match collision surfaces as well as visual meshes. Preserve collisions; unverified finger extensions or invisible pads can manufacture success.

## Fit physical response, not a replacement trajectory

Fit shared gains, damping, reflected inertia, friction, force limits and latency against native-state error. Then examine reference geometry, scene estimation and contact response. Bound parameters with measurements or explicit estimates.

Separate blocked-gripper plateaus from closing transients. Do not require every grasp to have opened above an arbitrary threshold; real successful recordings can open less. Inspect stable observed plateaus.

When mutating compiled-model inertia or other constant-dependent fields during fitting, refresh derived MuJoCo constants with the installed API (`mj_setConst` in the validated implementation), reset data and reinitialize original state/history. Check trial rollout equality against a fresh model rebuild. In-memory optimizer loss is not final physical validation.

Do not combine arm degrees and normalized gripper units into an unlabeled RMSE. Evaluate images from integrated states as well as measured-state camera-fitting diagnostics. Mask IoU is silhouette overlap, not physical accuracy.

## Failure evidence

| Symptom | Inspect first |
|---|---|
| No jaw contact | Initial object pose, references, joint tracking, height, camera projection |
| One jaw / cube pushed away | Per-jaw onset, approach orientation, shape, initial pose, surface friction |
| Grasp then slip | Loaded bilateral duration, penetration, relative motion, torque limits, friction/compliance |
| Lift but miss bowl | Original-command release pose/velocity and actual inner bowl geometry |
| Initial self/table overlap | Mesh envelope, references, support; do not hide by disabling collisions |
| Better images but worse grasp | Identifiability, occlusion, stale initial poses, camera/reference coupling |

Retain original failures and new regressions for every candidate. Keep rejected trials separate. If evidence cannot distinguish camera, reference, compliance or material error, state the ambiguity and identify the measurement that would resolve it rather than adding arbitrary parameters.

## Acceptance evidence

Define task criteria before selection. Pick/place needs loaded bilateral contact for meaningful duration, secure lift, release, full object geometry inside the physical receptacle and terminal settling. A momentary touch, centroid in a visual circle or one high-Z sample is insufficient.

Report overlap/contact forces, numerical warnings and initial penetration separately from success. Thresholds are task/model-specific and remain fixed across comparisons. Recheck screening proxies with the full scorer.

Hash inputs, native actions, targets and archived models/configs; verify each hold boundary. Renderer-only pose assignments are playback, not evidence of contact-driven simulation—trace provenance establishes the difference.

Verify video frame counts, FPS/timebase and duration, and visually inspect recovered and failing cases when available. Deliver honest counts and limitations even if perfect replay remains unresolved.
