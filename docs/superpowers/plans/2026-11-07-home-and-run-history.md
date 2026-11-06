# Occluded Home and Run History Implementation Plan

**Goal:** Return Home after placement while Top bowl is occluded; expose every saved run for review.
**Architecture:** Keep visible-Top SAM verification; add current Wrist SAM plus calibrated measured-joint camera geometry for post-place Home. Anchor recovery to a completed released trajectory with matching measured joints and calibration. Verify current bowl geometry with a bounded projection fit, inflate Home-only bowl collision geometry for camera uncertainty, validate physics and retain Stop/frame-age/trajectory checks. History is a read-only filesystem catalog, separate from active control state.
**Tech Stack:** Existing Python/FastAPI, MuJoCo, SAM 2.1 worker, plain browser JavaScript.
**Authorization:** User explicitly requests implementing and physically testing Home; previous session delegates website design. Implementation proceeds inline.

## Constraints and review focus
- No torque release, blind motion, copied simulated robot state, or bypass of hardware gates.
- Wrist fallback needs completed pick/place release, unchanged calibration, matching end joints and current camera evidence. Stale reference alone cannot authorize motion.
- Fit rejects wrong shapes, excessive movement, insufficient rim visibility and poor residual. Home collision margin applies only to recovery, never benchmark physics.
- Both planning and immediate preflight use current joint pose for moving Wrist camera. No re-prompt of old From point.
- All saved 32-character capture directories appear in history, including captured-only, failed jobs and Home. Opening history never changes controller owner or active capture; active operations remain bound to their own ID.
- Missing/corrupt files and old reports do not crash catalog; physical task success must remain distinct from command completion and sim success.

## Tasks
- [x] Reproduce Top occlusion and measure Wrist SAM mask / projection fit on current images, with no movement.
- [x] Add regressions for large Wrist masks, completed-release/pose binding, geometry bounds and stale/changed preflight.
- [x] Implement Wrist evidence module and Home-only conservative scene, integrate server freshness and job metadata.
- [x] Validate Home from saved images in physics; inspect clearances and render before hardware. Earlier pose passed; the later drifted pose failed and was not dispatched.
- [x] Add run catalog API plus selectable Live run history and reload/deep-link support; test archive vs active-control isolation.
- [x] Review, run focused/full checks, restart local server, click Home on real robot, test selecting old runs without hardware motion. Three attempts were blocked; reconnect fix verified with zero encoder change.
- [ ] Complete physical Home and verify camera-clear final pose. Blocked by current-pose association and conservative initial clearance; requires a separately validated current-state retreat, not weaker thresholds.
- [x] Save reports/screenshots and document observed outcome and limits in `docs/studio-home-history-validation-2026-11-07.md`.
