# SAM From / To selection implementation plan

> Implement inline with verification before hardware execution.

**Goal:** Replace live color-based object discovery with explicit SAM image prompts for From and To, then test the full real pipeline.

**Architecture:** A persistent isolated SAM 2.1 worker uses the installed Ultralytics runtime and local weights. The browser sends point prompts tied to an immutable captured top image. Mask geometry feeds the existing calibrated cube/bowl scene and MuJoCo planning. Preflight reruns SAM on a new frame with the saved prompts and compares scene poses. Archived benchmark replay stays unchanged.

**Scope:** Arbitrary visual object selection is supported; physical planning remains restricted to the measured cube and bowl profiles. No claim of arbitrary-object 3D reconstruction from RGB.

- [ ] Add SAM worker/client with image embedding cache, finite point validation, foreground mask validation and model/image hashes.
- [ ] Add capture-bound selection API, mask overlays and From/To controls; invalidate plans when selection changes.
- [ ] Derive scene poses from SAM masks only, including cube silhouette fitting without HSV segmentation; require both roles before planning.
- [ ] Run SAM again in execution preflight; preserve calibration, movement, stop, ownership and freshness checks. Integrate Home with SAM-selected scenes.
- [ ] Regression test wrong/stale selections, same-object roles, prompt coordinates, mask geometry and execution preflight. Review before hardware.
- [ ] Test actual browser clicks, planning, MuJoCo, real execution and final camera outcome. Record evidence and limitations.

## Validation result

Implemented and tested on 2026-11-06. See `docs/studio-sam-validation-2026-11-06.md`: SAM UI selection, simulation, and actual SO-101 transfer succeeded in run `4def04d354554195a264d4eb08ea49c3`; final wrist image confirms cube inside bowl. 96 Python and 3 Node tests passed. Home after transfer correctly blocked on insufficient visible bowl area; no Home motion was dispatched. The live vision freshness budget is five seconds, measured against actual camera timestamps and rechecked inside the hardware dispatch lock.
