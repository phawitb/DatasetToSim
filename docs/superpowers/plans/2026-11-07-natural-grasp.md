# Natural grasp correction

Goal: Retain coordinated Natural travel while validating actual cube retention and placement.

Evidence: Real run 24e8dc missed grasp. Encoder-derived Approach midpoint differed by 14.5 mm; pan was still converging during low descent. Error alone does not isolate calibration, compliance and travel coupling.

Hypothesis: Finish coordinated reach at a Pregrasp endpoint 45 mm above the grasp pose; use a centered short descent and 0.4 s open-jaw settle before closing. This preserves high travel curves while providing a predictable local grasp approach. Do not change shared calibration, robot forces, collision masks or execution gates to manufacture success. Alternative global offsets lack independent evidence; runtime feedback compensation would invalidate the exact-simulated-command contract.

- [x] Add regression for centered descent, open-jaw settle and exact critical endpoints; observe failure before implementation.
- [x] Implement Natural-only Pregrasp and settle, retaining legacy methods.
- [x] Full contact simulations on failed scene and earlier bowl/stack scenes.
- [x] Fresh UI capture/SAM/plan/sim before physical execution.
- [x] Inspect loaded gripper plateau and recorded transfer/placement, not just command completion.
- [x] Record failures and successful evidence separately; regression verification and review.

Outcome: fresh physical run dc7fe3a0b44040d6838eb254eb28e2b0 retained the cube, stacked it and returned Home. Full evidence and limitations in docs/studio-natural-motion-2026-11-07.md.
