# Selectable Flow Motion Implementation Plan

**Goal:** Add a visibly smoother, more continuous and less hesitant motion technique, with Classic preserved in Plan the grasp.

**Design:** Keep Classic commands byte-for-byte unchanged. Add Gentle: C2 quintic joint segments, shape-preserving nonzero velocities at designated free-space waypoints, zero velocity/acceleration at grasp/release boundaries, and bounded velocity/acceleration/jerk. Shorten conservative travel floors where limits allow; preserve contact holds until physics proves a change. Existing limits, freshness, collision and contact gates still apply. Add Flow using local Ruckig state-to-state S-curves with shared travel-knot velocities and exact position-extrema checks; never use its cloud waypoints. Both bowl and stack use the selected technique, including stack planning probes and return Home. Frozen plan records technique and trajectory metrics. UI selection changes require regeneration and simulation, not replay-speed changes.

**Approaches considered:** Speeding up the existing cubic still stops at every waypoint and creates acceleration jumps. Unconstrained splines can overshoot near objects. Chosen bounded quintic segments blend only explicitly designated travel knots and retain mandatory task stops. Gentle uses NumPy polynomial extrema. Flow uses Ruckig 0.17.3; its source build needs scikit-build-core<0.10.

**Tech stack:** existing Python/NumPy/SciPy/MuJoCo backend and plain English HTML/JS UI. User additionally requested primary-source research; compare path blending, jerk limiting, TOPPRA/TOTG and STOMP in the validation note.

- [x] Add failing motion tests: Classic preservation, Flow moving through travel knots, contact stops/dwells, no overshoot on reversals, limits and finite derivatives, invalid technique rejection.
- [x] Implement `scripts/studio_motion.py`: sampler, analytic derivative extrema, bounded timing and diagnostics. Integrate named waypoints and full-plan placement index in planner; retain Classic default for legacy callers.
- [x] Add validated API plan option and immutable report metadata; failed/new plan invalidates old simulation; mismatched technique cannot execute. Add API tests.
- [x] Add English Motion technique selector in Plan the grasp, explanation, stale-plan lock and result duration/technique.
- [x] Compare Classic/Flow on saved bowl/stack scenes with unchanged geometry and full physics; measure duration, stopping and derivative limits. Test fresh UI and hardware only when scene and gates permit.
- [x] Full regression suite, independent review, evidence and usage documentation.

Review focus: duplicated hold points; reversals/initial slightly out-of-limit pose; stack probe index after retiming; old clients default Classic; selector changes on a successful plan must block execution until regenerated.

Completed: three selectable techniques, primary-source comparison, all regression checks, independent review, fresh physical Flow stack `6257815b1c6f4fa380f67ded78a48ee9` completed 909 commands in 30.30 seconds and returned Home. Evidence in docs/studio-motion-validation-2026-11-07.md.
