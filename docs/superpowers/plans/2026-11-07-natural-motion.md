# Natural coordinated motion

**Goal:** Remove unnecessarily orthogonal, staged arm routes while retaining grasp/release constraints and existing motion techniques.

**Research/design:** Modern Robotics 9.3 describes B-splines with approximate control points and convex-hull bounds; 9.1–9.2 separates path from time scaling, and 10.7 requires renewed obstacle validation after path changes. Flash/Hogan 1985 supports smooth hand-space minimum jerk as a human reaching model; joint-space smoothing alone is not a validated human movement model. Implement an inspired coordinated path, not claim an exact human model or CHOMP implementation.

**Approaches:** Merely faster Ruckig preserves the existing compulsory route and cannot address the complaint. Full ROS/CHOMP/STOMP migration is disproportionate for this calibrated local workcell. Select a local quintic B-spline route through soft travel guides with exact task endpoints, derivative-limited timing and existing full contact simulation. Keep old techniques unchanged.

**Architecture:** scripts/studio_natural.py owns C2-or-better rounded macro-motion and diagnostics; studio_planner.py partitions Reach, Grasp, Carry, Settle, Release, Return, Final settle. Natural softens only travel guides, uses early gripper opening for reach, delays closing during return (clearance still requires full simulation), keeps critical endpoint poses and settle times. Fixed 30 Hz, original joint/velocity/acceleration/jerk limits, frozen object/robot physics. Collision verification and full simulation remain mandatory; no silent fallback or modified collision masks.

- [x] Test spline endpoints, continuous derivatives, convex bounds, simultaneous multi-axis motion and grip holds.
- [x] Implement Natural macro phases and analytically bounded timing, metadata, placement timing for stack refinement.
- [x] Add English option and saved history label; preserve selectable Flow/Gentle/Classic and travel speed.
- [x] Compare old/new path shape and complete physics on saved bowl/stack scenes at 100% and 150%.
- [x] Review implementation, run full regression, restart and verify UI/simulation.
- [x] Validate fresh scene before any authorized hardware execution; retain physical evidence separately.

Hardware test completed with failed grasp; evidence and unresolved tracking/calibration discrepancy are recorded in docs/studio-natural-motion-2026-11-07.md. Motion execution and physical task success are reported separately.
