# Cube stacking in Studio

User requests From cube → To cube as well as bowl placement. Both cubes are confirmed 19.5 mm. Continue the existing local Python/SAM/MuJoCo workflow and English UI.

- Add explicit Bowl / Stack task selector before Capture. Freeze task and bowl-presence choice in the capture. Changing task starts a clean session. Stack selects From, To and (when present) a bowl obstacle with SAM.
- Estimate both cube poses from their masks and measured dimensions. Keep the bowl as an explicit obstacle, never substitute a cube mask into bowl fitting. Preflight refreshes every selected object and checks both cube poses.
- Add a free, physical target cube; trace and render both objects. Keep benchmark replay and bowl behavior unchanged. Stack success requires lifted source, release, stable upright support on the target, contact and an unmoved stable target, not bowl containment.
- Stack trajectory: grasp using aperture-aware approach, lift/transfer, low controlled placement, open, retreat, and validated return corridor to rest. No changes to hardware limits, action hashes or freshness/collision gates.
- Add geometry/model/scoring/API regression tests; replay the user's failed capture offline; test UI task selection and fresh simulation. Only dispatch a newly captured plan that passes all gates. Record physical success separately from simulated/command success.

Completed: implementation and independent review, 127 Python tests and 3 playback tests, fresh three-mask UI capture, successful full simulation and physical run `739d5e1b983a43079151407ff3f50a79` (1,189 commands), automatic Home, Already Home check, and recorded history playback. The physical upper cube has visible overhang; evidence and limitations are in `docs/studio-stack-validation-2026-11-07.md`.
