# Motion techniques and validation — 2026-11-07

## Research and choice

The original cubic smoothstep restarted at zero velocity at every waypoint. Its acceleration jumped at segment boundaries. Increasing playback speed would not improve the hardware path, and globally shortening timestamps would violate velocity limits.

Primary sources consulted through Firecrawl:

| Technique | What it addresses | Decision for this workcell |
|---|---|---|
| [Path blending / Pilz motion sequences](https://moveit.picknik.ai/main/doc/how_to_guides/pilz_industrial_motion_planner/pilz_industrial_motion_planner.html#sequence-of-multiple-segments) | Joins neighboring motions without stopping at intermediate goals; allows bounded geometric deviation | Use shared nonzero travel-knot velocities and bounded joint paths. Do not blend contact/release boundaries. This implementation is not the Pilz Cartesian-radius algorithm. |
| [Ruckig](https://github.com/pantor/ruckig) | State-to-state trajectories with velocity, acceleration and jerk limits | Flow uses offline local Ruckig sections with shared pass-through velocities. Community intermediate-waypoint calculation uses a cloud API, which is deliberately never invoked. The full chain is not claimed globally time-optimal. |
| [MoveIt TOTG and Ruckig smoothing](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html) | Time parameterization controls velocity/acceleration; Ruckig adds jerk limits. TOTG may change the geometric path within tolerance. | Revalidate complete geometry and contact physics after timing/blending changes. Faster timing alone is insufficient. |
| [TOPPRA](https://github.com/hungpham2511/toppra) | Computes timing along a prescribed path under velocity, acceleration and other constraints | Useful if a future Cartesian spline path must be retained. It does not by itself choose a natural geometric route; current upstream also warns about dropping its Python support. |
| [STOMP](https://moveit.picknik.ai/main/doc/how_to_guides/stomp_planner/stomp_planner.html) | Optimizes a whole path using collision, smoothness and other costs | Appropriate for a later obstacle-aware route optimizer; requires a broader planning integration. Not installed or represented as implemented here. |

Recommendation for this fixed workcell: combine continuous travel with jerk-bounded timing, retain deliberate stops at grasp/release, then verify actual motor tracking. There is no universal “most natural” planner independent of scene, payload and controller. A future Cartesian blend or trajectory optimizer would need new clearance and orientation validation, rather than relaxing existing checks.

## Selectable implementation

In **New Run → Plan the grasp → Motion technique**:

- **Classic — step by step:** original cubic commands and timing. A byte-for-byte regression protects compatibility.
- **Gentle — quintic blend:** C2 position/velocity/acceleration continuity, monotone joint-space Bezier bounds, analytic polynomial derivative extrema. Travel knots can have nonzero velocity; contact knots have zero velocity and acceleration. This option can be slower.
- **Flow — blended S-curve:** Ruckig 0.17.3 local state-to-state sections with shared pass-through velocities. Sections are synchronized to 30 Hz command boundaries. Exact position extrema stay within each waypoint interval; shared velocities are reduced if needed to prevent overshoot. No cloud calls or controller changes.

Both new techniques retain the system's speed ceiling (35 degrees/s on five arm joints, 50 native gripper units/s). The planner uses 92% of those limits. Added planning bounds: acceleration 80 degrees/s² (gripper 120 native units/s²), jerk 600 degrees/s³ (gripper 900 native units/s³). These are conservative planning settings, not identified motor capabilities. Full physical replay still checks tracking, contacts, collision, lift/release and task outcome. Grasp, placement and final settling holds remain present.

The chosen technique, duration, Classic timing estimate, knot indexes and derivative peaks are saved with the plan. Stack placement probes use the same selected technique and the final full-plan knot timing. Changing the UI selection locks simulation/execution for a mismatched plan; the API independently rejects mismatches. Old API callers and saved plans default to Classic. No silent fallback from Flow to another technique is performed.

## Offline physical comparison

Same saved camera/mask scenes, fixed calibration and model parameters, newly planned actions:

| Scene | Classic | Flow | Time reduction | Full physical verdict |
|---|---:|---:|---:|---|
| Bowl, `a4f89f79aa54423f9e784536e55f4fe6` | 22.10 s | 17.67 s | 20.1% | Both success; transfer checks passed |
| Stack, `739d5e1b983a43079151407ff3f50a79` | 39.63 s | 30.60 s | 22.8% | Both success; transfer checks passed |
| Stack, `d9e71cfd16334c16a3bb7c3e67d1ae92` | 39.53 s | 30.43 s | 23.0% | Both success; transfer checks passed |

The preliminary quintic experiment passed all three scenes but took 23.87/39.93/39.30 seconds, motivating the Ruckig option. These preliminary values preceded the final bowl release-boundary stop correction; final Flow bowl was retested and remains 17.67 seconds with success. No speed-limit, friction, collision or success-criterion relaxation was used. Artifacts are in `output/motion-diagnostics/`; those simulation-only captures are not dispatchable live runs.

Independent review identified nonzero velocity at bowl Transfer immediately before opening. Bowl Transfer is now a mandatory stop; stack Transfer remains blendable before its separate lowering/settling stages. The regression checks actual arm command speeds around the release boundary and jaw clearance for all three techniques.

## Installation

Install Studio with the supplied build constraint:

```sh
.venv/bin/python -m pip install --build-constraint requirements-studio-build.txt -e '.[studio]'
```

Ruckig 0.17.3 source builds require the pre-0.10 scikit-build-core configuration. This was built and imported successfully on the project’s macOS ARM64 / Python 3.14 environment. Classic and Gentle do not import Ruckig.

## Fresh physical validation

Run `6257815b1c6f4fa380f67ded78a48ee9` used a new scene with the bowl between two separated cubes. Classic generated a 39.10 s plan; changing to Gentle locked the stale simulation controls; Flow regenerated the same capture into 909 commands / 30.30 s (8.80 s shorter). SAM selected From 93.2%, To 87.4%, bowl 98.0%.

The full Flow simulation passed with a settled supported stack, 0.220 mm target displacement and zero robot-self or robot-workcell contacts. The physical robot completed all 909 commands with no execution error. Recorded Top/Wrist frames show source pickup, release onto the destination cube, an empty withdrawing gripper and the stack still supported after automatic Home. `physical-review.json`, `flow-release.jpg`, `flow-final.jpg` and the paired video retain this evidence separately from simulated success. This is one successful physical test, not a general success rate or a claim of globally optimal/natural motion.

Final automatic verification: 135 Python tests and 3 playback tests passed, with two existing Python dependency deprecation warnings. Independent review was completed; the release-boundary correction was verified. Local UI checks covered three technique options, Classic and Flow generation, mismatch locking, simulation, real execution and recorded playback.
