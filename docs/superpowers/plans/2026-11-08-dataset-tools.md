# Dataset tools implementation plan

> **For agentic workers:** Use the existing scoped parallel tasks and verify their integration.

**Goal:** Inspect dataset task counts and spatial coverage, build simulation exports from real recordings, and browse actual 3D assets.
**Architecture:** Standalone Python services and JS view modules integrated by the root into the existing server/navigation. Shared visual settings and existing LeRobot exporter/renderer are reused.
**Tech Stack:** FastAPI, MuJoCo, NumPy, vanilla JavaScript/WebGL.
**Spec:** docs/superpowers/specs/2026-11-08-dataset-tools-design.md

## Constraints and review focus
- Preserve all recorded and generated datasets and prior working-tree changes.
- Failed replay episodes must not silently enter successful training exports.
- Use actual source row identities and source video slices, with unchanged native actions.
- Coverage must count inherited episodes and all task IDs; missing poses must be reported.
- Background uploads and numeric settings follow current validation; navigation must not start hardware.

## Tasks
- [x] Coverage service/view: test pose extraction, bounds, counts, inherited rows, and missing data; integrate task counts and click-to-select scatter.
- [x] Replay backend: test source selection, immutable actions/provenance, options, manager lifecycle, export integrity; run a one-episode rendering/export smoke.
- [x] Object catalog: test real geometry and task mappings; integrate responsive local 3D preview page.
- [x] Root integration: dataset replay form with existing appearance controls, upload/preview/start/status; route/lifecycle guards; module and API tests.
- [x] Run Python/JS suites, restart idle server, exercise all new UI paths and inspect screenshots.
