# Dataset Extension Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development task-by-task.

**Goal:** Extend datasets safely and configure/preview visual diversity.
**Architecture:** Immutable parent-linked generation jobs, seeded renderer options, local image library, isolated preview worker and existing Build/Visualize tabs.
**Tech Stack:** Python FastAPI, MuJoCo, OpenCV/Pillow, native JavaScript, LeRobot v3.
**Spec:** docs/superpowers/specs/2026-11-07-dataset-extension-design.md

## Constraints
English UI; preserve existing recordings and generated data; no physical robot commands; no commits/push in this task; keep generation graphics out of server threads. Preserve train/val groups and snapshot selected image assets. Tests must cover bad upload content, stale previews, failed continuation, changed settings and lineage split stability.

## Tasks
- [x] Backend continuation: studio_generation.py and studio_generation_export.py, tests. Add parent_job_id, inherited accepted episodes, seed continuation, stable split groups, combined export and lineage. Import visual defaults/validation from studio_generation_visuals.
- [x] Visual controls: studio_generation_visuals.py, generation_render.py and real_appearance.py, tests. Shared seeded light setup and selected background handling, same renderer helper for preview/final. Expose defaults/validation.
- [x] UI: web/index.html, app.js, style.css and JS tests. Add episodes prefill Build, cancel extension, adjustable visual controls/library/upload and five-image preview, stale state invalidation.
- [x] Root integration: studio_generation_assets.py and studio_generation_preview.py, server APIs. Safe local uploads, asset snapshots, isolated preview worker with status/media APIs, lifecycle guards.
- [x] Verify unit/JS tests, preview actual graphics, UI upload/select/preview, actual one-episode continuation with old-data/split integrity and official LeRobot reader; save screenshot evidence.
