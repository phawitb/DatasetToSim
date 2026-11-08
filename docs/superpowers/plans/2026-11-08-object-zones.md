# Object Zones Implementation Plan

> Implement inline with tests and final review.

**Goal:** Select allowed initial placement zones for each object in Build a dataset.
**Architecture:** Shared validated object-zone mapping converts color identities to sampler roles; existing sampler and accepted-layout balancer enforce the constraints. A small UI module renders editable 3×3 checkbox grids and serializes/restores selections.
**Tech Stack:** Python, NumPy/MuJoCo, vanilla JavaScript/CSS.
**Spec:** docs/superpowers/specs/2026-11-08-object-zones-design.md

## Constraints / review focus
- Default all zones; IDs internally 0–8; same orientation as Workspace coverage.
- Empty, malformed, unknown constraints fail; unstack intersection cannot be empty.
- Color-to-role reversal, bowl distractor cube, unstack shared start/empty destination must be correct.
- Preserved old datasets and fixed-reference behavior; continuation cannot exhaust allowed quotas early.
- Preview and submission share settings; Add episodes restores and displays them.

## Tasks
- [ ] Add failing zone validation/scene/balance tests; implement studio_generation_zones.py, integrate validation and sampling/scene mapping.
- [ ] Extend accepted-layout balancing with allowed source/target cells and continuation quotas; test restricted cells and inherited imbalance.
- [ ] Add frontend zone grid module and tests; integrate settings, restoration, saved summaries, disabled and error states.
- [ ] Run scoped/full tests, restart idle server, click controls and render five-scene preview; verify stored poses and unchanged existing dataset; record evidence and review.
