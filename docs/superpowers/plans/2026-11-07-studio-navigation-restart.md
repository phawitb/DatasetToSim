# Studio navigation, restart and hardware validation

User-provided design: English-only UI, Datasets → dataset → episodes; New Run starts a clean live session; Run history contains only actual robot executions, successful or failed. Add a full server/worker restart and test the prepared workcell. Continue inline using existing Python/plain-JS stack and local hardware; no publishing.

## Implementation
- Separate page selection from video mode and active robot capture. Dataset catalog/selector precedes episodes. Hide dataset metadata and episode controls on New Run/History. Hide robot pipeline when viewing archives.
- Add a supervised server lifecycle: restart request cancels pending dispatch and asks hardware to stop, then supervisor kills only its child process group (server and spawned workers) and starts a fresh instance. Fresh token and explicit reconnect; no automatic motion replay. Capture stopped-run progress before killing when possible.
- Filter history by actual sent-command evidence, including partial/error executions. Exclude failed preparation and draft simulations from history; preserve raw diagnostic files separately from the user-facing catalog.
- Translate all visible UI and dates to English. Do not restore an unrelated stale job into a new page/session.
- Add API/session/catalog/restart/history tests, then browser-check navigation, playback, selection, restart and worker cleanup. Test live Capture → SAM From/To → Plan → Sim → Execute from the user's newly prepared Home pose; inspect physical result and test Home if current evidence passes.

## Review focus
Restart cancellation races; worker descendants left alive; partial executed run lost after restart; stale async UI updates; archive view accidentally enabling robot commands; dataset identity in media caches; English-only empty/error/loading states.

## Validation
Implemented and reviewed. Python suite: 118 passed; playback suite: 3 passed. Browser navigation/playback and full worker restart checked. Actual pick/place completed 547 commands but failed placement; the subsequent Wrist-verified Home completed 391 commands without manual repositioning. See `docs/studio-navigation-validation-2026-11-07.md` for evidence and limits.
