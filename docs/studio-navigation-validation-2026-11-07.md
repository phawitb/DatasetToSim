# Studio navigation and restart validation — 2026-11-07

## Delivered behavior

- English-only Datasets, New Run and Run history pages. Datasets opens a catalog before episodes; the configured catalog currently contains one dataset, SIMCUBE_1O1B_VF (72 episodes). New Run clears the active scene, selection, plan, video, speed and prior activity. Dataset statistics/episodes are absent from New Run; hardware pipeline controls are absent from archive playback.
- Run history includes only live runs with sent-command evidence. Failed preparation, capture-only and simulation-only runs stay out of the catalog. Their files remain available for diagnostics. Failed/interrupted robot executions remain visible.
- Restart server requests a stop, invalidates pending motion dispatch, kills the managed server process group and starts a new instance. A fresh page opens at New Run with hardware disconnected. This requires the HTTP server to receive the restart request; it is not an external recovery service for an unresponsive HTTP listener.
- Executed commands are journaled incrementally so an interrupted recording does not erase the run from history. Journal write/close failures still execute the hold and recording cleanup paths.

## Automated checks

- Full Python suite: **118 passed**, two existing dependency warnings.
- Playback tests: **3 passed**; JavaScript syntax and `git diff --check` passed.
- Regressions cover restart cancellation at dispatch/operation locks, a stubborn descendant process surviving its parent, execution-only history including partial journals, and disk failures during journal write/close.
- Independent review findings were fixed: late restart dispatch, journal cleanup skipping hold, false interruption of an active journal, and a stale asynchronous history navigation response.

## Browser and process checks

- Dataset card → episode list, episode search/selection, playback, camera switching and speed selection.
- New Run hides dataset controls, clears state and restores 1× playback. History displays actual robot executions and distinguishes command completion from verified/failed placement.
- Restart clicked with hardware and warmed SAM workers present. Old server PID 64376, hardware PID 64419 and SAM PID 64976 were all gone after restart; supervisor PID 64375 remained and launched server PID 65100. Browser returned to a fresh disconnected New Run. A second UI restart loaded the reviewed implementation.
- Recorded Home playback tested through play, pause, replay after end, Top/Wrist switching and seeking. No browser console warnings/errors at the final check.
- Screenshots: `output/studio/navigation-validation/`.

## Physical pick/place

Run `3c0bbb7e4b904b6e8f369c150536e39e` used a new capture and user-selected SAM masks (From 93.5%, To 96.7%). The plan passed simulation and transfer checks: 547 commands, 18.23 seconds, maximum IK error 0.110 mm. All 547 commands completed without a hardware or recording error. The saved paired video has 548 frames on the shared command clock.

**Physical placement failed.** Fresh Top and Wrist images show the cube still on the mat and the bowl empty. Recorded frames show the grasp did not retain the cube. This is saved in `physical-review.json` and shown as “Commands complete · placement failed” in Run history. Simulation success is not physical task success. No calibration or contact limits were weakened to make this test pass.

## Physical Home

Immediately afterward, Return Home created run `6936048573974c13849fe21526c49979`. The arm covered the bowl in Top view. Fresh Wrist SAM, the completed release reference and the existing 20 mm bowl clearance envelope passed validation.

All **391 Home commands** completed over **13.03 seconds**, with no hardware/recording error. Final measured joints were within **2.29 native units** of the Home target. The final Top image shows the lowered arm and visible cube/bowl. No Torque off or manual repositioning was used. The paired Home recording contains 392 frames.

The robot was left connected, torque enabled, stopped at Home. These single runs do not establish a general grasp success rate or validate arbitrary objects/workcells.
