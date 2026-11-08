# Balanced stacking collection to 600 episodes

## Requested target and recovery

User requested 300 red-on-green and 300 green-on-red accepted demonstrations, distributed roughly equally over many workspace positions.

Failed source job: `b4db8b1b5edc4205bb36135a6c7f61f8` (`so101-stack-400`). It requested 211 total rather than 400: 11 inherited plus 100 new per task. At failure it held 45 accepted episodes (40 red-on-green, 5 green-on-red), after 70 new attempts and 35 journaled rejects. The fatal error was `A task cube is never visible in the rendered reference views` in the unjournaled final attempt. Physical rejection reasons were principally target movement/tipping and unreachable or nonconvergent planning.

Recovered all45 accepted episodes into independent completed/audited checkpoint `d658311f1b784db080de93a2060a9c83`, 20,164frames, 36train/9validation. All1,048 snapshotted source files remained unchanged. Recovery preserved prior inherited group splits and reused exact accepted artifacts.

Started persisted job `28808b6f1afb4044bf978d23e994f766`, name `so101-stack-600`, with total task quotas300/300, inherited45, deficits260/295, seed100. It retains Natural, recorded pace, 150% requested travel speed and the failed job's selected real background/lighting settings. Idle-sleep prevention is bound to the worker lifetime. No robot commands were issued.

## Changes retained

- Known per-layout invisible-cube render error is journaled as rejection and retried. Disk/encoder/configuration errors remain fatal and visible.
- Optional `Balanced workspace zones` strategy partitions the full calibrated mat into a3×3 grid. Source and destination marginal counts are based on accepted episodes, including inherited data. Each task targets33–34 percell in each role; no zone is removed for being hard. Quota selection preserves an off-diagonal feasible assignment so easier zones cannot consume capacity and leave only a difficult same-zone pair at the end.
- Initial cube separation80mm for conditioned stacking proposals, versus the old cube-only margin ~36mm; calibrated open pad-center span is58–65mm. This is an initial-layout clearance rule, not a swept-volume certificate.
- Cheap bounded IK screens calibrated source/placement waypoints before expensive planning. All raw filtered proposals, reasons and waypoint errors are retained. Every final candidate still undergoes full contact physics and original success/collision/tracking gates.
- Conservative target-placement pitch-risk preference: reject proposed target-lowering IK pitch residual above20degrees, with at most128 resamples in the same requested cells. This is an empirical preference, not proof of impossibility. It would exclude3 of45 historically valid layouts (seeds62,68,88), so no claim is made that it retains all physically possible poses. All45 existing accepted examples remain included unchanged.
- Explicit API total quotas avoid adding300 to each inherited task accidentally. Ordinary Add episodes remains additive unless the caller explicitly supplies `target_counts`. Count input supports up to1,000 per task. UI shows accepted/target totals for each task and source/destination zone counts in batch settings.

## Evidence

- Initial IK screen retained16/16 successes and filtered7/7 known IK failures in36 saved trials. Warm screening ~28ms/layout.
- Difficult target cell2 baseline20 trials:1success,16target movement/tipping,3planning failures. Target displacement begins during lowering; held cube and sometimes jaw geometry push the lower cube.
- An experimental synthetic-only vertical-descent guard reduced command-space lateral sweep but still gave1/20 in that difficultcell and regressed2/16 prior successes. All experimental planner/natural changes were reverted. No acceptance criterion was loosened.
- Final pitch-filter probe:24/36 accepted (66.7%), physical successes in all9 target cells, percell[3,4,3,3,1,3,3,2,2]. Cells2/8 achieved3/4 and2/4. The probe filtered295 cheap candidates before fullplanning; remaining failures were8movement/tipping and4planning errors. The original36 benchmark used100% speed and passed16/36; final probe used150%, so the total difference is not an isolated estimate of the pitch filter's effect.
- Rejected observations never enter the export. Remaining physics rejections are expected; no global production acceptance rate is promised from this small targeted probe.

## Runtime and verification

This is a long-running local collection. Starting the job is not evidence that600 episodes are complete. The worker exports only after both accepted-task quotas are met; a bounded retry limit or system failure remains visible as a failed job with accepted artifacts retained. Restarting Studio cancels active work; the separate recovered45 checkpoint remains available.

Evidence and full test logs: `output/balanced-600-check/`. Important files: recovery.log, start-options.json, worker-process.json, pitch-filter/summary.json, python-tests.txt, final-python-tests.txt and javascript-tests.txt. Final completion must be checked in job.json and the600-episode export audit, including task totals and all four spatial marginals.

Final verification:277Python tests passed (two dependency warnings),30JavaScript tests passed; git diff whitespace check clean. Live UI confirmed46/600 accepted,41/300red-on-green and5/300green-on-red, with the next episode rendering and no job errors. First newly accepted example uses sourcecell3/destinationcell2, which had no inherited target examples. Screenshot:output/balanced-600-check/running-600.png. These are launch-time counts, not final completion.
