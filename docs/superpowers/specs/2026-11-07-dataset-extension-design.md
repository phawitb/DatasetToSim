# Dataset extensions and visual configuration

User-authorized implementation: Add episodes opens Build with editable inherited settings; adjustable randomized lighting/shadows; reusable/uploadable backgrounds; five approximate still previews before generation. UI remains English. Existing local FastAPI/MuJoCo/LeRobot implementation remains the platform.

Chosen approach: immutable combined dataset revisions. A completed parent stays unchanged; a new job copies accepted parent episode artifacts and adds newly validated episodes. Export combines them, preserves existing trajectory-group train/val assignments, regenerates metadata/statistics, and records lineage. Failure never replaces the old dataset. Defaults inherit the parent; the seed advances beyond previously consumed seeds. User may edit all ordinary generation settings; continuation seed must avoid ancestral attempted seeds.

Alternative considered: overwrite parent in place. Rejected because a cancelled/failed generation could damage existing training data. Independent jobs without combining do not meet Add episodes semantics.

Visual options: explicit light intensity, azimuth, elevation, random angular spread and shadow strength. Seeded coherent lighting draws per scene; motion and physics unchanged. Built-in calibrated background plus locally uploaded PNG/JPEG/WebP library, size/pixel limits and normalized safe image storage. Uploaded images are mapped as a workcell background plate and labeled uncalibrated; no false calibration claims. Foreground remains rendered geometry.

Preview: asynchronous local graphics worker renders five seeded initial-layout Top/Wrist still pairs using the same visual pipeline as final rendering. No IK/hardware movement. Label as appearance/layout preview; generation still validates physical feasibility. Changed settings invalidate stale previews; don't silently show results for old settings. Preview and dataset workers are bounded/cancellable through server lifecycle.
