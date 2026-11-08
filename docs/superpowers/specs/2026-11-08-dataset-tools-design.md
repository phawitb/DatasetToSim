# Dataset inspection and real replay builder

The user requests per-task counts, spatial coverage, a real-dataset simulation builder, and a 3D asset catalog. Prior user instructions delegate UI/implementation design. Keep the existing local Python/vanilla-JS app and English product copy.

- Visualize data counts accepted episodes for every task; render actual saved initial source/destination positions, with task/layer filters, calibrated workspace bounds, zone counts and episode selection.
- Datasets opens a dedicated replay builder for selected recorded episodes. Replay archived physical traces under the frozen shared calibration and unchanged recorded native actions. Re-render with shared background/appearance/lighting settings. Export regenerated valid LeRobot metadata and source provenance, preserving all source files. Successful replay episodes form training exports; failures are explicit, not relabeled successes. Source rows use actual episode IDs, never stale real metadata ranges.
- 3D-Objects uses actual MuJoCo assets and primitives; local interactive previews and task-role mappings. Do not invent object models or rely on a remote viewer.

Alternative considered: redirect real datasets into synthetic planning. Rejected because it would not preserve the real demonstrations. An overhead scatter is preferable to a perspective point cloud for comparing measured XY coverage.

No hardware movement or publishing is required. The prior 600-episode generation is completed; it must remain intact.
