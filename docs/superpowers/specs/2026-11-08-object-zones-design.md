# Object placement zones

User requests a GitHub checkpoint first, then per-object allowed zones in Generate Data > Build a dataset. Checkpoint pushed as 466a859 on main. Existing design discretion remains applicable; implement in the current app.

Use the existing full calibrated mat 3×3 grid, numbered Z1–Z9 exactly as Workspace coverage (positive world Y up, X right). Defaults allow all zones. Select multiple zones independently for red cube, green cube and bowl. Constraints apply to initial centers, not full paths or final task states. Keep physical footprints on the mat and all existing collision, IK and physics validation. Show the robot-side orientation.

Unstack starts both cubes at one shared XY, so use the intersection of their allowed zones and reject an empty intersection. Add allowed zones for the empty-mat release destination. Bowl constraints apply only to bowl tasks; bowl is absent for stacking/unstacking. Both cube selectors remain applicable to bowl tasks because a second cube is present.

Approaches: free polygon drawing is more complex and does not match coverage; per-role zones change meaning when tasks swap colors; fixed per-object nine-zone selections match the user's objects and existing coverage. Choose the latter as an implementation decision.

Selections feed previews and generation identically, persist in job and episode settings, restore through Add episodes, and display in saved batch settings. When randomization is off, use the fixed reference layout and clearly disable zone controls; submitted settings use unrestricted zones. Balanced stacking distributes new accepted positions among allowed cells only, accounting for inherited counts without moving or excluding existing episodes. Impossible requests fail explicitly without falling back outside allowed zones.

No old dataset changes, no robot execution, no automatic generation of a large batch. Verify request validation, actual sampled scene poses for every task and both unstack colors, balanced quotas including continuations, UI selection/persistence, and a five-scene rendered preview.
