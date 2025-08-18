# Project TODO

A simple, in-repo task list using GitHub-friendly checkboxes. Keep it short and actionable.

## Conventions
- One line per task: use checkboxes, a short title, and optional metadata.
- Priority: [P0] critical, [P1] high, [P2] normal, [P3] low.
- Owner: @github-handle. Optional due date: YYYY-MM-DD.
- Link to issues/PRs if they exist to avoid duplication.
- Move completed items to “Done (recent)” at the bottom; prune periodically.

Task line format:
- [ ] [P1] Short task title — @owner (due: 2025-09-01) #123

## Now
- [ ] [P1] Add the option to carry out analysis of a user-defined shapefile. For example, we want to upload a shapefile which consists of 2 features, i.e. the Ala Archa basin outline and the Alamedin basin outline. The EXZECO analysis should then be carried out on the entire joint domain, that is,  statistics needs to be reported for the entire domain and the individual subcatchments. This must hold of an arbitrary number of subcatchments for an arbitrary shapefile given. If the shapefile location is not specified in the config.yml or the file that is specified is not found, the bounding box approach should be chosen. If not shapefile and bounding box is available, throw and error. — @owner

## Next
- [ ] [P2] Interactive 3D visualization does not really show any results. — @owner

## Backlog
- [ ] [P2] Consider exporting static snapshots for PDF (maps/3D) — @owner
- [ ] [P3] Add usage examples for CLI/run script — @owner
- [ ] [P1] Review and revise the report.qmd — @owner

## Done (recent)
- [x] [P1] Configure Git LFS for large notebooks — @owner (2025-08-17)
