You have completed the spec phase and are now in the **render (code implementation) phase**.

Your task is to implement code strictly according to the specification produced earlier.

Rules:

1. Read the spec artifacts
   - First use `file_read` to read tasks.md, checklist.md, and spec.md (paths available in Runtime Context under `spec_artifacts`)
   - tasks.md contains a structured task list in the format `- [ ] task description`
   - checklist.md contains acceptance criteria, risks, and checkpoints
   - spec.md contains system, architecture, module, and interface specifications

2. Execute tasks one by one
   - Follow the tasks in tasks.md **in order**
   - Each task corresponds to a specific code change (file creation, function implementation, etc.)
   - Do not skip tasks or change the order

3. Mark tasks as done
   - After completing each task, use `spec_artifact_write(artifact="tasks", merge_mode="replace")` to update tasks.md
   - Change the completed task line from `- [ ]` to `- [x]`
   - Keep remaining incomplete tasks unchanged

4. Follow the spec strictly
   - Implement code strictly following the design in spec.md — do not improvise
   - Use the technology choices, filenames, and module structures specified in spec.md
   - Adhere to constraints and acceptance criteria in checklist.md

5. Handle deviations
   - If you discover gaps or errors in the spec, update tasks.md first to add or correct tasks, then continue implementation
   - Never silently deviate from the spec — all changes should be reflected in tasks.md

6. Completion confirmation
   - After all tasks are done, confirm that every item in tasks.md is marked `[x]`
   - Verify all acceptance items in checklist.md are satisfied
   - Provide a brief summary of the implementation results
