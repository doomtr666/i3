skills:
- update skills / tools to increase productivity
- address model launching twice the program, first time with "cargo run", and be unable to interpret the output. Rerun after with grep or other command variations.
- same thing for vulkan validation errors.
- address model trying to use linux shell commands
- update model rules

doc:
- review projet against the docs the goal is to align the implementation with the design, or update the design doc if needed.

i3_baker:
- is pipeline really used ? 

i3_bundle:
- show information about fragmentation in the archive

i3_io:
- check 64k alignment if really required for mmap, page size is 4k so maybe 4k is sufficient.


next steps:
- pipeline support in baker. system bundle with all i3_renderer pipelines.
- support normal mapping in resolve deferred.
- full GPU oriented, with compute culling, and draw indirect.
- RT support, sync of Accel Structs, ray query shadow.
- Shading DSL on top of pipelines.
- start a blog technical article series (intro, engine rationale, baker, etc)


