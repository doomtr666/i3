def _slangc_impl(ctx):
    outputs = []

    for src in ctx.attr.srcs:
        src_file = src.files.to_list()[0]
        out_file = src_file.basename.replace(".slang", ".spv")
        output = ctx.actions.declare_file(out_file)

        ctx.actions.run(
            inputs = [src_file],
            outputs = [output],
            executable = ctx.executable._compiler,
            arguments = [
                "-matrix-layout-row-major",
                "-profile",
                "glsl_460",
                "-target",
                "spirv",
                "-o",
                output.path,
                src_file.path,
            ],
        )
        outputs.append(output)

    return DefaultInfo(runfiles = ctx.runfiles(files = outputs))

slangc = rule(
    implementation = _slangc_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "_compiler": attr.label(default = "@vcpkg//:slangc", executable = True, cfg = "exec", allow_files = True),
    },
)
