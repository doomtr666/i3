def _slangc_impl(ctx):
    outputs = []

    for src in ctx.attr.srcs:
        src_file = src.files.to_list()[0]
        out_file = src_file.basename.replace(".slang", ".spv")
        output = ctx.actions.declare_file(out_file)

        args = ctx.actions.args()
        if ctx.var["COMPILATION_MODE"] == "opt":
            args.add("-O2")
        else:
            args.add("-O0")
            args.add("-g2")

        args.add("-matrix-layout-row-major")
        args.add("-profile", "glsl_460")
        args.add("-target", "spirv")
        args.add("-o", output.path)
        args.add(src_file.path)

        ctx.actions.run(
            inputs = [src_file],
            outputs = [output],
            executable = ctx.executable._compiler,
            arguments = [args],
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
