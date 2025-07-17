def _model_impl(ctx):
    outputs = []

    for src in ctx.attr.srcs:
        src_file = src.files.to_list()[0]

        out_file = src_file.basename.replace(".glb", ".bin")

        output = ctx.actions.declare_file(out_file)

        args = ctx.actions.args()
        args.add("-i", src_file.path)
        args.add("-o", output.path)

        ctx.actions.run(
            inputs = [src_file],
            outputs = [output],
            executable = ctx.executable._compiler,
            arguments = [args],
        )
        outputs.append(output)

    return DefaultInfo(runfiles = ctx.runfiles(files = outputs))

model = rule(
    implementation = _model_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "_compiler": attr.label(default = "//tools/model_processor", executable = True, cfg = "exec", allow_files = True),
    },
)
