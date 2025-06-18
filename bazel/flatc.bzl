def _flatc_impl(ctx):
    outputs = []
    bin_dir = ctx.bin_dir.path

    src_files = []
    for src in ctx.attr.srcs:
        src_file = src.files.to_list()[0]
        src_files.append(src_file)

    for src in ctx.attr.srcs:
        src_file = src.files.to_list()[0]

        out_dir = "fbs/"
        out_file = out_dir + src_file.basename.replace(".fbs", "_generated.h")

        flatc_output_dir = bin_dir + "/" + ctx.label.package + "/" + out_dir
        output = ctx.actions.declare_file(out_file)

        flatc_args = ctx.actions.args()
        flatc_args.add("--cpp")
        flatc_args.add("-o")
        flatc_args.add(flatc_output_dir)
        flatc_args.add(src_file.path)

        # run flatc
        ctx.actions.run(
            executable = ctx.executable._compiler,
            arguments = [flatc_args],
            inputs = src_files,
            outputs = [output],
            progress_message = "flatc " + src_file.path,
        )

        outputs.append(output)

    compilation_context = cc_common.create_compilation_context(headers = depset(outputs), includes = depset([bin_dir + "/" + ctx.label.package, bin_dir]))
    return CcInfo(compilation_context = compilation_context)

flatc = rule(
    implementation = _flatc_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "_compiler": attr.label(default = "@vcpkg//:flatc", executable = True, cfg = "exec", allow_files = True),
    },
)
