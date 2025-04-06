_DEFAULT_NAME = "seven_zip"
_7ZIP_LABEL = "@seven_zip//:seven_zip/7z.exe"

def extract_7zip(ctx, src, dst):

    ctx.report_progress("extracting 7zip archive {0}".format(src))

    """Extracts the 7zip archive to the destination directory."""
    ressult = ctx.execute([
        Label(_7ZIP_LABEL),
        "x",
        "-y",
        "-o{0}".format(dst),
        src,
    ])
    if ressult.return_code != 0:
        fail("unable to extract 7zip archive", ressult.stderr)
        
def _seven_zip_rule_impl(ctx):
    # download older 7-zip
    ctx.download_and_extract(
        url = "https://www.7-zip.org/a/7za920.zip",
        output = "7za920",
    )

    # to unpack newer 7-zip
    ctx.download(
        url = "https://www.7-zip.org/a/7z{0}-x64.exe".format(ctx.attr.version),
        output = "7z.exe",
    )

    result = ctx.execute([
        "7za920/7za.exe",
        "x",
        "-y",
        "-oseven_zip",
        "7z.exe",
    ])

    if result.return_code != 0:
        fail("unable to unpack seven_zip package", result.stderr)

    # generate build file
    ctx.file(
        "BUILD",
        content = "",
    )

seven_zip_rule = repository_rule(
    local = True,
    attrs = {
        "version": attr.string(mandatory = True),
    },
    implementation = _seven_zip_rule_impl,
)

def _seven_zip_impl(ctx):
    tags = ctx.modules[0].tags.install[0]
    seven_zip_rule(name = tags.name, version = tags.version)

seven_zip = module_extension(
    implementation = _seven_zip_impl,
    tag_classes = {
        "install": tag_class(attrs = {
            "name": attr.string(
                doc = "Base name for generated repositories",
                default = _DEFAULT_NAME,
            ),
            "version": attr.string(
                doc = "Version of seven_zip archiver",
            ),
        }),
    },
)