_ZZIP_VERSION = "2409"

def extract_7zip(ctx, src, dst):
    # download older 7-zip
    ctx.download_and_extract(
        url = "https://www.7-zip.org/a/7za920.zip",
        output = "7za920",
    )

    # to unpack newer 7-zip
    ctx.download(
        url = "https://www.7-zip.org/a/7z{0}-x64.exe".format(_ZZIP_VERSION),
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

    ctx.report_progress("extracting 7zip archive {0}".format(src))

    ressult = ctx.execute([
        "seven_zip/7z.exe",
        "x",
        "-y",
        "-o{0}".format(dst),
        src,
    ])

    if ressult.return_code != 0:
        fail("unable to extract 7zip archive", ressult.stderr)
