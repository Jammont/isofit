"""
Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar
"""

import os
import subprocess
from pathlib import Path

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output, untar

CMD = "sixs"
URL = "https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar"


def precheck():
    """
    Checks if gfortran is installed before downloading SixS

    Returns
    -------
    True or None
        True if `gfortran --version` returns a valid response, None otherwise
    """
    proc = subprocess.run("gfortran --version", shell=True, stdout=subprocess.PIPE)

    if proc.returncode == 0:
        return True

    print(
        f"Failed to validate an existing gfortran installation. Please ensure it is installed on your system."
    )


def build(directory):
    """
    Builds a 6S directory

    Parameters
    ----------
    directory : str
        Directory with an unbuilt 6S
    """
    # Update the makefile with recommended flags
    file = directory / "Makefile"
    with open(file, "r") as f:
        lines = f.readlines()
        lines.insert(3, "EXTRA   = -O -ffixed-line-length-132 -std=legacy\n")

    with open(file, "w") as f:
        f.write("".join(lines))

    # Now make it
    subprocess.run(
        f"make -j {os.cpu_count()}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=directory,
    )


def download(path=None):
    """
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar.

    Parameters
    ----------
    output : str | None
        Path to output as. If None, defaults to the ini path.
    version : str
        Release tag to pull from the github.
    """
    if not precheck():
        print(
            "Skipping downloading 6S. Once above errors are corrected, retry via `isofit download sixs`"
        )
        return

    print("Downloading 6S")

    output = prepare_output(path, env.sixs)
    if not output:
        return

    file = download_file(URL, output.parent / "6S.tar")

    untar(file, output)

    print("Building via make")
    build(output)

    print(f"Done, now available at: {output}")


def validate(path=None, debug=print, error=print, **_):
    """
    Validates a 6S installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    debug : function, default=print
        Print function to use for debug messages, eg. logging.debug
    error : function, default=print
        Print function to use for error messages, eg. logging.error
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with env.validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if path is None:
        path = env.sixs

    debug(f"Verifying path for 6S: {path}")

    if not (path := Path(path)).exists():
        error("[x] 6S path does not exist, please download it via `isofit download 6S`")
        return False

    if not (path / f"sixsV2.1").exists():
        error("[x] 6S does not appear to be installed correctly, please ensure it is")
        return False

    debug("[✓] Path is valid")
    return True


def update(check=False, **kwargs):
    """
    Checks for an update and executes a new download if it is needed
    Note: Not implemented for this module at this time

    Parameters
    ----------
    check : bool, default=False
        Just check if an update is available, do not download
    **kwargs : dict
        Additional key-word arguments to pass to download()
    """
    kwargs.get("debug", print)(
        "SixS does not support versioning at this time, no update to be found"
    )


@cli.download.command(name=CMD)
@cli.path(help="Root directory to download sixs to, ie. [path]/sixs")
@cli.validate
def download_cli(validate_, **kwargs):
    """\
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --sixs /path/sixs download sixs`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sixs --output /path/sixs`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if validate_:
        validate(**kwargs)
    else:
        download(**kwargs)
