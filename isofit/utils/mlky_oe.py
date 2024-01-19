import logging
import os
import sys

import click
from mlky import Config

Config._opts.convertListTypes = False
Config.Null._warn = False

# Push as a new branch from lut


def getSizeGDAL(file):
    """ """
    return 1, 2

    data = gdal.Open(file, gdal.GA_ReadOnly)
    return data.RasterXSize, data.RasterYSize


def setup_outputs():
    "Setup output directory structure"
    import template_construction

    pn = PathNames(Config)  # creates output names
    pn.mkdirs()  # creates outputs
    pn.stage_files()  # copies some files if needed

    if Config.processors.type_specific_inversion_parameters:
        pn.add_surface_subs_files()

    return pn


def apply_oe():
    """
    Pseudo script providing processing sequence and structure.
    """
    setup_outputs()

    # check sensor settings
    # this is based on lines 116-130 of apply_oe.py to check the value of Config.sensor
    sensor_check(Config.sensor)

    # check settings of empirical and analytical line
    if len(Config.num_neighbors) > 1:
        if Config.empirical_line:
            Logger.error(
                "Empirical Line algorithm cannot be used with greater than 1 num_neighbors"
            )
            return

        if not Config.analytical_line:
            Logger.warning(
                "Analytical Line was not set in the config but should be, enabling now"
            )
            Config.analytical_line = True

    use_superpixels = (Config.empirical_line == 1) or (Config.analytical_line == 1)

    # check input radiance, location, and observation files for consistency in shape
    rdn_size = getSizeGDAL(Config.input_radiance)
    for file_name, file in zip(
        ["input_radiance", "input_loc", "input_obs"],
        [Config.input_radiance, Config.input_loc, Config.input_obs],
    ):
        if os.path.isfile(file) is False:
            err_str = (
                f"Config key {file_name} given as: {file}.  File not found on"
                " system."
            )
            raise ValueError("key " + err_str)
        size = getSizeGDAL(file)
        if size != rdn_size:
            Logger.error(
                f"Input file does not match input radiance size, expected {rdn_size} got {size} for file: {file}"
            )
            return

    # set up in- and output directories
    # this is based on the Pathnames class of apply_oe.py, to be eventually part of template_construction.py
    if Config.copy_input_files == 1:
        Config.copy_input_files = True
    else:
        Config.copy_input_files = False

    from apply_oe import Pathnames

    paths = Pathnames(Config)
    paths.make_directories()
    paths.stage_files()

    # set up LUT configuration
    # this is based on the LUTConfig class of apply_oe.py, to be eventually part of template_construction.py
    from apply_oe import LUTConfig

    lut_params = LUTConfig(Config.lut_config_file)
    if Config.emulator_base is not None:
        lut_params.aot_550_range = lut_params.aerosol_2_range
        lut_params.aot_550_spacing = lut_params.aerosol_2_spacing
        lut_params.aot_550_spacing_min = lut_params.aerosol_2_spacing_min
        lut_params.aerosol_2_spacing = 0

    # get observation metadata, including time and geometry
    # this is based on lines 189-247 of apply_oe.py, including the get_metadata_from_obs() function,
    # to be eventually part of template_construction.py
    get_obs_metadata(paths, lut_params, Config.sensor)

    # get instrument wavelengths
    # this is based on lines 249-275 of apply_oe.py
    get_wavelengths(paths, Config.wavelength_path)

    # get location metadata, including lon/lat and elevation
    # this is based on lines 277-325 of apply_oe.py, including the get_metadata_from_loc() function,
    # to be eventually part of template_construction.py
    get_loc_metadata(paths.loc_working_path, lut_params, Config)

    # set uncorrelated radiometric uncertainty
    if Config.model_discrepancy_path is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = UNCORRELATED_RADIOMETRIC_UNCERTAINTY

    # superpixel segmentation
    # this is based on lines 334-368 of apply_oe.py, including the segment() and extractions() functions,
    # to be eventually part of template_construction.py
    if use_superpixels:
        superpixel_segmentation(paths, Config)

    # water vapor presolve
    # this is based on lines 370-457 of apply_oe.py, including the write_modtran_template(), calc_modtran_max_water(),
    # and build_presolve_config() functions, to be eventually part of template_construction.py
    if Config.presolve == 1:
        wv_presolve(paths, lut_params, Config, **args)

    # main inversion
    # this is based on lines 460-522 of apply_oe.py, including the write_modtran_template() and build_main_config()
    # functions, to be eventually part of template_construction.py
    if (
        not exists(paths.state_subs_path)
        or not exists(paths.uncert_subs_path)
        or not exists(paths.rfl_subs_path)
    ):
        main_inversion(paths, lut_params, Config, **args)

    # empirical or analytical line
    # this is based on lines 524-561 of apply_oe.py
    if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
        line_inference(paths, Config)

    print("Done")


#%%
@click.group(name="apply_oe")
def cli():
    """\
    apply_oe demo using mlky
    """
    ...


@cli.command(name="run")
@click.option(
    "config",
    help="Configuration YAML",
)
@click.argument("input_radiance")
@click.argument("input_loc")
@click.argument("input_obs")
@click.argument("working_directory")
@click.argument("sensor")
@click.option("-p", "--patch", help="Sections to patch with")
@click.option("-d", "--defs", help="Definitions file", default="mlky_oe.defs.yml")
@click.option("--print", help="Prints the configuration to terminal", is_flag=True)
def main(config, patch, defs, print, input_radiance):
    """\
    Executes the main processes
    """
    # Initialize the global configuration object
    Config(config, patch, defs=defs)

    # Accept CLI override, otherwise default to yaml
    Config.input_radiance = input_radiance or Config.input_radiance
    Config.input_loc = input_loc or Config.input_loc
    Config.input_obs = input_obs or Config.input_obs
    Config.working_directory = working_directory or Config.working_directory
    Config.sensor = sensor or Config.sensor

    if print:
        click.echo(Config.dumpYaml())

    # Logging handlers
    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, Config.log.terminal))
    handlers.append(sh)

    if Config.log.file:
        if Config.log.mode == "write" and os.path.exists(Config.log.file):
            os.remove(Config.log.file)

        # Add the file logging
        fh = logging.FileHandler(Config.log.file)
        fh.setLevel(Config.log.level)
        handlers.append(fh)

    logging.basicConfig(
        level=getattr(logging, Config.log.level),
        format=Config.log.format
        or "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt=Config.log.datefmt or "%m-%d %H:%M",
        handlers=handlers,
    )

    if Config.validate():
        apply_oe()
    else:
        click.echo("Please correct any configuration errors before proceeding")


@cli.command(name="generate")
@click.option(
    "-f", "--file", help="File to write the template to", default="mlky_oe.template.yml"
)
@click.option("-d", "--defs", help="Definitions file", default="mlky_oe.defs.yml")
def generate(file, defs):
    """\
    Generates a default config template using the definitions file
    """
    Config(data={}, defs=defs)
    Config.generateTemplate(file=file)
    click.echo(f"Wrote template configuration to: {file}")


if __name__ == "__main__":
    cli()
