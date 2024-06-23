import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from collections import ChainMap
import logging
from datetime import datetime, date
import os

import yaml
from jinja2 import Environment, FileSystemLoader

from songtwister import SongTwister

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger('loading_logger')


def guess_prefix_length(song_object: SongTwister, export_prefix=False,
                        test_bars=[1, 9, 12, 17, 33, 65]) -> None:
    logger.info("Trying to determine the prefix of song '%s'", song_object)
    autoprefix = song_object.detect_prefix()
    if export_prefix:
        song_object.save_audio(song_object.get_prefix(), 'offset')
    sections = []
    notes = [
        f"Autodetected prefix: {autoprefix}"
    ]
    for note in notes:
        logger.info(note)
    for bar_number in test_bars:
        result = song_object.save_bar(bar_number)
        if not result:
            return
        filename, peaks = result
        if isinstance(filename, str):
            file = filename.split('/')[-1]
        elif isinstance(filename, Path):
            file = filename.name
        sections.append({
            'title': f'Bar {bar_number}',
            'file': file,
            'waveform': peaks
        })
    html_file = f"{song_object.stem_filepath}_prefix-test.html"
    make_html(
        output_file=html_file, song=song_object, sections=sections,
        title=song_object.title, notes=notes)


def make_html(output_file, song, sections, template_file=None, title=None,
              notes=None) -> None:
    environment = Environment(loader=FileSystemLoader("."))
    if not template_file:
        template_file = './resources/waveform_template.html.j2'
    if not isinstance(template_file, Path):
        template_file = Path(template_file)
    if not template_file.exists():
        logger.error("Could not export HTML - template file not found at %s",
                     template_file)
    template = environment.get_template()

    content = template.render(
        song=song,
        title=title,
        notes=notes or [],
        sections=sections or [],
    )

    with open(output_file, mode="w", encoding="utf-8") as writer:
        writer.write(content)
        logger.info(f"Wrote {output_file}")


def read_yaml(path: str | Path) -> dict:
    """Read a yaml file or a dir of yaml files"""
    def _load(file) -> dict:
        if file.suffix not in ('.yml', '.yaml'):
            logger.warning(
                "'%s' does not appear to be a yaml file. Skipping", path)
            return {}
        with open(file, 'r') as reader:
            return yaml.safe_load(reader.read())

    if isinstance(path, str):
        path = Path(path)
    assert isinstance(path, Path)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return _load(path)
    elif path.is_dir():
        data = [_load(file) for file in path.iterdir()]
        data = [x for x in data if x is not None]
        skipping = [x for x in data if not isinstance(x, dict)]
        if skipping:
            logger.warning("Skipping invalid preset entries: %s", skipping)
        return dict(ChainMap(*[x for x in data if isinstance(x, dict)]))


def read_json(file: str | Path) -> dict | list:
    with open(file, 'r') as reader:
        return json.loads(reader.read())


def write_json(file: str, content: dict | list) -> None:
    with open(file, 'w') as writer:
        return writer.write(json.dumps(content, indent=2))


def save_song_html(song_object: SongTwister, peaks, filename: str,
                   preset: str, version_name: Optional[str] = None) -> None:
    version = f"{version_name}_" if version_name else ""
    html_file = f"{song_object.stem_filepath}_{version}{preset}.html"
    sections = [{
        'file': filename.split('/')[-1],
        'waveform': peaks
    }]
    make_html(
        output_file=html_file, song=song_object, sections=sections,
        title=song_object.title, notes=[version_name, f"Preset: {preset}"])


def export_song(song_object: SongTwister) -> None:
    """Save the state of the song object, without the AudioSegment,
    to a json file"""
    write_json(
        file=f'{song_object.stem_filepath}.json',
        content=song_object.export_state())


def import_song(filename: str | Path) -> SongTwister:
    """Load the state of the song object, without the AudioSegment,
    from a json file"""
    data = read_json(filename)
    return SongTwister(**data)


def get_args(overwrite_default: bool = False, make_html_default: bool = False,
             verbose_default: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--song", required=True, type=str,
                        help="The name of the song definition, "
                        "set in the songs file")
    parser.add_argument("-p", "--preset", required=False, type=str,
                        help="The preset to use. Presets may be defined "
                        "in the presets file and called here by name. "
                        "If this is not defined, the main set of presets "
                        "from the config will be generated. If * is passed, "
                        "every preset will be generated.")
    parser.add_argument("-c", "--crossfade", required=False,
                        help="Length of the crossfade between cuts in the "
                        "song, overriding values set elsewhere. May be "
                        "an int of the number of milliseconds, or a "
                        "fraction of a bar length. Eg. 1/4, corresponding "
                        "to a full beat in a four beat bar. Default is 1/128.")
    parser.add_argument("-n", "--version-name", required=False, type=str,
                        help="Optionally give the processing of the song "
                        "a version name.")
    parser.add_argument("-g", "--guess-prefix",
                        action="store_true",
                        default=False,
                        help="Guess the number of milliseconds of prefix "
                        "in the song before the beat starts, and generate "
                        "files with single bars to evaluate the guessed value.")
    parser.add_argument("-l", "--make-html",
                        action="store_true",
                        default=make_html_default,
                        help="Generate an HTML page with audio players for "
                        "the exported audio files.")
    parser.add_argument("-a", "--all",
                        action="store_true",
                        default=False,
                        help="Generate the default set of presets "
                        "as defined in the config file.")
    parser.add_argument("-y", "--overwrite",
                        action="store_true",
                        default=overwrite_default,
                        help="Overwrite existing files with same name.")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        default=verbose_default,
                        help="Output detailed logging.")
    return parser.parse_args()


def set_up_logging(logging_config: dict, logging_level: int) -> logging.Logger:
    log_handlers = [logging.StreamHandler()]
    log_to_file = logging_config.get('log_to_file', False)
    if log_to_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / f"songtwister_{date.isoformat(
            datetime.now())}.log"
        log_handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s %(levelname)s [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        handlers=log_handlers,
    )
    return logging.getLogger("run")


def main() -> None:
    global logger
    try:
        config = read_yaml('config.yml')
    except FileNotFoundError:
        logger.error("ERROR: Could not find config file.")
        sys.exit(1)

    locations_config: dict = config.get('locations')
    preferences_config = config.get('preferences')
    html_config = config.get('html_visualization')
    logging_config = config.get('logging')

    make_html_default = html_config.get('generate')
    html_template = html_config.get('template_path')

    logging_level = logging._nameToLevel.get(logging_config.get('level'))
    if logging_level is None:
        raise ValueError(f"Invalid log level: {logging_config.get('level')}")
    verbose_default = logging_level == logging.DEBUG

    overwrite_default = preferences_config.get('overwrite')
    all_presets: dict = read_yaml(locations_config.get('presets'))

    args = get_args(overwrite_default=overwrite_default,
                    make_html_default=make_html_default,
                    verbose_default=verbose_default)

    song_name: str = args.song
    preset_name = args.preset
    crossfade = args.crossfade
    make_all: bool = args.all
    perform_prefix_guess: bool = args.guess_prefix
    version_name: str = args.version_name or ''
    create_html_file: bool = args.make_html
    overwrite: bool = args.overwrite

    logging_level = logging.DEBUG if args.verbose else logging_level
    logger = set_up_logging(logging_config, logging_level)

    all_songs: dict = read_yaml(locations_config.get('song_definitions'))
    song_data = all_songs.get(song_name)
    if not song_data:
        raise ValueError(f'ERROR: Song not found: {song_name}')

    # Set how detailed the peak data of the generated audio will be
    waveform_resolution = html_config.get('waveform_resolution')
    if waveform_resolution and 'waveform_resolution' not in song_data:
        song_data['waveform_resolution'] = waveform_resolution

    try:
        input_file = Path(song_data.get('filename'))
    except TypeError as e:
        logger.error("Could not find input file. Did you write a 'filename' "
                     "in the song definition for '%s'? %s", song_name, e)
        sys.exit(1)
    if not input_file.is_absolute():
        input_file = SCRIPT_DIR / locations_config.get(
            'default_data_path') / input_file
    if not input_file.exists():
        logger.error("Could not find input file at %s. Please check the "
                     "song definition for '%s'", input_file, song_name)
        sys.exit(2)
    song_data['filename'] = input_file

    if perform_prefix_guess:
        song = SongTwister(**song_data)
        guess_prefix_length(song)
        return  # In this case, we quit here

    # Apply a specific preset or the main set defined in config
    if make_all or not preset_name:
        presets_to_apply: list[str] = preferences_config.get('main_preset_set')
        logger.info("Rendering the main preset set.")
    elif preset_name == '*':
        presets_to_apply = list(all_presets.keys())
        logger.info("Rendering every preset.")
    else:
        presets_to_apply = [preset_name]

    # Set crossfade if it has been supplied as an arg
    if args.crossfade:
        crossfade = args.crossfade
        if crossfade.isnumeric():
            crossfade = int(crossfade)

    song = SongTwister(**song_data)
    if 'edit' in song_data:
        song = song.edit(song_data.get('edit'))

    logger.info("Presets that will be applied: %s",
                ', '.join(presets_to_apply))
    for preset in presets_to_apply:
        preset_data = all_presets.get(preset)
        if not preset_data:
            logger.error("Failed to find preset '%s'", preset)
            continue

        # If a specifc crossfade has not been passed, we look in the preset
        if crossfade is None:
            crossfade = preset_data.get('crossfade')
        # If the preset does not have a crossfade defined, we use the default
        if crossfade is None:
            crossfade = preferences_config.get('crossfade')
        song.set_crossfade(crossfade)

        logger.info(
            "Processing '%s' using the preset %s and a crossfade of %s",
            song.title, preset, crossfade)

        preset_effects: list[dict] = preset_data.get('effects')

        # The song config may limit the preset to certain bars
        # (see README for details on selecting)
        if 'bars' in song_data:
            for preset_effect in preset_effects:
                preset_effect['bars'] = song_data.get('bars')

        song.add_effects(preset_effects)

        fade_label = crossfade.replace('/', '-') if isinstance(crossfade, str) else str(crossfade)
        export_version_name = "_".join((
            version_name, preset, f"fade-{fade_label}")).removeprefix('_')
        # TODO: Get the output path first and check that it is can be
        # written to, before generating the audio.
        try:
            edited_audio = song.apply_effects()
            exported = song.save_audio(
                audio=edited_audio,
                version_name=export_version_name,
                overwrite=overwrite
            )
        except (FileExistsError, FileNotFoundError, PermissionError) as e:
            logger.error("Skipping preset due to this error: %s", e)
            continue
        if create_html_file:
            save_song_html(
                song_object=song,
                peaks=exported.peaks,
                filename=exported.filename,
                preset=preset_name,
                version_name=version_name
            )


if __name__ == '__main__':
    main()
