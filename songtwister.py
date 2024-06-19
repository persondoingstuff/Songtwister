import random
from typing import Optional, Union
from pathlib import Path
from collections import namedtuple
import logging

from pydub import AudioSegment
from pydub import effects as pd_effects
from pydub import silence as pd_silence
from pydub.utils import mediainfo

logger = logging.getLogger("songtwister")

ExportResult = namedtuple("ExportResult", ["filename", "peaks"])


class SongTwister:
    def __init__(self,
                 filename: str,
                 bpm: int | float,
                 title: Optional[str] = None,  # The filename without extension, if not defined
                 format: Optional[str] = None,  # file format, taken from the file extension
                 stem_filepath: Optional[str] = None,  # path of file, without extension
                 audio_length_ms: Optional[int | float] = None,
                 prefix_length_ms: int = 0,  # Number of ms before the beat or metered part of the song starts
                 suffix_length_ms: int = 0,  # Number of ms after the last bar of the song (to be manipulated)
                 load_audio: bool = True,
                 bar_sequence: Optional[list] = None,
                 peaks=None,
                 waveform_resolution: int = 400,
                 beats_per_bar: int = 4,
                 beat_length_ms: Optional[int] = None,
                 bar_length_ms: Optional[int] = None,
                 crossfade: int | str = 15,
                 crossfade_before: bool = True,
                 crossfade_after: bool = True,
                 bitrate: Optional[int] = None,
                 fade_out: Optional[int | float] = None,
                 prefix_silence_threshold: float = -30.0,
                 **kwargs):
        """Most of the values will rarely be supplied manually when instantiating.
        The mostly exist to be able to export the object state to json and the create
        a new instance from the data, passing them as keyword args."""
        self.additional_data = kwargs
        self.filename = str(filename)  # In case a Path is passed
        self.title = title
        self.format = format
        self.stem_filepath = stem_filepath
        self._set_file_info()

        self.bpm = bpm
        self.beats_per_bar = beats_per_bar

        self.audio_length_ms = audio_length_ms
        self.waveform_resolution = waveform_resolution
        self.audio = None
        if load_audio:
            self.load_audio()

        self.bitrate = bitrate or mediainfo(self.filename).get('bit_rate')
        self.prefix_silence_threshold = prefix_silence_threshold

        self.fade_out = fade_out
        self.prefix_length_ms = prefix_length_ms
        self.suffix_length_ms = suffix_length_ms
        self.bar_sequence = bar_sequence or []

        self.beat_length_ms = beat_length_ms or self._get_beat_length()
        self.bar_length_ms = bar_length_ms or self._get_bar_length()
        if isinstance(crossfade, (int, float)):
            self.crossfade = crossfade
        # crossfade may be given as a note length, eg. 1/16.
        # In this case, it must be a string and must be one fraction of
        # a bar. It does not have to be note lengths -- 1/24 is valid.
        elif isinstance(crossfade, str) and crossfade.startswith('1/'):
            crossfade_beats = int(crossfade.removeprefix('1/'))
            self.crossfade = int(self.bar_length_ms / crossfade_beats)
        else:
            self.crossfade = 0
        self.crossfade_before = crossfade_before
        self.crossfade_after = crossfade_after

    def __repr__(self) -> str:
        return (f"SongTwister: {self.title if self.title else self.filename} "
                f"(Audio {'loaded' if self.audio else 'not loaded'})")

    # PRIVATE METHODS
    def _set_file_info(self) -> None:
        """Derive title, file format and filepath without extension
        from the filename, if the values are not specifically passed."""
        if not self.format:
            self.format = self.filename.split('.')[-1]
        if not self.stem_filepath:
            self.stem_filepath = self.filename.removesuffix(f'.{self.format}')
        if not self.title:
            self.title = self.stem_filepath.split('/')[-1]

    def _get_beat_length(self) -> float:
        """Calculate the length of a beat, from the bpm and
        the number of beats per bar."""
        beats_per_second = self.bpm / 60
        milliseconds_per_beat = 1000 / beats_per_second
        # Don't convert this to int
        return milliseconds_per_beat

    def _get_bar_length(self) -> float:
        """Calculate the length of a bar in ms, from the beat length
        and the number of beats per bar."""
        return self.beat_length_ms * self.beats_per_bar

    def _get_random_id(self, prefix: str = '') -> str:
        """Generate a random id number, with an optional prefix"""
        return f"{prefix}#{str(random.randint(100, 999))}"

    def _calculate_peaks(self, audio: Optional[AudioSegment] = None,
                         waveform_resolution: Optional[int] = None,
                         db_ceiling: int = 100) -> list[int]:
        """Get a list of audio level peaks."""
        if not audio:
            audio = self.audio
        if not waveform_resolution:
            waveform_resolution = self.waveform_resolution
        chunk_length = len(audio) / waveform_resolution

        loudness_of_chunks = [
            audio[i * chunk_length: (i + 1) * chunk_length].rms
            for i in range(waveform_resolution)]

        max_rms = max(loudness_of_chunks) * 1.00

        return [int((loudness / max_rms) * db_ceiling)
                for loudness in loudness_of_chunks]

    @staticmethod
    def perform_selection(items: list[int], criteria) -> list[int]:
        """Select certain items from a list of ints.

        criteria:
        int (or list of ints) selects that item or those items.
        A range() selects a range of items.
        'even' or 'odd' selects alternating items.
        'first' or 'last' selects those items.
        Non-existing items are ignored
        None, 0, 'none', False selects no items -- a pass-through.
        'all' or True selects all items.
        'random X' selects X random items. No number results in a random number of random items.
        'every X of Y' selects item X in each batch of Y, for as many batches as can be made.

        """
        # First some cleanup

        # Make sure that items is an iterable, and not a single int
        if isinstance(items, int):
            items = [items]

        # Make sure that strings are lower for easier comparison
        if isinstance(criteria, str):
            criteria = criteria.lower()

        # Make sure that criteria is iterable
        if isinstance(criteria, (int, range)):
            criteria = (criteria,)

        if not criteria or criteria == 'none':
            # Catch: None, 0, [], '', False, 'none'
            # Return no items
            return []

        if criteria in ['all', True] and not isinstance(criteria, int):
            # Catch 'all' and True (and not 1!)
            # Return all items
            return items

        if criteria == 'even':
            # Return even numbered bars
            return [item for item in items if not item % 2]

        if criteria == 'odd':
            # Return odd numbered bars
            return [item for item in items if item % 2]

        if criteria == 'first':
            # return a list containing the first item, if any
            return [items[0]] if items else []

        if criteria == 'last':
            # return a list containing the last item, if any
            return [items[-1]] if items else []

        # This handles complex string based selections
        if isinstance(criteria, str):
            # Get one or more random items
            if criteria.startswith('random'):
                sample_size = criteria.split()[-1]
                if sample_size and sample_size.isdigit():
                    sample_size = int(sample_size)
                else:
                    # Fallback to a random sample size
                    sample_size = random.randint(1, len(items))
                if sample_size > len(items):
                    logger.warning(
                        "Could not select %s as it is more than the number "
                        "of items, %s. Returning all items",
                        sample_size, len(items))
                    return items
                return sorted(random.sample(items, sample_size))

            # Get "every x of y" selections
            if criteria.startswith('every'):
                parts = criteria.split()
                assert len(parts) == 4 and parts[2] == 'of'
                try:
                    batch_size = int(parts[3])
                    selection = int(parts[1]) - 1
                    if selection > batch_size:
                        logger.warning(
                            "Too large sample size: every %s of %s. "
                            "Original criterium: %s",
                            selection, batch_size, criteria)
                        selection = batch_size
                except ValueError:
                    logger.error("Invalid criterium: %s", criteria)
                    return []
                try:
                    # Select "every x of y":
                    # Divide items into batches of y,
                    # and select item x in each.
                    return [batch[selection] for batch in (
                        items[i:i + batch_size]
                        for i in range(0, len(items), batch_size))]
                except IndexError:
                    logger.error(
                        "Failed to select every %s of %s. criterium: %s",
                        selection, batch_size, criteria)
                    return []

            if '-' in criteria or ',' in criteria:
                # Iterate through at least one range
                # Defined as "1-8" or "1-8, 17-32"
                selection = []
                # If multiple comma-separated selections made, process them
                # one at a time. Otherwise, just process the one.
                for criterium in [x.strip() for x in criteria.split(',')]:
                    # Split at - in case the criterium is a range
                    criterium_parts = criterium.split('-')
                    if len(criterium_parts) == 1 and criterium_parts[0].isnumeric():
                        # It is not a range
                        selected_item = int(criterium_parts[0])
                        if criterium.endswith('-') and 0 < selected_item <= max(items):
                            # Eg. 5-
                            # Select the rest of the song, if the number is
                            # less than or equal to the total number of items
                            selection.extend(list(range(
                                selected_item, max(items) + 1)))
                        elif criterium.startswith('-') and 0 < selected_item <= max(items):
                            # Eg. -5
                            # Selects the beginning of the song, if the number
                            # is less than or equal to the total number of items
                            selection.extend(list(range(1, selected_item + 1)))
                            # FIXME Test me
                        else:
                            # A single bar
                            selection.append(selected_item)
                    elif len(criterium_parts) == 2 and [x.isnumeric() for x in criterium_parts]:
                        # It consists of two number: It is a range.
                        first, last = criterium_parts
                        if first < last:
                            selection.extend(list(
                                range(int(first), int(last) + 1)))
                    else:
                        logger.error(
                            "Invalid selection: %s. Original criteria: %s",
                            criterium, criteria)
                return [item for item in items if item in selection]

        if isinstance(criteria, (list, tuple)):
            # Return specific items from the list.
            # May be a list of ints or ranges or a single one
            # (converted to a list above).
            # Ranges are turned into individual integers.
            selection = []
            for criterium_parts in criteria:
                if isinstance(criterium_parts, int):
                    selection.append(criterium_parts)
                if isinstance(criterium_parts, range):
                    selection.extend(list(range(
                        criterium_parts.start, criterium_parts.stop + 1)))
            # Any int less than 1 is removed.
            # Anything larger than the largest item is removed.
            selection = [i for i in selection if 0 < i <= max(items)]
            # The remaining item numbers are selected from the list of items.
            return [item for item in items if item in selection]
        logger.warning('No valid criteria found')
        return []

    @staticmethod
    def perform_single_selection(selector: str, current: int, first: int,
                                 last: int) -> Optional[int]:
        """Select an int.
            - 'this' selects the current beat or bar number
            - 'next' and 'previous' selects the one before or after
            - 'first' and 'last' selects those
            - 'random' selects any bar or beat
            - A number selects that specific beat or bar
            - If nothing is matched (eg. if you try to select the bar
              before #1 or a beat thad doesn't exist), it is just
              silently ignored
        """
        if selector == 'this':
            selected = current
        elif selector == 'random':
            selected = random.randint(first, last)
        elif selector.isnumeric():
            selector = int(selector)
            if first <= selector <= last:
                selected = selector
            else:
                return
        elif selector == 'first':
            selected = first
        elif selector == 'last':
            selected = last
        elif selector == 'previous':
            if current == first:
                return
            selected = current - 1
        elif selector == 'next':
            if current == last:
                return
            selected = current + 1
        # TODO: Add support for next_n (eg. next_1, next_3) -
        # and the same for previous_n
        return selected

    # PUBLIC METHODS
    def load_audio(self) -> None:
        """Make AudioSegment from the audio file and set the audio length in ms."""
        self.audio: AudioSegment = AudioSegment.from_file(
            file=self.filename, format=self.format)
        self.audio_length_ms = len(self.audio)

    def save_audio(self, audio: AudioSegment,
                   output_dir: Optional[str | Path]=None,
                   output_format: Optional[str]=None,
                   overwrite: bool=False,
                   version_name: Optional[str]=None,
                   waveform_resolution: Optional[int]=None) -> ExportResult:
        """Write an AudioSegment to a file. To prevent overwriting the original,
        if no version_name is passed, a random one is generated."""
        if not version_name:
            version_name = self._get_random_id()
        if not output_format:
            output_format = self.format
        if output_dir:
            if not output_dir(isinstance, Path):
                output_dir = Path(output_dir)
            if not output_dir.exists():
                raise FileNotFoundError(
                    f'Selected output path {output_dir} does not exist.')
            original_name = Path(self.stem_filepath).name
            filename = f"{original_name}_{version_name}.{output_format}"
            file_path = output_dir / filename
        else:
            file_path = Path(
                f"{self.stem_filepath}_{version_name}.{output_format}")
        if not overwrite and file_path.exists():
            raise FileExistsError(f"Cannot write {file_path}, as it already "
                                  "exists and overwriting is not enabled.")
        if self.fade_out:
            audio = audio.fade_out(self.fade_out * 1000)
        logger.info("Writing file: %s", file_path)
        try:
            audio.export(
                out_f=file_path, format=output_format, bitrate=self.bitrate)
        except PermissionError as e:
            logger.error('Failed to write %s: %s', file_path, e)
            return
        peaks = self._calculate_peaks(audio, waveform_resolution)
        return ExportResult(file_path, peaks)


    def save_excerpt(self, start, end, name: Optional[str]=None,
                     waveform_resolution: Optional[int]=None) -> ExportResult:
        """Write an excerpt of the song audio to file, given a start and end time."""
        if not self.audio:
            self.load_audio()
        if start > self.audio_length_ms or end > self.audio_length_ms:
            raise ValueError("Invalid length", start, end, self.audio_length_ms)
        excerpt = self.audio[start:end]
        version_name = name or self._get_random_id("excerpt")
        return self.save_audio(
            audio=excerpt, version_name=version_name,
            waveform_resolution=waveform_resolution)

    def save_bar(self, bar_number: int, waveform_resolution: Optional[int]=None):
        """Save a specific bar to file, given the bar number."""
        try:
            bar = self.get_single_bar(bar_number)
            return self.save_excerpt(
                start=bar.get('start'),
                end=bar.get('end'),
                name=f"bar-{bar_number}",
                waveform_resolution=waveform_resolution)
        except KeyError as e:
            logger.error("Could not find bar number %s. %s", bar_number, e)

    def export_state(self) -> dict:
        """Return a dict of all self vars, except the AudioSegment."""
        all_vars: dict = vars(self)
        all_vars.pop('audio')
        additional_data = all_vars.pop('additional_data')
        all_vars.update(additional_data)
        return all_vars

    def set_new_tempo(self, bpm: float | int) -> None:
        """Change the bpm setting and update derived properties."""
        if not bpm or not isinstance(bpm, (int, float)):
            raise ValueError(f"Invalid bpm value: {bpm} ({type(bpm)})")
        self.bpm = bpm
        self.beat_length_ms = self._get_beat_length()
        self.bar_length_ms = self._get_bar_length()
        self.build_bar_sequence()

    def build_bar_sequence(self) -> None:
        """Create a list of dicts, representing each bar in the song.
        This is used to add effects to, without touching the actual audio,
        and finally used to cut up the audio and apply effects."""
        # Take full length int
        # Take prefix length int
        # Take bar length int
        # Start at prefix point
        # Create items at a bar's length and update the remaining length
        # When the remaining length is less than a bar or equals the suffix length, stop
        # If the suffix length is not set, set it to the remainder
        # Set bars in self.bar_sequence

        remainder: int | float = self.audio_length_ms - self.prefix_length_ms
        bar_sequence = []
        bar_number = 1
        current_position = self.prefix_length_ms
        while True:
            if self.suffix_length_ms and remainder <= self.suffix_length_ms:
                break
            if remainder < self.bar_length_ms:
                break
            end = current_position + self.bar_length_ms
            bar_sequence.append({
                'number': bar_number,
                'start': current_position,
                'end': end
            })
            current_position = end
            remainder = remainder - self.bar_length_ms
            bar_number += 1

        self.suffix_length_ms = remainder
        self.bar_sequence = bar_sequence
        # TODO: Make a test: This should generate a list of dicts.
        # Each dict should be like this:
        # {'number': int, 'start': int | float, 'end': int | float}

    def get_bars(self, selection, bars: Optional[list] = None) -> list[dict]:
        """Extract a selection of bars from a list, or from
        the main bar sequence.
        See the general documentation on making selections to
        see the valid options."""
        if not bars:
            if not self.bar_sequence:
                self.build_bar_sequence()
            bars = self.bar_sequence
        # We perform the selection using a list of bar numbers as ints
        bar_numbers = [bar.get('number') for bar in bars]
        selected_bars = self.perform_selection(bar_numbers, selection)
        # Then we get the full bar dict for each selected bar
        bars = [bar for bar in bars if bar.get('number') in selected_bars]
        if not bars:
            logger.warning(
                "Could not find any bars matching '%s' out of a total %s bars",
                selection, len(bars))
        return bars

    def get_single_bar(self, number: int) -> dict:
        """Get a bar dict by its bar number"""
        bars = self.get_bars(number)
        if not bars:
            raise KeyError
        return bars[0]

    def detect_prefix(self) -> int:
        """Guess the length of the prefix, before the song proper starts,
        based on the leading silence."""
        if not self.audio:
            self.load_audio()
        return pd_silence.detect_leading_silence(
            self.audio, silence_threshold=self.prefix_silence_threshold)

    def set_prefix_and_suffix(
            self, prefix_length_ms: Optional[int | float] = None,
            suffix_length_ms: Optional[int | float] = None) -> None:
        """Reset prefix and suffix lengths in ms.
        Causes bar sequence to be rebuilt."""
        if prefix_length_ms is not None and isinstance(
            prefix_length_ms, (int, float)):
            self.prefix_length_ms = prefix_length_ms
        if suffix_length_ms is not None and isinstance(
            suffix_length_ms, (int, float)):
            self.suffix_length_ms = suffix_length_ms
        self.build_bar_sequence()  # Build or rebuild

    def get_prefix(self) -> AudioSegment:
        """Get the prefix -- the chunk of audio leading up to
        the beginning of the first proper bar."""
        if not self.prefix_length_ms:
            return AudioSegment.empty()
        if not self.audio:
            self.load_audio()
        prefix_end = max(0, self.prefix_length_ms - 1)
        return self.audio[:prefix_end]

    def get_suffix(self) -> AudioSegment:
        """Get the trailing AudioSegment, after all the processable bars.
        If there is nothing, an empty AudioSegment is returned.
        If the audio has not been loaded, it will be. Same if the bars have not been built."""
        if not self.audio:
            self.load_audio()
        if not self.suffix_length_ms:
            # If there is a bar sequence, but no suffix, it is assumed that the song does not have one
            if self.bar_sequence:
                return AudioSegment.empty()
            else:
                self.build_bar_sequence()
        return self.audio[self.suffix_length_ms:]

    def make_loop(self, selection: Union[str, int], duration, keep_prefix=False, fade_out=None):
        def is_time(value):
            return isinstance(value, str) and ':' in value and all(
                [x.isnumeric() for x in value.split(':')])

        if isinstance(selection, int):  # A specific bar number
            start = selection
            end = selection
            time = True
        elif '-' in selection:  # A range of bars or timecodes
            selection_parts = selection.split('-')
            # Select the smallest part and make it an int
            start = int(min(selection_parts).strip())
            # Select the largest part, make it an int and make sure that it
            # is not larger than the total number of bars in the song
            end = min(int(max(selection_parts).strip()),
                      self.get_single_bar('last').get('number'))
            time = is_time(start) and is_time(end)
        
        if time:
            pass
        else:
            start_time = self.get_single_bar(start).get('start')
            end_time = self.get_single_bar(end).get('end')
            bar_count = end - start
            selected_audio = self.audio[start_time:end_time]



    def create_section(self, name: str, start_bar: int, end_bar: int) -> None:
        """TODO: This has not been used yet, and may not work properly.
        Create a section of the song that effects can be applied to,
        enabling different processing for various parts of the song.
        """
        if not self.bar_sequence:
            self.build_bar_sequence()
        if isinstance(start_bar, dict):
            start_bar = start_bar.get('number')
        if isinstance(end_bar, dict):
            end_bar = end_bar.get('number')
        if not start_bar or not end_bar or len(self.bar_sequence) < start_bar or len(self.bar_sequence) < end_bar:
            logger.error(
                'Bad section. Start bar: %s. End bar: %s. Sequence length: %s',
                start_bar, end_bar, len(self.bar_sequence))
            return
        for index, bar in enumerate(self.bar_sequence):
            if bar.get('number') in range(start_bar, end_bar + 1):
                bar['section'] = name
                self.bar_sequence[index] = bar

    def get_section(self, name: str, joined: bool = False) -> dict:
        """TODO: This has not been used yet, and may not work properly."""
        if not self.bar_sequence:
            self.build_bar_sequence()
        name = name.lower()
        section = [
            bar for bar in self.bar_sequence
            if bar.get('section', '').lower() == name]
        if joined:
            return {
                'section': name,
                'start': min([x.get('start') for x in section]),
                'end': min([x.get('end') for x in section]),
            }
        return section

    def add_effect(self, effect: str = 'remove', beats: str = 'last',
                   bars: str = 'all', section: Optional[str] = None,
                   beats_per_bar: Optional[int] = None, **kwargs) -> None:
        """Add an effect to any beats in any bars - default: remove.
        Select a number of bars - default: all.
        If a section is supplied, only bars within this are selected.
        Within each bar, select beats - default: last.
        By default, a bar is divided into the class-wide setting.
        This may be changed by setting number_of_beats.
        """
        if not self.bar_sequence:
            self.build_bar_sequence()
        if beats_per_bar is None:
            beats_per_bar = self.beats_per_bar
        selected_bars = self.bar_sequence
        if section:
            selected_bars = self.get_section(section)
        selected_bars = self.get_bars(selection=bars, bars=selected_bars)
        all_beats = list(range(1, beats_per_bar + 1))
        selected_beats = self.perform_selection(all_beats, beats)
        if kwargs:
            logger.info("These properties were supplied, but not used: %s", kwargs)

        # Divide each bar into beats_per_bar chunks, and add effect on selected_beats
        for bar in selected_bars:
            bar_number = bar.get('number')
            if 'effects' not in bar:
                bar['effects'] = []
            for beat in selected_beats:
                effect_item = {
                    'number': beat,
                    'resolution': beats_per_bar,
                    'effect': effect
                }
                if effect_item not in bar['effects']:  # prevent duplicates
                    bar['effects'].append(effect_item)
            # Replace the bar in the main sequence with the new one with effect added
            for original_bar in self.bar_sequence:
                if original_bar.get('number') == bar_number:
                    self.bar_sequence[self.bar_sequence.index(original_bar)] = bar
                    break

    def add_effects(self, effects: list[dict]) -> None:
        """Add multiple effects in one go."""
        for effect in effects:
            self.add_effect(**effect)

    def _prepare_effects(self) -> dict[str, dict]:
        if not self.bar_sequence:
            # if the sequence has not been generated, there are no effects to apply
            return
        sequence = {}
        for bar in self.bar_sequence:
            bar_effects = bar.get('effects')
            if not bar_effects:
                # We only need to process bars with effects added.
                continue
            new_effects = []
            # Get largest number of bar subdivisions, and adjust all others to match
            new_number_of_beats = max([x.get('resolution', 0) for x in bar_effects])
            for effect in bar_effects:
                resolution = effect.get('resolution', self.beats_per_bar)
                # Number of new beats to get an old beat
                beat_length = int(new_number_of_beats / resolution)
                effect['resolution'] = new_number_of_beats

                old_beat_number = effect.get('number')
                # To get the start position of the beat with a higher resolution,
                # calculate the number of new beats in the beats up to the selected beat,
                # and then add 1 to get the start of the new beat.
                new_beat_number = (beat_length * (old_beat_number - 1)) + 1

                # get number of beats that the beat covers in the new resolution
                for new_beat in range(new_beat_number, new_beat_number + beat_length):
                    new_effect = {k: v for k, v in effect.items()}
                    new_effect['number'] = new_beat
                    new_effects.append(new_effect)

            new_beat_length = self.bar_length_ms / new_number_of_beats
            beat_map = {}
            for effect in new_effects:
                number = effect.get('number')
                if number not in beat_map:
                    beat_map[number] = {
                        'number': number,
                        'start': bar.get('start') + (new_beat_length * (number - 1)),
                        'end': bar.get('start') + (new_beat_length * number),
                        'resolution': effect.get('resolution'),
                        'effects': []
                    }
                beat_map[number].get('effects').append(effect.get('effect'))
                # These effects override others
                if 'remove' in beat_map[number].get('effects'):
                    beat_map[number]['effects'] = ['remove']
                elif 'silence' in beat_map[number].get('effects'):
                    beat_map[number]['effects'] = ['silence']

            sequence[bar.get('number')] = beat_map

            # Now the beat map has each beat that should have
            # effects applied, the slice times, and what effect(s).
            # Remove has highest priority, then silence -- they are exclusive.
            # Other effects are all applied
        return sequence

    def _effect_speedup(self, audio: AudioSegment, speed: float | int = 2,
                        crossfade: int = 150, chunk_size: int = 150,
                        chop_to_length: Optional[str | bool] = None) -> AudioSegment:
        if float(speed) == 1.0:
            return audio
        if len(audio) < 1000:
            chunk_size = 50
            crossfade = min(crossfade, 10)
        elif len(audio) < 1500:
            chunk_size = 100
            crossfade = min(crossfade, 25)
        sped_up_audio: AudioSegment = pd_effects.speedup(
            seg=audio, playback_speed=speed,
            chunk_size=chunk_size, crossfade=crossfade)
        if chop_to_length:
            # The speedup method tends to return a longer AudioSegment than asked for.
            # Here we brutally chop it down to the desired length, sacrificing
            # audio content for the overall timing of the song
            desired_length = len(audio) / speed
            if len(sped_up_audio) > desired_length:
                if chop_to_length == 'end':
                    sped_up_audio = sped_up_audio[:desired_length]
                elif chop_to_length == 'start':
                    sped_up_audio = sped_up_audio[int(len(sped_up_audio) - desired_length):]
                else:
                    padding = int((len(sped_up_audio) - desired_length) / 2)
                    sped_up_audio = sped_up_audio[padding:-padding]
                    # NOTE: This is not entirely precise, but only off at
                    # about 1-2 ms, so we accept it for now.
        return sped_up_audio

    def _effect_speed_change(self, audio: AudioSegment,
                             speed: float | int = 1.0) -> AudioSegment:
        if float(speed) == 1.0:
            return audio
        # Manually override the frame_rate.
        audio_with_altered_frame_rate = audio._spawn(
            audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed)})
        # Convert the sound with altered frame rate to a standard frame rate
        return audio_with_altered_frame_rate.set_frame_rate(audio.frame_rate)

    def _get_effect(self, name: str, effects: list, fallback: int = 1,
                    get_float: bool = False) -> Optional[tuple]:
        matches = [effect for effect in effects if effect.startswith(name)]
        if not matches:
            return
        # We will only apply one speed change effect at a time.
        # Take the last applied
        match = matches[-1].split()
        key = match.pop(0)
        value = match.pop(0) if match else fallback
        try:
            value = float(value) if get_float else int(value)
        except ValueError:
            value = fallback
            logger.error('Not a number: %s', key)
        return key, value

    def apply_effects(self) -> AudioSegment:
        """Effects are first added to a mapping, allowing them to be
        added one at a time. This generates an AudioSegment with the
        effects applied."""
        if not self.audio:
            self.load_audio()
        effect_map = self._prepare_effects()
        if not effect_map:
            logger.warning("No effects to apply - returning original audio.")
            return self.audio

        joined_audio = AudioSegment.empty()
        end_of_last_cut = 0  # ms index in audio where last cut point ended
        # We don't just take values(), so we can sort by the key
        for current_bar_number, bar in sorted(effect_map.items()):
            for cut in sorted(list(bar.values()), key=lambda x: x.get('number')):
                start_time = cut.get('start')
                end_time = cut.get('end')
                cut_duration = end_time - start_time
                beat_effects: dict = cut.get('effects')
                # We use the global crossfade length, unless there is not enough audio
                # before or after.
                if self.crossfade == 0:
                    fade_length = 0
                elif int(end_of_last_cut) < self.crossfade or int(cut_duration) < self.crossfade:
                    fade_length = int(min(end_of_last_cut, cut_duration))
                else:
                    fade_length = self.crossfade
                before_fade_length = fade_length if self.crossfade_before else 0
                after_fade_length = fade_length if self.crossfade_after and 'remove' not in beat_effects else 0
                # Make an AudioSegment of the audio between the last time we made a cut
                # and the beginning of this new cut. In the first iteration, this is from
                # the beginning of the song until the first cut begins. Otherwise, it's
                # the in-between section that we skip, because it doesn't need effects.
                audio_since_last_cut = self.audio[
                    end_of_last_cut - before_fade_length:start_time + after_fade_length]
                # We append the section since the last cut was made to the overall rejoined
                # song. If this is the first iteration, we append to an empty AS.
                # We use the crossfade to smoothen the transition, if the last cut made a
                # big change, like 'remove'. Otherwise, we might get a nasty click or pop.
                # Therefore we extend the audio_since_last_cut section at the beginning with
                # the length of the crossfade, so we have a piece "too much" of the audio
                # before. When we append the audio to the joined_audio, we do a crossfade of
                # the same length. This should make the newly joined audio have the right
                # length.
                joined_audio = joined_audio.append(
                    seg=audio_since_last_cut,
                    crossfade=min(before_fade_length, len(joined_audio)))
                # Right now, it seems that we only do a crossfade when we append "since last cut"
                # to the latest joined audio. That is, the end of a section with effects applied
                # to it, crossfades with the next section of untreated audio.
                # But when we append treated audio to the main joined_audio, it is done without
                # crossfading, resulting in a potentially harsh cut with pops and clicks.
                # This is because the treatments may change the length of the treated section -
                # so it is hard to just add an additional chunk at the start. And we can't just add
                # an untreated pre-section to the start of the treated section after effects,
                # cause that will just cause more pops.
                # It seems like regular crossfading slides section B into the tail of section A
                # by n milliseconds, with a fade out and in at the same time.
                # Could we push that, so section A slides into the start of section B instead?
                # Could we just append a piece of audio from the beginning of section B with
                # the length of the crossfade to the end of section A (the joined_audio),
                # without crossfade, and then do a crossfade between A and B?

                end_of_last_cut = end_time
                if 'remove' in beat_effects:
                    # When removing, we skip the rest of the effects processing, including
                    # appending the segment that we want removed.
                    # The crossfading setting will then smoothen the cut between before and
                    # after this beat.
                    continue

                if 'silence' in beat_effects:
                    # Create a silent audiosegment with the duration of this beat
                    beat_audio = AudioSegment.silent(duration=cut_duration)
                else:
                    # NOTE: Consider get_sample_slice()
                    beat_audio = self.audio[start_time:end_time]
                    beat_length = len(beat_audio)

                    # if 'insert' or 'replace' are in effects we need to find and extract a piece of audio from the full song audio.
                    # 'insert' appends it to the current beat audio. 'replace' removes the beat audio and puts this in instead.
                    # Syntax: insert/replace <bar> <beat>
                    # bar selectors may be: this, next, previous, next_n, previous_n, first, last, or a specific number (int).
                    # If there are invalid values or the selection is out of range, we just move on.
                    # So:
                    # - get the effect and split it out
                    # - if it is valid, parse the bar selection
                    # - parse the beat selection
                    # - determine the sample indices of the requested audio
                    # - either replace beat_audio or join it with the crossfade
                    insert_effect = [x for x in beat_effects
                                     if x.startswith('insert')
                                     or x.startswith('replace')]
                    if insert_effect:
                        # Parse selectors
                        insert_parts = insert_effect[0].split()
                        insert_type: str = insert_parts[0]
                        if len(insert_parts) not in (2, 3):
                            logger.error("Could not parse effect: %s", insert_effect)
                            continue
                        if len(insert_parts) == 2:
                            insert_bar = 'this'
                            insert_beat: str = insert_parts[1]
                        else:
                            insert_bar: str = insert_parts[1]
                            insert_beat: str = insert_parts[2]
                        # Set bar boundaries
                        first_bar = 1
                        last_bar = max([x.get('number') for x in self.bar_sequence])
                        # Select bar
                        selected_insert_bar = self.perform_single_selection(
                            selector=insert_bar, current=current_bar_number,
                            first=first_bar, last=last_bar)
                        if not selected_insert_bar:
                            continue
                        # Select beat
                        beat_count = cut.get('resolution')
                        first_beat = 1
                        last_beat = beat_count
                        current_beat_number = cut.get('number')
                        selected_insert_beat = self.perform_single_selection(
                            selector=insert_beat, current=current_beat_number,
                            first=first_beat, last=last_beat
                        )
                        if not selected_insert_beat:
                            continue
                        logger.debug("in bar %s at beat %s I will %s beat %s from bar %s. Beat count: %s",
                                     current_bar_number, current_beat_number, insert_type,
                                     selected_insert_beat, selected_insert_bar, beat_count)
                        # Fetch bar from sequence and determine beat position
                        target_bar = self.get_single_bar(selected_insert_bar)
                        target_bar_start = target_bar.get('start')
                        target_bar_end = target_bar.get('end')
                        bar_length = target_bar_end - target_bar_start
                        target_beat_length = bar_length / beat_count
                        target_start_time = target_bar_start + (target_beat_length * (selected_insert_beat - 1))
                        target_end_time = target_start_time + target_beat_length
                        # Cut out the audio
                        target_audio = self.audio[target_start_time:target_end_time]
                        # Extend or replace beat audio
                        if insert_type == 'replace':
                            beat_audio = target_audio
                        else:
                            # FIXME support crossfade?
                            beat_audio = beat_audio.append(
                                target_audio, crossfade=0)
                        beat_length = len(beat_audio)

                    speed = self._get_effect(
                        'speed', beat_effects, get_float=True)
                    if speed:
                        # We will only apply one speed change effect at a time.
                        # Take the last applied
                        speed_change_type, speed_rate = speed

                        # In 'fill' mode, we extend the selected audio by that amount.
                        # So that when we speed it up, it retains the overall duration,
                        # it just covers more audio content
                        if 'fill' in speed_change_type and speed_rate >= 1:
                            new_length = beat_length * speed_rate
                            end_time = start_time + new_length
                            beat_audio = self.audio[start_time:end_time]

                        if 'speedup' in speed_change_type and speed_rate >= 1:
                            beat_audio = self._effect_speedup(
                                audio=beat_audio,
                                speed=speed_rate,
                                chop_to_length='x')
                        else:
                            beat_audio = self._effect_speed_change(
                                audio=beat_audio, speed=speed_rate)

                    pitch = self._get_effect('pitch', beat_effects)
                    if pitch:
                        # We will only apply one speed change effect at a time.
                        # Take the last applied
                        pitch_change, pitch_semitones = pitch

                        # NOTE: we only support downpitching currently, and barely that
                        if pitch_change == 'pitchdown':
                            step = (1/12) * pitch_semitones
                            downwards = 1 - ((step) / 2)

                            # First we slow down, with pitch following along
                            # Then we speed back up, preserving the altered pitch
                            crossfade = 50
                            chunk = 150
                            if step > 10:
                                chunk = 20
                                crossfade = 5
                            elif step > 6:
                                chunk = 40
                                crossfade = 20
                            elif step > 5:
                                chunk = 80
                                crossfade = 35
                            elif step > 3:
                                chunk = 100
                            # This is some wonky handheld math with magic numbers.
                            # Could probably be a log scale thing?
                            old_len = beat_length
                            beat_audio = self._effect_speed_change(
                                audio=beat_audio, speed=downwards)
                            down_len = len(beat_audio)
                            beat_audio = self._effect_speedup(
                                audio=beat_audio, speed=down_len / old_len,
                                crossfade=crossfade, chunk_size=chunk)
                            # the speedup result ends up longer
                            # (probably because of chunking and crossfading)
                            # To keep the rythm, we chop down the new
                            # version to retain the timing
                            speed_change_type = len(beat_audio) - old_len
                            if speed_change_type and speed_change_type > 10:
                                half_change = speed_change_type / 2
                                beat_audio = beat_audio[half_change:-half_change]

                    if 'reverse' in beat_effects:
                        beat_audio = beat_audio.reverse()
                    elif 'repeat' in beat_effects:
                        beat_audio = beat_audio.append(
                            beat_audio, crossfade=0)
                    elif 'repeatreverse' in beat_effects:
                        beat_audio = beat_audio.append(
                            beat_audio.reverse(), crossfade=0)
                    elif 'reverserepeat' in beat_effects:
                        beat_audio = beat_audio.reverse().append(
                            beat_audio, crossfade=0)
                    # FIXME These two should be reworked
                    elif 'reverseping' in beat_effects:
                        beat_audio = beat_audio.pan(-1).append(
                            beat_audio.pan(1), crossfade=0)
                    elif 'reversepong' in beat_effects:
                        beat_audio = beat_audio.pan(1).append(
                            beat_audio.pan(-1), crossfade=0)
                    elif 'across left' in beat_effects:
                        beat_audio = beat_audio.pan(1).append(
                            beat_audio.pan(-1), crossfade=beat_length)
                    elif 'across right' in beat_effects or 'across' in beat_effects:
                        beat_audio = beat_audio.pan(-1).append(
                            beat_audio.pan(1), crossfade=beat_length)
                    elif 'bounceback' in beat_effects:
                        beat_audio = beat_audio.append(
                            beat_audio.reverse(), crossfade=beat_length)

                    pingpong = self._get_effect(
                        'pingpong', beat_effects, fallback=2)
                    if pingpong:
                        _, pong_count = pingpong
                        # split here into pingpong_count segments,
                        # and do alternating hard pans on them.
                        segment_length = beat_length / pong_count
                        pong_audio = AudioSegment.empty()
                        for pong_number in range(pong_count):
                            pong_start = segment_length * pong_number
                            pong_end = segment_length * (pong_number + 1)
                            pan = 1 if pong_number % 2 else -1
                            segment = beat_audio[pong_start:pong_end].pan(pan)
                            pong_crossfade = 5 if len(pong_audio) else 0
                            pong_audio = pong_audio.append(
                                segment, crossfade=pong_crossfade)
                        beat_audio = pong_audio
                    elif 'left' in beat_effects:
                        beat_audio = beat_audio.pan(-1)
                    elif 'right' in beat_effects:
                        beat_audio = beat_audio.pan(1)

                if not self.crossfade_after:
                    fade_length = 0
                elif fade_length >= len(joined_audio):
                    fade_length = len(joined_audio)

                joined_audio = joined_audio.append(
                    beat_audio,
                    crossfade=min(fade_length, len(beat_audio)))
        return joined_audio
