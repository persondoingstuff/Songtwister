import random
from typing import Optional, Union, Self
from pathlib import Path
from collections import namedtuple
import logging
from dataclasses import dataclass

# from pydub import AudioSegment
from pydub import effects as pd_effects
from pydub import silence as pd_silence
from pydub.utils import mediainfo

from audiosegment_patch import PatchedAudioSegment as AudioSegment
import effect_functions
import helpers

logger = logging.getLogger("songtwister")

ExportResult = namedtuple("ExportResult", ["filename", "peaks"])
ProcessingResult = namedtuple("ProcessingResult", ["audio", "bpm"])
Excerpt = namedtuple("Duration", ["start", "end", "duration"])
# BeatsPerBar = namedtuple("BeatsPerBar", ["number", ""])



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
                 audio: Optional[AudioSegment] = None,
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
        self.audio = audio
        if not self.audio and load_audio:
            self.load_audio()

        self.bitrate = bitrate or mediainfo(self.filename).get('bit_rate')
        self.prefix_silence_threshold = prefix_silence_threshold

        self.fade_out = fade_out
        self.prefix_length_ms = prefix_length_ms
        self.suffix_length_ms = suffix_length_ms
        self.bar_sequence = bar_sequence or []

        self.beat_length_ms = beat_length_ms or self._get_beat_length()
        self.bar_length_ms = bar_length_ms or self._get_bar_length()
        self.crossfade = 0
        self.set_crossfade(crossfade)
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

    def set_crossfade(self, crossfade: Union[int, str, float]) -> None:
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

    @staticmethod
    def _get_random_id(prefix: str = '') -> str:
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
    def _is_int(value: Union[int, str]) -> bool:
        return isinstance(value, int) or (
            isinstance(value, str) and value.isnumeric())

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
            if criteria.isnumeric():
                criteria = int(criteria)

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
              before #1 or a beat that doesn't exist), it is just
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

    def save_audio(self, audio: Optional[AudioSegment] = None,
                   output_dir: Optional[str | Path] = None,
                   output_format: Optional[str] = None,
                   overwrite: bool = False,
                   version_name: Optional[str] = None,
                   waveform_resolution: Optional[int] = None,
                   extra_parameters: Optional[list] = None) -> ExportResult:
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
        if not audio:
            audio = self.audio
        if self.fade_out:
            audio = audio.fade_out(self.fade_out * 1000)
        logger.info("Writing file: %s", file_path)
        try:
            audio.export(
                out_f=file_path, format=output_format,
                bitrate=self.bitrate, parameters=extra_parameters)
        except PermissionError as e:
            logger.error('Failed to write %s: %s', file_path, e)
            return
        peaks = self._calculate_peaks(audio, waveform_resolution)
        logger.info("Finished writing file")
        return ExportResult(file_path, peaks)

    def _samples_to_ms(self, samples: int, framerate: Optional[int] = None) -> float:
        if not framerate:
            framerate = self.audio.frame_rate
        return (samples / framerate) * 1000

    def _ms_to_samples(self, ms: Union[int, float],
                       framerate: Optional[int] = None) -> int:
        if not framerate:
            framerate = self.audio.frame_rate
        return int((ms / 1000) * framerate)

    def slice(self, start: Union[float, int, None] = None,
              end: Union[float, int, None] = None,
              audio: Optional[AudioSegment] = None) -> AudioSegment:
        """Using the standard slice notation a[1:5], an AudioSegment
        rounds off to whole milliseconds. In many cases this is fine.
        But when dealing with rhythms in music and you are making many
        cuts, loosing the sub-millisecond resolution can add up and
        create offsets in rhythm.
        So this implements a finegrained slicing on sample level instead."""
        if start is not None:
            start_sample = int(self._ms_to_samples(max(start, 0)))
        else:
            start_sample = None
        if end is not None:
            end_sample = int(min(
                self._ms_to_samples(end), self.audio.frame_count()))
            if start_sample is not None:
                end_sample = max(start_sample, end_sample)
        else:
            end_sample = None
        if not audio:
            audio = self.audio
        # NOTE: This does not seem to change anything, so it isn't needed after all.
        # return audio[start:end]
        # print(start_sample, end_sample)
        mine = audio.get_sample_slice(
            start_sample=start_sample, end_sample=end_sample)
        return mine

    def save_excerpt(self, start, end, name: Optional[str]=None,
                     waveform_resolution: Optional[int]=None) -> ExportResult:
        """Write an excerpt of the song audio to file, given a start and end time."""
        if not self.audio:
            self.load_audio()
        if start > self.audio_length_ms or end > self.audio_length_ms:
            raise ValueError("Invalid length", start, end, self.audio_length_ms)
        # excerpt = self.audio[start:end]
        excerpt = self.slice(start, end)
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

    def export_state(self, keep_audio=False) -> dict:
        """Return a dict of all self vars, except the AudioSegment."""
        # If we just do vars(self), we modify self in the following lines
        all_vars = dict(vars(self))
        if not keep_audio:
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

    def make_seq(self):
        sequence = helpers.BarSequence(
            time=helpers.Duration(self.audio_length_ms),
            bpm=self.bpm,
            time_signature=helpers.TimeSignature(self.beats_per_bar)
        )
        # prefix = helpers.Bar()
        return sequence

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
        if not self.audio:
            self.load_audio()
        remainder = self.audio_length_ms - self.prefix_length_ms
        bar_sequence = [{
            'type': 'prefix',
            'bpm': None,
            'start': 0,
            'end': self.prefix_length_ms,
        }]
        bar_number = 1
        current_position = self.prefix_length_ms
        bpm = self.bpm
        while True:
            if self.suffix_length_ms and remainder <= self.suffix_length_ms:
                break
            if remainder < self.bar_length_ms:
                break
            end = current_position + self.bar_length_ms
            bar_sequence.append({
                'type': 'bar',
                'number': bar_number,
                'start': current_position,
                'end': end,
                'bpm': bpm,
            })
            current_position = end
            remainder = remainder - self.bar_length_ms
            bar_number += 1

        self.suffix_length_ms = remainder
        bar_sequence.append({
            'type': 'suffix',
            'bpm': None,
            'start': self.suffix_length_ms,
            'end': self.audio_length_ms,
        })
        self.bar_sequence = bar_sequence
        # TODO: Make a test: This should generate a list of dicts.
        # Each dict should be like this:
        # {'number': int, 'start': int | float, 'end': int | float}

    def get_timeframe_sequence(self, start: Union[int, float, None] = None,
                               end: Union[int, float, None] = None) -> list[dict]:
        pass

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
        bar_numbers = [bar.get('number') for bar in bars if 'number' in bar]
        selected_bars = self.perform_selection(bar_numbers, selection)
        # Then we get the full bar dict for each selected bar
        bars = [bar for bar in bars if bar.get('number') in selected_bars]
        if not bars:
            logger.warning(
                "Could not find any bars matching '%s' out of a total %s bars",
                selection, len(bars))
        return bars

    def get_single_bar(self, selection: Union[int, str]) -> dict:
        """Get a bar dict by its bar number, or another selection criterium"""
        bars = self.get_bars(selection=selection)
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
        prefix_end = max(0, self.prefix_length_ms)
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
        return self.audio[len(self.audio) - self.suffix_length_ms:]

    @staticmethod
    def _is_time(value: str, accept_number: bool = False) -> bool:
        if accept_number and isinstance(value, (float, int)):
            return True
        if isinstance(value, str):
            if ':' in value:
                if all([x.isnumeric()
                        for x in value.replace('.', '').split(':')]):
                    return True
            for suffix in ('ms', 's'):
                if value.endswith(suffix):
                    if value.removesuffix(suffix).isnumeric():
                        return True
        return False

    @staticmethod
    def _time_to_ms(value: str) -> float:
        hours = 0
        minutes = 0
        seconds = 0
        time_parts = str(value).split(':')
        if len(time_parts) == 3: # 00:00:01
            hours = float(time_parts[0])
            minutes = float(time_parts[1])
            seconds = time_parts[2]
        elif len(time_parts) == 2: # 0:01
            minutes = float(time_parts[0])
            seconds = time_parts[1]
        elif len(time_parts) == 1 and time_parts[0].endswith('ms'):
            seconds = "0." + time_parts[0].removesuffix('ms')
        elif len(time_parts) == 1 and time_parts[0].endswith('s'):
            seconds = time_parts[0].removesuffix('s')
        else:
            raise ValueError(f"Unknown timeformat: {value}")
        seconds = float(seconds) if '.' in seconds else int(seconds)
        seconds += (hours * 60 * 60) + (minutes * 60)
        return seconds * 1000

    @staticmethod
    def _get_bars_and_beats(value: str):
        """Numbers (int or float) refer to bars. x/y refers to fractions of a bar (ie. beats).
        """
        parts = str(value).split()
        bars = 0
        bar_fraction = 0
        for part in parts:
            try:
                if '/' in part:
                    a, b = [int(x) for x in part.split('/')]
                    bar_fraction += max(a, 1) / max(b, 1)
                else:
                    part = float(part)
                    bar_number = int(part)
                    bars += bar_number
                    bar_fraction += part - bar_number
            except ValueError as e:
                logger.error("Invalid format: %s. %s", part, e)
        bars += int(bar_fraction)
        bar_fraction = bar_fraction - int(bar_fraction)
        return bars, bar_fraction

    @staticmethod
    def _get_duration(full_duration: Union[float, int],
                      start: Union[float, int, None] = None,
                      end: Union[float, int, None] = None,
                      length: Union[float, int, None] = None) -> Excerpt:
        """full_duration is a length of time. Select a excerpt by passing any
        of offset from the start, offset from the end, and the length of the
        offset. If all three are passed, the longest of end or start + length
        wins.
        All values are floats or ints of milliseconds.
        Returns Excerpt(start, end, duration)
        """
        if not isinstance(full_duration, (float, int)) or full_duration <= 0:
            raise ValueError(f"Invalid duration: {full_duration}")
        if any([not isinstance(x, (float, int, None)) for x in (start, end, length)]):
            logger.error("Invalid value passed. %s (%s), %s (%s), %s (%s)",
                         start, type(start), end, type(end), length,
                         type(length))
            return full_duration

        def _ensure_within_range(number):
            if number is None:
                return
            if number < 0:
                number = 0
            elif number > full_duration:
                number = full_duration
            return number

        def _matches(*a):
            return [x is not None for x in a]

        start = _ensure_within_range(start)
        end = _ensure_within_range(end)
        length = _ensure_within_range(length)

        duration = full_duration
        start_at = 0
        end_at = full_duration
        if _matches(start, end, length) == [False, False, False]:
            logger.warning("No start, end or length passed - "
                           "returning original duration")
            return full_duration
        elif _matches(start, end, length) == [True, True, False]:
            start_at = start
            end_at = end
        elif _matches(start, end, length) == [True, False, True]:
            start_at = start
            end_at = min(start + length, full_duration)
        elif _matches(start, end, length) == [False, True, True]:
            end_at = end
            start_at = max(end - length, 0)
        elif _matches(start, end, length) == [True, False, False]:
            start_at = start
            end_at = full_duration
        elif _matches(start, end, length) == [False, True, False]:
            start_at = 0
            end_at = end
        elif _matches(start, end, length) == [False, False, True]:
            start = 0
            end_at = length
        elif _matches(start, end, length) == [True, True, True]:
            start_at = start
            end_at = end
            max_length = max(start_at + length, full_duration)
            if max_length > end_at:
                end_at = max_length
        else:
            logger.info("Unknown combination. Start: %s, end: %s, length: %s",
                        start, end, length)
        duration = max(end_at - start_at, 0)
        return Excerpt(start_at, end_at, duration)


    def spawn_new_instance(self, new_audio: Optional[AudioSegment] = None, **kwargs):
        if new_audio:
            state = self.export_state()
            state['audio'] = new_audio
            state['audio_length_ms'] = len(new_audio)
            state.pop('bar_sequence', None)
            state.pop('bar_length_ms', None)
            state.pop('beat_length_ms', None)
        else:
            state = self.export_state(keep_audio=True)
        state.update(**kwargs)
        return self.__class__(**state)



    def add_processing(self, effect: dict) -> Self:
        # This should live somewhere else
        effect_index = {
            'test': {
                'function': effect_functions.apply_noop,
                'ffmpeg': False,
                'changes_timing': True,
            },
            'cut': {
                'function': effect_functions.apply_cut,
                'ffmpeg': False,
                'changes_timing': True,
            },
            'pad': {
                'function': effect_functions.apply_pad,
                'ffmpeg': False,
                'changes_timing': True,
            },
            'tempo': {
                'function': effect_functions.apply_noop,
                'ffmpeg': True,
                'changes_timing': True,
            },
            'pitch': {
                'function': effect_functions.apply_noop,
                'ffmpeg': True,
                'changes_timing': False,
            },
            'speed': {
                'function': effect_functions.apply_speed,
                'ffmpeg': False,
                'changes_timing': True,
            },
            'mute': {
                'function': effect_functions.apply_mute,
                'ffmpeg': False,
                'changes_timing': False,
            },
            'pan': {
                'function': effect_functions.apply_pan,
                'ffmpeg': False,
                'changes_timing': False,
            },
            'reverse': {
                'function': effect_functions.apply_reverse,
                'ffmpeg': False,
                'changes_timing': False,
            },
        }

        songtwister = self
        if not songtwister.bar_sequence:
            songtwister.build_bar_sequence()

        print(effect)
        if 'group' in effect:
            effect_list = effect.pop('group')
        else:
            effect_list = [effect]
        effects_to_apply = []
        ffmpeg_effects_to_apply = []
        for effect_item in effect_list:
            effect_name = effect_item.get('do')
            if not effect_name:
                logger.error("Skiping invalid effect: %s", effect_item)
                continue
            effect_function = effect_index.get(effect_name)
            if not effect_function or 'function' not in effect_function:
                logger.error("Unknown effect effect: %s", effect_item)
                continue
            effect_item['function'] = effect_function.get('function')
            if effect_function.get('ffmpeg'):
                ffmpeg_effects_to_apply.append(effect_item)
            else:
                effects_to_apply.append(effect_item)     
        if not effects_to_apply and not ffmpeg_effects_to_apply:
            logger.warning("No valid effects to apply in %s. Doing nothing", effect)
            return songtwister

        crossfade = effect.get('crossfade') or songtwister.crossfade
        crossfade = 0 # Disable for now
        # TODO: Split into before and after
        # And parse crossfade - it may be 1/4 instead of ms. And that is relative to the current bpm

        # Mark out each segment to be processed
        targeted_segments = []


        bar_selection = effect.get('bars')
        beat_selection = effect.get('beats')
        # For now, disable slice targeting
        bar_selection = None
        beat_selection = None
        if bar_selection is None and beat_selection is None:
            # Don't analyze further. Process the entire audio as one and move on
            targeted_segments.append({
                'start_ms': 0,
                'end_ms': songtwister.audio_length_ms
            })
            logger.info("Applying effects to full audio")
        # else:
        #     beats_per_bar = effect_item.get('beats_per_bar') or songtwister.beats_per_bar

        #     if bar_selection and not (isinstance(bar_selection, str) and bar_selection.lower() == 'all'):
        #         #TODO: Perform bar selection. A subset of all the bars will be looked at. 
        #         selected_bars = songtwister.get_bars(selection=bar_selection, bars=songtwister.bar_sequence)
        #     else: # all bars are selected
        #         selected_bars = songtwister.bar_sequence

        #     for bar in selected_bars:
        #         number = bar.get('number')
        #         beat_count = bar.get('beats') or beats_per_bar
        #         beats_in_this_bar = list(range(1, beat_count + 1))
        #         start_ms = bar.get('start')
        #         end_ms = bar.get('end')
        #         bar_length = end_ms - start_ms
        #         beat_length = bar_length / beat_count
        #         segment = {
        #                 'bar_number': number,
        #                 'contains_beats': beats_in_this_bar,
        #                 'beat_count': beat_count,
        #                 'start_ms': start_ms,
        #                 'end_ms': end_ms
        #             }

        #         if beat_selection and not (isinstance(beat_selection, str) and beat_selection.lower() == 'all'):
        #             # TODO: Perform beat selection. A subset of all the beats will be looked at.
        #             selected_beats = self.perform_selection(items=beats_in_this_bar, criteria=beat_selection)
        #             for beat in selected_beats:
        #                 beat_end = beat * beat_length
        #                 beat_start = beat_end - beat_length
        #                 segment['start_ms'] = beat_start
        #                 segment['start_end'] = beat_end
        #                 segment['contains_beats'] = [beat]
        #         else: # selected bars are targeted in their entirity
        #             targeted_segments.append(segment)

        print(effects_to_apply, ffmpeg_effects_to_apply)
        transformed_audio = songtwister.audio
        # Now, each targeted_segment represents a chunk of audio to be processed.
        for segment in targeted_segments:
            logger.info("Segment: %s", segment)
            segment_start = segment.get('start_ms')
            segment_end = segment.get('end_ms')
            # Determine crossfade lengths here and store them for the reassembly
            audio_chunk = songtwister.slice(
                start=segment_start,
                end=segment_end,
                audio=transformed_audio)
            if segment_start:
                audio_before = songtwister.slice(
                    end=segment_start,
                    audio=transformed_audio)
            else:
                audio_before = AudioSegment.empty()
            if segment_end:
                audio_after = songtwister.slice(
                    start=segment_end,
                    audio=transformed_audio)
            else:
                audio_after = AudioSegment.empty()

            updated_params = {}
            applied = []
            # Group edits in ffmpeg based and native ones. Do them separately
            for effect_item in ffmpeg_effects_to_apply:
                logger.info("(Not) Applying ffmpeg effects: %s", effect_item)

            for effect_item in effects_to_apply:
                logger.info("Applying effect: %s", effect_item)
                effect_function: function = effect_item.get('function')
                transformed: effect_functions.Transformed = effect_function(
                    audio=audio_chunk, songtwister=songtwister, **effect_item)
                audio_chunk = transformed.audio
                updated_params.update(transformed.updates)
                applied.append(transformed.effect)
                if transformed.timing_changed:
                    logger.info("Timing changed")
                # Here we hand off to a function that we get from an index.
                # They have a common interface. Take a audio chunk and the songtwister object, it came from.
                # Return a new audio chunk and a description of what changed.
                # Eg. it was removed, had its speed changed, or was repeated - 
                # in which case we will need to adapt the song structure element.

            logger.info("Effects applied: %s", applied)
            logger.info("Updated: %s", updated_params)
            # NOTE!! We need to make the audio chunks longer to match the audio lost with the crossfade
            crossfade_after = crossfade #min(crossfade, len(audio_chunk), len(audio_after))
            audio_chunk = audio_chunk.append(audio_after, crossfade=crossfade_after, dynamic_crossfade=True)
            crossfade_before = crossfade #min(crossfade, len(audio_chunk), len(audio_before))
            transformed_audio = audio_before.append(audio_chunk, crossfade=crossfade_before, dynamic_crossfade=True)

        logger.debug("Length before: %s - after: %s", len(songtwister.audio), len(transformed_audio))
        songtwister = songtwister.spawn_new_instance(new_audio=transformed_audio)
        # Then at the end of this iteration, respawn the object with new attributes and audio.
        # And then the next iteration does it again.
        return songtwister

    # PROCESSING
    def edit(self, edit_list: list) -> Self:
        edit_index = {
            'trim': 'edit_trim',
            'keep': 'edit_keep',
            'cut': 'edit_cut',
            'loop': 'edit_loop',
            'fade': 'edit_fade',
            'process': 'apply_processing',
        }
        edited = self
        for edit_item in edit_list:
            if 'do' not in edit_item:
                logger.error("Invalid edit: %s", edit_item)
                continue
            logger.info("Edit: %s", edit_item.get('do'))
            edit_function_name = edit_index.get(edit_item.pop('do'))
            if not edit_function_name:
                logger.error("Invalid edit action: %s", edit_function_name)
                continue
            edited = getattr(edited, edit_function_name)(**edit_item)
        return edited

    def edit_trim(self, start=None, end=None, keep_prefix=False, keep_suffix=False, **kwargs) -> Self:
        # TODO Support cutting at silence
        # TODO This could be DRYer
        prefix = self.get_prefix() if keep_prefix else AudioSegment.empty()
        suffix = self.get_suffix() if keep_suffix else AudioSegment.empty()
        edited = self.audio
        updated_attrs = {}
        if not self.bar_sequence:
            self.build_bar_sequence()
        if start:
            if self._is_time(start):
                start_trim_length = self._time_to_ms(start)
            elif start == 'prefix':
                start_trim_length = self.prefix_length_ms
                if not prefix:
                    updated_attrs['prefix_length_ms'] = 0
            elif self._is_int(start):
                start_trim_length = self.get_single_bar(start).get('end')
            else:
                raise ValueError(f"Unknown timeformat: {start}")
            start_trim_length = min(start_trim_length, len(edited))
            logger.info("Trimming %s from start", start_trim_length)
            edited = self.slice(start=start_trim_length, audio=edited)
        if end:
            if self._is_time(end):
                end_trim_length = self._time_to_ms(end)
            elif end == 'suffix':
                end_trim_length = self.suffix_length_ms
                if not keep_suffix:
                    updated_attrs['suffix_length_ms'] = 0
            elif self._is_int(end):
                last_bar_number = self.get_single_bar('last').get('number')
                target_bar = self.get_single_bar(last_bar_number - (int(end) - 1))
                end_trim_length = target_bar.get('start')
            else:
                raise ValueError(f"Unknown timeformat: {end}")
            end_trim_length = min(end_trim_length, len(edited))
            logger.info("Trimming %s from end", end_trim_length)
            edited = self.slice(end=end_trim_length, audio=edited)
        logger.debug("After: %s - Before: %s", len(edited), len(self.audio))
        return self.spawn_new_instance(prefix + edited + suffix, **updated_attrs)

    def edit_keep(self, start, end, **kwargs) -> Self:
        edited = self.audio
        if not self.bar_sequence:
            self.build_bar_sequence()

        if self._is_time(start):
            start_position = self._time_to_ms(start)
        elif self._is_int(start):
            start_position = self.get_single_bar(start).get('start')
        else:
            raise ValueError(f"Unknown timeformat: {start}")

        if self._is_time(end):
            end_position = self._time_to_ms(end)
        elif self._is_int(end):
            end_position = self.get_single_bar(end).get('end')
        else:
            raise ValueError(f"Unknown timeformat: {end}")

        logger.info("Keeping from %s to %s", start_position, end_position)
        edited = self.slice(start=start_position, end=end_position, audio=edited)
        logger.debug("After: %s - Before: %s", len(edited), len(self.audio))
        return self.spawn_new_instance(edited)

    def edit_cut(self, start, end=None, length=None, **kwargs) -> Self:
        # A lot of this logic is common "select range" and should be generified

        edited = self.audio
        if not self.bar_sequence:
            self.build_bar_sequence()

        if end is None and length is None:
            logger.warning("Cut requires end (%s) or length (%s) to be set. "
                           "To remove the rest of the audio, use trim instead.",
                           end, length)
            return self

        if self._is_time(start):
            start_position = self._time_to_ms(start)
        elif self._is_int(start):
            start_position = self.get_single_bar(start).get('end')
        else:
            logger.warning("Unknown timeformat for start: %s", start)
            return self

        if length:
            if self._is_time(length):
                length_ms = self._time_to_ms(end)
            elif self._is_int(length):

                length_ms = 0
                # TODO!!!! If a number of bars is supplied, we need to be able to add that to the start bar and get the end time using that
            else:
                logger.warning("Unknown timeformat for length: %s", length)
        if end:
            if self._is_time(end):
                end_position = self._time_to_ms(end)
            elif self._is_int(end):
                end_position = self.get_single_bar(end).get('start')
            else:
                logger.warning("Unknown timeformat for end: %s", end)
                return self
        if end_position <= start_position:
            logger.warning("Could not cut from %s to %s. Skipping.", start_position, end_position)
            return self

        logger.info("Cutting from %s to %s", start_position, end_position)
        before = self.slice(end=start_position, audio=edited)
        after = self.slice(start=end_position, audio=edited)
        edited = before + after
        logger.debug("After: %s - Before: %s", len(edited), len(self.audio))
        return self.spawn_new_instance(edited)

    def edit_loop(self, times=None, duration=None, prioritize='times',
                  keep_prefix=False, keep_suffix=False) -> Self:
        """Loop the audio. Either supply a number of times or a duration that 
        the audio should be looped for.
        If both are supplied, 'prioritize' will determine which one is used.
        Optionally, the prefix and suffix of the audio may be kept."""
        prefix = self.get_prefix() if keep_prefix else AudioSegment.empty()
        suffix = self.get_suffix() if keep_suffix else AudioSegment.empty()
        edited = self.audio
        time_edited = None
        duration_edited = None

        if times and self._is_int(times):
            logger.info("Looping %s times", times)
            time_edited = edited * float(times)
        if duration:
            duration_ms = self._time_to_ms(duration)
            logger.info("Looping for %s (%s ms)", duration, duration_ms)
            if duration_ms < len(edited):
                duration_edited = edited[duration_ms:]
            else:
                times = int(duration_ms / len(edited)) + 1
                duration_edited = edited * times
                duration_edited = edited[:duration_ms]
        
        if time_edited and not duration_edited:
            edited = time_edited
        elif duration_edited and not time_edited:
            edited = duration_edited
        else: # Both
            edited = time_edited if prioritize == 'times' else duration_edited
        return self.spawn_new_instance(prefix + edited + suffix)

    def edit_fade(self, fade_in=None, fade_out=None) -> Self:
        edited = self.audio
        if fade_in:
            if self._is_time(fade_in):
                fade_in_ms = self._time_to_ms(fade_in)
            elif self._is_int(fade_in):
                fade_in_ms = self.bar_length_ms * fade_in
            else:
                raise ValueError(f"Unknown timeformat: {fade_in}")
            fade_in_ms = min(fade_in_ms, len(edited))
            logger.info("Fading in for %s", fade_in_ms)
            edited = edited.fade_in(int(fade_in_ms))
        if fade_out:
            if self._is_time(fade_out):
                fade_out_ms = self._time_to_ms(fade_out)
            elif self._is_int(fade_out):
                fade_out_ms = self.bar_length_ms * fade_out
            else:
                raise ValueError(f"Unknown timeformat: {fade_out}")
            fade_out_ms = min(fade_out_ms, len(edited))
            logger.info("Fading out for %s", fade_out_ms)
            edited = edited.fade_out(int(fade_out_ms))
        return self.spawn_new_instance(edited)

    @staticmethod
    def process_audio(audio: AudioSegment, ffmpeg_parameters: Optional[list] = None, **kwargs) -> ProcessingResult:
        if not ffmpeg_parameters:
            ffmpeg_parameters = []

        bpm = kwargs.pop('bpm', None)
        note_key = kwargs.pop('note_key', None)

        def _interpret(arg: str, pitch=False, bpm=bpm, note_key=note_key):
            if isinstance(arg, (float, int)):
                if pitch: # Interpret as semitones
                    arg = 1 + ((1/12) * arg)
                return float(arg)
            if not arg or not isinstance(arg, str):
                return None
            arg = arg.lower()
            if arg == 'follow':
                return arg
            suffix = None
            for char in ('x', '%', 'bpm'):
                if arg.endswith(char):
                    suffix = char
                    arg = arg.removesuffix(char).strip()
            # notes = "abcdefg".split()
            # if arg[0] in notes and note_key:
            #     note = arg[0]
            #     if len(arg) == 2 and arg[1] in "#b":
            #         accidental = -1 if arg[1] == 'b' else 1
            #     # TODO
            #     logger.warning("The fancy note calculation has not been built yet.")
            #     return 1.0
            try:
                arg = float(arg)
            except ValueError:
                logger.error("'%s' could not be converted to float. "
                            "Falling back to 1.0", arg)
                arg = 1.0
            if suffix == 'x':
                return arg
            if suffix == '%':
                print(arg, (arg / 100))
                return (arg / 100)
            if suffix == 'bpm' and bpm:
                return arg / bpm
            return arg

        pitch = _interpret(arg=kwargs.pop('pitch', None),
                           pitch=True, bpm=None, note_key=note_key)
        tempo = _interpret(arg=kwargs.pop('tempo', None),
                           pitch=False, bpm=bpm, note_key=None)
        follow = 'follow'

        if tempo == follow and pitch == follow:
            logger.warning(
                "No processing applied. Pitch: '%s', tempo: '%s'", pitch, tempo)
            return ProcessingResult(audio, bpm)
        if tempo == follow and pitch:
            tempo = pitch
        elif pitch == follow and tempo:
            pitch = tempo

        new_bpm = round(bpm * tempo, 2) if tempo else bpm

        if tempo:
            ffmpeg_parameters.append(f'rubberband=tempo={tempo}')
        if pitch:
            ffmpeg_parameters.append(f'rubberband=pitch={pitch}')
        return ProcessingResult(audio.process_with_ffmpeg(parameters=[
            f'-af \"{",".join(ffmpeg_parameters)}\"'
            ]),
            new_bpm)

    def apply_processing(
            self, ffmpeg_parameters: Optional[list] = None, **kwargs) -> Self:
        processed = self.process_audio(
                audio=self.audio, ffmpeg_parameters=ffmpeg_parameters, bpm=self.bpm, **kwargs)
        return self.spawn_new_instance(new_audio=processed.audio, bpm=processed.bpm)


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

    def apply_effects(self) -> Self:
        """Effects are first added to a mapping, allowing them to be
        added one at a time. This generates a new SongTwister instance with the
        effects applied."""
        logger.info("Applying effects")
        if not self.audio:
            self.load_audio()
        if not self.bar_sequence:
            self.build_bar_sequence()
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
                # !!audio_since_last_cut = self.audio[
                #     end_of_last_cut - before_fade_length:start_time + after_fade_length]
                audio_since_last_cut = self.slice(
                    end_of_last_cut - before_fade_length, start_time + after_fade_length)
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
                    # NOTE: Consider get_sample_slice() !!
                    # beat_audio = self.audio[start_time:end_time]
                    beat_audio = self.slice(start_time, end_time)
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
                        last_bar = max([x.get('number') for x in self.bar_sequence if 'number' in x])
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
                        # !!target_audio = self.audio[target_start_time:target_end_time]
                        target_audio = self.slice(target_start_time, target_end_time)
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
                            # !!beat_audio = self.audio[start_time:end_time]
                            beat_audio = self.slice(start_time, end_time)

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
        logger.info("Finished applying effects")
        return self.spawn_new_instance(joined_audio)
