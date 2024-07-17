from dataclasses import dataclass, InitVar, field
from typing import Optional, Union, Self
import logging

logger = logging.getLogger('songtwister')


@dataclass
class Duration:
    full_duration: Union[int, float]
    start: Union[int, float, None] = None
    end: Union[int, float, None] = None
    length: Union[int, float, None] = None

    def __post_init__(self):
        if not isinstance(self.full_duration, (float, int)) or self.full_duration <= 0:
            raise ValueError(f"Invalid duration: {self.full_duration}")
        for val in (self.start, self.end, self.length):
            if not isinstance(val, (float, int)) and val is not None:
                raise ValueError(f"Invalid type: {val} is {type(val)}. "
                                 "It must be int, float or None")

        self.start = self._ensure_within_range(self.start, self.full_duration)
        self.end = self._ensure_within_range(self.end, self.full_duration)
        self.length = self._ensure_within_range(self.length, self.full_duration)

        # Match: start, end, duration
        pattern = self._matches()
        if pattern == [False, False, False]: # none are set
            logger.debug("No start, end or length passed - "
                         "using full duration")
            self.start = 0
            self.end = self.full_duration
            self.length = self.full_duration
        elif pattern in ([True, True, False], [True, True, True]): # length missing or all are set
            self.length = self.end - self.start
        elif pattern == [True, False, True]: # end missing
            self.end = min(self.start + self.length, self.full_duration)
        elif pattern == [False, True, True]: # start missing
            self.start = max(self.end - self.length, 0)
        elif pattern == [True, False, False]: # end and length missing
            self.end = self.full_duration
            self.length = self.end - self.start
        elif pattern == [False, True, False]: # start and length missing
            self.start = 0
            self.length = self.end
        elif pattern == [False, False, True]: # start and end missing
            self.start = 0
            self.end = self.length
        else:
            logger.info("Unknown combination. Start: %s, end: %s, length: %s",
                        self.start, self.end, self.length)
        assert self.start + self.length == self.end
        assert self.end - self.start == self.length
        assert self.end - self.length == self.start

    @staticmethod
    def _ensure_within_range(number, max_number):
        if number is None:
            return
        if number < 0:
            number = 0
        elif number > max_number:
            number = max_number
        return number

    def _matches(self):
        return [x is not None for x in (self.start, self.end, self.length)]


@dataclass
class TimeSignature:
    beat_count: Optional[int] = None
    division: Optional[int] = None
    grouping: Optional[list[int]] = None
    notation: Optional[str] = None

    def __post_init__(self):
        # Make sure that grouping is in the correct format, if set
        if self.grouping:
            self.grouping = self.parse_grouping(self.grouping)
        # If beat count is not set, derive it or use default
        if not self.beat_count:
            if self.notation:
                self.beat_count, self.division = self.parse_notation(
                    self.notation, self.division)
            elif self.grouping:
                self.beat_count = self.get_beat_count_from_grouping(self.grouping)
            else:
                self.beat_count = 4

        if not self.grouping or self.get_beat_count_from_grouping() != self.beat_count:
            self.grouping = self.generate_grouping()

        if not self.division:
            self.division = self.generate_division()

        # Make sure that notation is set
        if not self.notation:
            self.notation = self.generate_notation()

        assert self.beat_count and self.division and self.grouping and self.notation

    @staticmethod
    def parse_grouping(grouping: Union[str, list[int]]) -> list[int]:
        if isinstance(grouping, str):
            result = []
            for part in grouping.split():
                # TODO: This excepts if part is not an int.
                # It *should* fail, but probably with a custom exception.
                part = int(part)
                result.append(part)
            return result
        if isinstance(grouping, list) and all([isinstance(x, int) for x in grouping]):
            return grouping
        raise ValueError("Invalid grouping format: "
                         f"{grouping} ({type(grouping)})")

    def get_beat_count_from_grouping(self) -> int:
        grouping = self.parse_grouping(self.grouping)
        return sum(grouping)

    def generate_grouping(self) -> list[int]:
        if self.beat_count <= 4:
            group_size = 1
        elif self.beat_count % 3 == 0:
            group_size = 3
        elif self.beat_count > 16:
            group_size = 4
        elif self.beat_count > 5:
            group_size = 2
        else:
            group_size = 1
        remainder = self.beat_count - (
            int(self.beat_count / group_size) * group_size)
        beat_groups = int((self.beat_count - remainder) / group_size)
        grouping = [group_size] * beat_groups
        if remainder:
            grouping.append(remainder)
        return grouping

    def generate_division(self) -> int:
        """Get the most likely note length division for a given number of beats.
        Eg. 4/4, 3/4, 6/8, 7/8, 12/8, 24/32."""
        if self.beat_count == 6:
            return 8
        division = 32
        while division > 4:
            if self.beat_count > division * 0.75:
                return int(division)
            division = division / 2
        return int(division)

    @staticmethod
    def parse_notation(notation: str,
                       division: Optional[int] = None) -> tuple[int]:
        # TODO: This excepts if part is not an int.
        # It *should* fail, but probably with a custom exception.
        if isinstance(notation, int):
            number = notation
        elif isinstance(notation, str):
            notation_parts = notation.split('/')
            number = int(notation_parts[0])
            if len(notation_parts) >= 2 and not division:
                division = int(notation_parts[1])
        else:
            raise ValueError("Invalid notation format: "
                             f"{notation} ({type(notation)})")
        return number, division

    def generate_notation(self):
        return f"{self.beat_count}/{self.division or 4}"


@dataclass
class BPM:
    bpm: int
    beat_length: Optional[float] = None
    # bar_length: Optional[float] = None
    # beats_per_bar: Optional[InitVar[int]] = None

    def __post_init__(self):#, beats_per_bar):
        if not self.beat_length:
            self._set_beat_length()
        # if beats_per_bar and not self.bar_length:
        #     self._set_bar_length(beats_per_bar)

    def _set_beat_length(self) -> float:
        """Calculate the length of a beat, from the bpm and
        the number of beats per bar."""
        beats_per_second = self.bpm / 60
        milliseconds_per_beat = 1000 / beats_per_second
        # Don't convert this to int
        self.beat_length = milliseconds_per_beat

    # @classmethod
    # def from_bar_length_and_time_signature(cls, bar_length: float, time_signature: TimeSignature) -> Self:


    # def _set_bar_length(self, beats_per_bar: int) -> float:
    #     """Calculate the length of a bar in ms, from the beat length
    #     and the number of beats per bar."""
    #     self.bar_length = self.beat_length_ms * beats_per_bar


@dataclass
class SequenceItem:
    time: Duration
    section: Optional[str] = None
    tags: set[str] = field(default_factory=set)

    def add_tag(self, tag: str) -> None:
        self.tags.add(str(tag).strip())

    def remove_tag(self, tag: str) -> None:
        try:
            self.tags.remove(str(tag).strip())
        except KeyError:
            logger.debug("Tag '%s' could not be removed as it "
                         "was not present. Doing nothing.", tag)


@dataclass(kw_only=True)
class Bar(SequenceItem):
    number: int
    bpm: BPM
    time_signature: TimeSignature

    def convert_to_free_time(self) -> "FreeTime":
        return FreeTime(self.time, section=self.section, tags=self.tags)

    def retime_from_bpm(self, new_bpm: int):
        new_bpm_object = BPM(new_bpm)
        new_duration_object = Duration(
            full_duration=self.time.full_duration,
            start=self.time.start,
            length=new_bpm_object.beat_length * self.time_signature.beat_count
        )
        self.bpm = new_bpm_object
        self.time = new_duration_object

    def offset_time(self, new_full_duration: Union[float, int], offset: Union[float, int, None]):
        """Keep this bar the same length and tempo, but move it earlier or later
        in the full audio length by *offset* milliseconds.
        The new full duration must be passed. If an offset is not passed,
        the difference from the old full duration will be used.
        If the durations are the same, no change will be made."""
        if new_full_duration == self.time.full_duration:
            logger.debug("No offset applied, as before and after duration are the same.")
            return
        if offset is None:
            offset = new_full_duration - self.time.full_duration
        self.time = Duration(
            full_duration=new_full_duration,
            start=self.time.start + offset,
            end=self.time.end + offset
        )

    # def change_beats(self, new_full_duration: Union[float, int], )

    # def change_time(self, new_start=None, new_end=None, new_duration=None):
    #     new_time = 



@dataclass(kw_only=True)
class FreeTime(SequenceItem):
    def convert_to_metered_time(self, number: int, bpm: BPM,
                                time_signature: TimeSignature) -> Bar:
        return Bar(time=self.time, section=self.section, tags=self.tags,
                   number=number, bpm=bpm, time_signature=time_signature)


@dataclass
class BarSequence:
    time: Duration
    bars: list[Bar, FreeTime] = field(default_factory=list)
    bpm: Optional[int] = None
    time_signature: Optional[TimeSignature] = None

    def get_unsequenced(self):
        if not self.bars:
            pass

    def add_free_time(self, length: Union[int, float]):
        pass

    def add_bars(self, number: int = 1, bpm: Optional[int] = None,
                 time_signature: Optional[TimeSignature] = None):
        if not isinstance(number, 1) or int < 1:
            number = 0  # Fill mode
        bpm = bpm or self.bpm
        time_signature = time_signature or self.time_signature
        if not bpm or not time_signature:
            logger.warning("Could create bars due to missing data. BPM: %s. "
                           "Time signature: %s. Defaulting to free time.",
                           bpm, time_signature)


    # @classmethod
    # def from_audiosegment(cls, audiosegment, **kwargs) -> Self:
    #     duration = Duration(full_duration=len(audiosegment))
    #     return BarSequence(time=duration, **kwargs)
