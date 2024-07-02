from collections import namedtuple
import logging

from audiosegment_patch import PatchedAudioSegment as AudioSegment

logger = logging.getLogger('songtwister_effects')

TransformedAudio = namedtuple('TransformedAudio', ['audio', 'updates', 'effect', 'timing_changed'])

def apply_noop(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """Example effect function that does nothing."""
    effect_name = 'noop'
    updates = {}
    transformed_audio = audio
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=False)


def apply_cut(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """TODO"""
    effect_name = 'cut'
    updates = {}
    transformed_audio = audio
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    timing_changed = True
    raise NotImplementedError

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=timing_changed)


def apply_pad(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """TODO"""
    effect_name = 'pad'
    updates = {}
    transformed_audio = audio
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    raise NotImplementedError
    timing_changed = True

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=timing_changed)


def apply_repeat(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """TODO"""
    effect_name = 'repeat'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    times = kwargs.get('times')
    duration = kwargs.get('duration')
    prioritize = kwargs.get('prioritize', 'times')

    time_edited = None
    duration_edited = None

    if times and songtwister._is_int(times):
        logger.info("Looping %s times", times)
        time_edited = audio * float(times)
    if duration:
        duration_ms = songtwister._time_to_ms(duration)
        logger.info("Looping for %s (%s ms)", duration, duration_ms)
        if duration_ms < len(audio):
            duration_edited = audio[duration_ms:]
        else:
            times = int(duration_ms / len(audio)) + 1
            duration_edited = audio * times
            duration_edited = audio[:duration_ms]

    if time_edited and not duration_edited:
        transformed_audio = time_edited
    elif duration_edited and not time_edited:
        transformed_audio = duration_edited
    else: # Both
        transformed_audio = time_edited if prioritize == 'times' else duration_edited

    timing_changed = True

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=timing_changed)


def apply_insert(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """TODO"""
    effect_name = 'insert'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    transformed_audio = audio
    other_audio = audio
    # Extract the targeted audio from songtwister.audio.
    timing_changed = True
    mode = kwargs.get('mode')
    if mode == 'replace':
        # Replace the original audio with the targeted audio
        transformed_audio = other_audio
    elif mode == 'prepend':
        # Put the new audio before.
        transformed_audio = other_audio + transformed_audio
    elif mode == 'overlay':
        # Put it on top. TODO support anchoring, looping, etc.
        transformed_audio = transformed_audio.overlay(other_audio)
        timing_changed = False
    elif mode == 'transform':
        # Crossfade the original into the new
        transformed_audio = transformed_audio.append(
            other_audio, crossfade=len(other_audio))
        timing_changed = False
    else:  # Default: append. # Put the new audio after
        transformed_audio = transformed_audio + other_audio

    raise NotImplementedError

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=timing_changed)


def apply_mute(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """Replace the audio with silence."""
    effect_name = 'mute'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    transformed_audio = AudioSegment.silent(
        duration=len(audio), frame_rate=audio.frame_rate)
    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=False)


def apply_speed(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """Change the playback speed, changing tempo and pitch together.
    This is a fast and primitive operation, changing the framerate in the audio
    rather than transcoding with ffmpeg."""
    effect_name = 'speed'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    transformed_audio = audio
    if 'rate' in kwargs:
        try:
            rate = kwargs.get('rate')
            if isinstance(rate, str):
                pass # TODO Parse percentage and times
            rate = float(rate)
        except ValueError:
            logger.error("Invalid speed rate: %s", rate)
            rate = 1.0
    # TODO: percentage
    timing_changed = False
    # Only process if rate is not the same and rate is between 0 and 10
    if rate != 1.0 and 10 > rate > 0:
        transformed_audio = audio._spawn(
            audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * rate)}).set_frame_rate(
                    audio.frame_rate)
        timing_changed = True
        updates['duration_before'] = len(audio)
        updates['duration_after'] = len(transformed_audio)

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=timing_changed)


def apply_pan(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """Set the audio pan.
      pan: The pan position. Hard left is -100 and hard right is 100.
        Also accepted: 'left', 'right', center.
      pan_to: Optional. Moves the audio from one pan position to another 
        (using a crossfade)"""
    effect_name = 'pan'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    
    def _parse_pan_position(position):
        if isinstance(position, str):
            position = position.lower()
            if position == 'left':
                return -1
            elif position == 'right':
                return 1
            elif position == 'center':
                return 0
        try:
            position = float(position)
        except ValueError:
            logger.warning('Invalid pan position: %s', position)
            position = 0
        position = min(max(position, 100), -100)
        return position / 100

    pan_position = _parse_pan_position(kwargs.get('pan'))
    transformed_audio: AudioSegment = audio.pan(pan_position)
    
    pan_to = kwargs.get('pan_to')
    if pan_to:
        end_position = _parse_pan_position(pan_to)
        if pan_position != end_position:
            transformed_audio = transformed_audio.append(
                seg=audio.pan(end_position),
                crossfade=len(transformed_audio))

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=False)


def apply_reverse(audio: AudioSegment, songtwister, **kwargs) -> TransformedAudio:
    """Reverse the audio. If no 'mode' is specified, the audio is just reversed.
    If mode is 'bounceback', the original audio crossfades into the reversed.
    If mode is 'zoom', the second half of the audio plays in reverse over the
    first half, creating a reverse-delay like effect."""
    effect_name = 'reverse'
    updates = {}
    logger.info("Applying effect: %s on %s with params: %s",
                effect_name, songtwister, kwargs)
    mode = kwargs.get('mode')
    if mode == 'bounceback':
        transformed_audio = audio.append(
            seg=audio.reverse(),
            crossfade=len(audio))
    elif mode == 'zoom':
        half_length = int(len(audio) / 2)
        first_half, second_half = audio[::half_length]
        first_half = first_half.append(
            seg=second_half.reverse(),
            crossfade=(len(second_half)))
        transformed_audio = first_half + second_half
    else:
        transformed_audio = audio.reverse()

    return TransformedAudio(
        audio=transformed_audio, updates=updates,
        effect=effect_name, timing_changed=False)
