"""Some commits have been made to the pydub master branch that have not been included in a release.
One of these vastly improves performance (using memory instead of temp files on disk).

This patches the improvements in.
"""
import os
from typing import Self
import sys
import wave
import subprocess
from io import BytesIO, BufferedReader
from tempfile import NamedTemporaryFile, TemporaryFile

if sys.version_info >= (3, 0):
    basestring = str

from pydub.logging_utils import log_conversion, log_subprocess_output
from pydub.utils import audioop
from pydub.exceptions import (
    InvalidID3TagVersion,
    InvalidTag,
    CouldntEncodeError,
    CouldntDecodeError,
)

from pydub import AudioSegment
from pydub.audio_segment import fix_wav_headers

def _fd_or_path_or_tempfile(fd, mode='w+b', tempfile=True):
    close_fd = False
    if fd is None and tempfile:
        fd = TemporaryFile(mode=mode)
        close_fd = True

    if isinstance(fd, basestring):
        fd = open(fd, mode=mode)
        close_fd = True

    if isinstance(fd, BufferedReader):
        close_fd = True

    try:
        if isinstance(fd, os.PathLike):
            fd = open(fd, mode=mode)
            close_fd = True
    except AttributeError:
        # module os has no attribute PathLike, so we're on python < 3.6.
        # The protocol we're trying to support doesn't exist, so just pass.
        pass

    return fd, close_fd

class PatchedAudioSegment(AudioSegment):
    def append(self, seg, crossfade=100, dynamic_crossfade=False):
        seg1, seg2 = AudioSegment._sync(self, seg)

        if not crossfade:
            return seg1._spawn(seg1._data + seg2._data)
        elif crossfade > len(self):
            if dynamic_crossfade:
                crossfade = len(self)
            else:
                raise ValueError("Crossfade is longer than the original AudioSegment ({}ms > {}ms)".format(
                    crossfade, len(self)
                ))
        elif crossfade > len(seg):
            if dynamic_crossfade:
                crossfade = len(seg)
            else:
                raise ValueError("Crossfade is longer than the appended AudioSegment ({}ms > {}ms)".format(
                    crossfade, len(seg)
                ))

        xf = seg1[-crossfade:].fade(to_gain=-120, start=0, end=float('inf'))
        xf *= seg2[:crossfade].fade(from_gain=-120, start=0, end=float('inf'))

        # This is the code in the pydub repo (added after the latest release):
        # output = BytesIO()

        # output.write(seg1[:-crossfade]._data)
        # output.write(xf._data)
        # output.write(seg2[crossfade:]._data)

        # output.seek(0)
        # obj = seg1._spawn(data=output)
        # output.close()
        # return obj

        # This is another approach that seems to be faster.
        # The _spawn method by default accepts a list of bytes objects and
        # concatenates them. This results in the same data as the above solution,
        # but appears to work faster.
        # Might not handle large amounts of data as well, though.
        return seg1._spawn(data=[
            seg1[:-crossfade]._data, xf._data, seg2[crossfade:]._data])

    def export(self, out_f=None, format='mp3', codec=None, bitrate=None, parameters=None, tags=None, id3v2_version='4',
               cover=None):
        """
        Export an AudioSegment to a file with given options

        out_f (string):
            Path to destination audio file. Also accepts os.PathLike objects on
            python >= 3.6

        format (string)
            Format for destination audio file.
            ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

        codec (string)
            Codec used to encode the destination file.

        bitrate (string)
            Bitrate used when encoding destination file. (64, 92, 128, 256, 312k...)
            Each codec accepts different bitrate arguments so take a look at the
            ffmpeg documentation for details (bitrate usually shown as -b, -ba or
            -a:b).

        parameters (list of strings)
            Aditional ffmpeg/avconv parameters

        tags (dict)
            Set metadata information to destination files
            usually used as tags. ({title='Song Title', artist='Song Artist'})

        id3v2_version (string)
            Set ID3v2 version for tags. (default: '4')

        cover (file)
            Set cover for audio file from image file. (png or jpg)
        """
        id3v2_allowed_versions = ['3', '4']

        if format == "raw" and (codec is not None or parameters is not None):
            raise AttributeError(
                    'Can not invoke ffmpeg when export format is "raw"; '
                    'specify an ffmpeg raw format like format="s16le" instead '
                    'or call export(format="raw") with no codec or parameters')

        out_f, _ = _fd_or_path_or_tempfile(out_f, 'wb+')
        out_f.seek(0)

        if format == "raw":
            out_f.write(self._data)
            out_f.seek(0)
            return out_f

        # wav with no ffmpeg parameters can just be written directly to out_f
        easy_wav = format == "wav" and codec is None and parameters is None

        if easy_wav:
            data = out_f
        else:
            data = NamedTemporaryFile(mode="wb", delete=False)

        pcm_for_wav = self._data
        if self.sample_width == 1:
            # convert to unsigned integers for wav
            pcm_for_wav = audioop.bias(self._data, 1, 128)

        wave_data = wave.open(data, 'wb')
        wave_data.setnchannels(self.channels)
        wave_data.setsampwidth(self.sample_width)
        wave_data.setframerate(self.frame_rate)
        # For some reason packing the wave header struct with
        # a float in python 2 doesn't throw an exception
        wave_data.setnframes(int(self.frame_count()))
        wave_data.writeframesraw(pcm_for_wav)
        wave_data.close()

        # for easy wav files, we're done (wav data is written directly to out_f)
        if easy_wav:
            out_f.seek(0)
            return out_f

        output = NamedTemporaryFile(mode="w+b", delete=False)

        # build converter command to export
        conversion_command = [
            self.converter,
            '-y',  # always overwrite existing files
            "-f", "wav", "-i", data.name,  # input options (filename last)
        ]

        if codec is None:
            codec = self.DEFAULT_CODECS.get(format, None)

        if cover is not None:
            if cover.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')) and format == "mp3":
                conversion_command.extend(["-i", cover, "-map", "0", "-map", "1", "-c:v", "mjpeg"])
            else:
                raise AttributeError(
                    "Currently cover images are only supported by MP3 files. The allowed image formats are: .tif, .jpg, .bmp, .jpeg and .png.")

        if codec is not None:
            # force audio encoder
            conversion_command.extend(["-acodec", codec])

        if bitrate is not None:
            conversion_command.extend(["-b:a", bitrate])

        if parameters is not None:
            # extend arguments with arbitrary set
            conversion_command.extend(parameters)

        if tags is not None:
            if not isinstance(tags, dict):
                raise InvalidTag("Tags must be a dictionary.")
            else:
                # Extend converter command with tags
                # print(tags)
                for key, value in tags.items():
                    conversion_command.extend(
                        ['-metadata', '{0}={1}'.format(key, value)])

                if format == 'mp3':
                    # set id3v2 tag version
                    if id3v2_version not in id3v2_allowed_versions:
                        raise InvalidID3TagVersion(
                            "id3v2_version not allowed, allowed versions: %s" % id3v2_allowed_versions)
                    conversion_command.extend([
                        "-id3v2_version", id3v2_version
                    ])

        if sys.platform == 'darwin' and codec == 'mp3':
            conversion_command.extend(["-write_xing", "0"])

        conversion_command.extend([
            "-f", format, output.name,  # output options (filename last)
        ])

        log_conversion(conversion_command)
        
        # read stdin / write stdout
        with open(os.devnull, 'rb') as devnull:
            p = subprocess.Popen(conversion_command, stdin=devnull, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate()

        log_subprocess_output(p_out)
        log_subprocess_output(p_err)

        try:
            if p.returncode != 0:
                raise CouldntEncodeError(
                    "Encoding failed. ffmpeg/avlib returned error code: {0}\n\nCommand:{1}\n\nOutput from ffmpeg/avlib:\n\n{2}".format(
                        p.returncode, conversion_command, p_err.decode(errors='ignore') ))

            output.seek(0)
            out_f.write(output.read())

        finally:
            data.close()
            output.close()
            os.unlink(data.name)
            os.unlink(output.name)

        out_f.seek(0)
        return out_f






    def process_with_ffmpeg(self, parameters: list = None, **kwargs) -> Self:
        pcm_for_wav = self._data
        if self.sample_width == 1:
            # convert to unsigned integers for wav
            pcm_for_wav = audioop.bias(self._data, 1, 128)

        data = BytesIO()
        # data = NamedTemporaryFile(mode="w+b", delete=False)
        wave_data = wave.open(data, 'wb')
        wave_data.setnchannels(self.channels)
        wave_data.setsampwidth(self.sample_width)
        wave_data.setframerate(self.frame_rate)
        # For some reason packing the wave header struct with
        # a float in python 2 doesn't throw an exception
        wave_data.setnframes(int(self.frame_count()))
        wave_data.writeframesraw(pcm_for_wav)
        wave_data.close()
        data.seek(0)

        stdin_parameter = subprocess.PIPE
        stdin_data = data.read()

        # build converter command to export
        conversion_command = [self.converter, "-f", "wav"]

        read_ahead_limit = kwargs.get('read_ahead_limit', -1)  # Unlimited
        conversion_command.extend([
            "-read_ahead_limit", str(read_ahead_limit), "-i", "cache:pipe:0",
        ])

        if parameters is not None:
            # extend arguments with arbitrary set
            conversion_command.extend(parameters)

        conversion_command.extend([
            "-f", "wav", "-" # Stream output
        ])

        # Quotes within the command is handled poorly on Windows.
        # It works better if the command is one string, instead of a list of strings.
        if sys.platform == 'win32':
            conversion_command = " ".join(conversion_command)

        log_conversion(conversion_command)

        p = subprocess.Popen(conversion_command, stdin=stdin_parameter,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate(input=stdin_data)

        try:
            if p.returncode != 0 or len(p_out) == 0:
                raise CouldntDecodeError(
                    "Decoding failed. ffmpeg returned error code: {0}\n\n"
                    "Command:{1}\n\nOutput from ffmpeg/avlib:\n\n{1}".format(
                        p.returncode, conversion_command, p_err.decode(errors='ignore') ))

            p_out = bytearray(p_out)
            fix_wav_headers(p_out)
            p_out = bytes(p_out)
            obj = self.__class__(p_out)

        finally:
            data.close()
        return obj
