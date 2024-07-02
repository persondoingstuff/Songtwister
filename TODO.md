- Add proper testing
- Enable setting an output dir in config and in the song definitions
- Enable setting a standard input/output dir in config
- Make HTML template configurable
- Improve effects and documentation of them


------


start: begin processing at this offset from the start of the audio piece. If not set, start at the beginning
end: stop processing at this offset from the end of the audio piece. If not set, stop at the end
length: process for this amount of time from start, or before end.

Two of these may be supplied at a time. If all three are present, a warning is thrown and one is ignored (length, most likely)

Valid units are:
absolute time -- as 0:01, 00:00:01, 1s, 50ms, 00:05.200
time relative to bars and beats -- as 1, 1 1/16, 3/4

--

Remove the prefix and 1 second from the end
- group:
  - do: cut
    length: prefix
  - do: cut
    start: 0:01

Remove beat 4 in every bar:
- do: cut
  beats: 4

- do: repeat
  start: 1
  length: 8
  times: 4

--

# cut
Take an A--B range within a piece of audio and delete it, shortening the audio. The entire audio piece may be removed.

- comment: Delete between 0:30-1:00
- do: cut
  from: 0:30
  to: 1:00

# keep
Take an A--B range within a piece of audio and delete everything before and after, within the audio piece.

- comment: Keep bars 4-8
- group:
  - do: cut
    end: 4
  - do: cut
    start: 9

# trim
Delete from the beginning and/or from the end of the audio piece. Meeting in the middle deletes it all.

- comment: Remove 1050 ms from the start and 15 seconds from the end
- group:
  - do: cut
    start: 0:10.500
  - do: cut
    start: 15s

# pad
Add silence at the begining and/or end of the audio piece, either for a duration or until a total duration is reached.

# repeat
Take an audio piece and repeat either a number of times or for an amount of time.

# mute
Replace the audio with silence, in place, retaining the original duration.

# shift
Change the tempo, pitch and/or playback speed of the piece of audio
