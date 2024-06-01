# Songtwister
Twist songs by removing or transforming beats.

A Python script that takes an audio file, the beats per minute count and the position of the first beat, and uses it to apply various effects presets to the audio.

The main use case is to remove specific parts of each bar, transforming the time signature. Most pop songs are in 4/4 time. Removing one beat per bar makes it 3/4, giving it a waltz feel. Removing two half beats per bar, on the other hand, makes it 6/8 and gives it a boogie or folksy feel. And removing the last 16th note from each beat gives it a swing feel.

The script handles the calculations, cuts up the audio, and adds a crossfade to smoothen the cutpoints. It utilizes the PyDub library.

The primary effect is ‘remove’, but other effects are available. These include a very basic downpitching, speed changes, panning, reversing, etc.

## Installation and Getting  Started

Dependencies: a recent version of Python 3, ffmpeg and git.

Linux, and probably Mac OS:

```
git clone https://github.com/persondoingstuff/Songtwister.git
cd Songtwister
python -m venv venv
source venv/bin/activate
python3 run.py -s example
```

Windows:

```
git clone https://github.com/persondoingstuff/Songtwister.git
cd Songtwister
python -m venv venv
.\venv\Scripts\Activate.ps1
python .\run.py -s example
```

Find the original file in the audio directory, along with newly generated versions with swing and in 6/8, 7/8 and 3/4 time.

## Basic usage

`run.py` provides a command line interface to the Songtwister class.

In the file `songs.yml` the details of each song to process are defined, like this:

```yaml
example:
  filename: audio/sorry.mp3
  bpm: 110
  prefix_length_ms: 0
```

For now, you need to find out the BPM (beats per minute) of the song yourself. Usually this is easily found by searching the song title  and ‘bpm’.

You also need to determine the prefix length. This is the point in the audio file where the first proper bar starts. Songs usually have a few hundred milliseconds. If there is a sound effect at the beginning, or an upbeat, it might be several seconds.

The easiest way is to load the file into an audio editor (eg. REAPER), setting the BPM and aligning the audio to the grid. The the length of whatever comes before the it starts following the grid lines, will be the prefix.

You might also try a built-in tool to guess the prefix, based on when the audio level becomes higher than a certain level:

```
python .\run.py -s example --guess-prefix
```

When the songs config has been added, you can make the basic set of four presets (swing, folk, waltz, seven) like this:

```
python .\run.py -s example
```

Or you can make a specific preset:

```
python .\run.py -s example -p swing
```

You can also set a custom crossfade length. For some songs, it helps with a longer fade, to smoothen the hard edges. Other times, it helps with sharper cuts, as longer crossfades give a smeared, blurry effect. It’s trial and error.

Crossfade may be given as a number of milliseconds or as a fraction of a beat:

```
python .\run.py --song example --preset waltz --crossfade 100
python .\run.py --song example --preset folk --crossfade 1/64
```

Default is 1/128 of a beat.

You can see the available effect presets in the `effect_presets` directory. You can also add your own.

## Detailed usage of the command line interface

```
    parser.add_argument("-s", "--song", required=True, type=str,
                        help="The name of the song definition, "
                        "set in the songs file")
    parser.add_argument("-p", "--preset", required=False, type=str,
                        help="The preset to use. Presets may be defined "
                        "in the presets file and called here by name.")
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
                        "in the song before the beat starts, and generate files "
                        "with single bars to evaluate the guessed value.")
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
```



## Making custom effects preset

Each effect preset is defined in one of the yaml files in the `effect_presets` directory. For instance, the swing preset looks like this:

```yaml
swing:
  crossfade: 1/48
  effects:
  - effect: remove
    bars: all
    beats: every 4 of 4
    beats_per_bar: 16
```

It has a name (“swing”) and inside it the effects are defined in a list inside the “effects” item. Optionally, a crossfade setting may be provided.

Multiple effects may be added, but in the main cases, it is only one.

The above example instructs the script to target every bar of the song. Each bar is divided into 16, and every fourth is affected by the effect, which in this case simply removes the beats.

These are the available effects:
- 'remove': Delete a section of the audio
- 'silence': replace a section of the audio with silence
- 'reverse': reverse a section of the audio
- 'repeat': repeat a section of the audio, right after the original placement
- 'move': move a section of the audio to a new placement. FIXME: Does not exist, but it should
- 'speedup': increase the playback speed of a section, preserving pitch
- 'tempo': increase or decrease the speed of a section, changing pitch accordingly

FIXME: Some are missing. Add additional usage information. There is also pingpong, pan

Except for “remove” and “silence”, effects may be layered on top of each other.

For “bars” and “beats” in the effect preset, the selection may be made like this:

1. A number: select this specific bar
2. `"even"` or `"odd"`: select alternating items.
3. `"first"` or `"last"`: select the first or last item.
4. `None`, `0`, `"none"` or `False`: select no items – a simple pass-through that is not terribly useful.
5. `"all"` or `True`: select all items.
6. `"random X"` (where X is a number):  select X random items. No number results in a random number of random items.
7. `"every X of Y"` (where X and Y  are numbers): select item X in each batch of Y, for as many batches as can be made.

Non-existing items are ignored

## Importing and exporting song state

When a Songtwister object is instantiated, it carries state information about bpm, prefix, effects to be applied and more.

If you want to save this state, you can of course pickle the object.

However, in case you want it as json, you can use `song_object.export_state()` to get all but the audio data itself. You can then recreate the object using `song_object = Songtwister(**data)`.

It is also possible to instantiate the object without reading in the audio file itself.

I did this because I considered building a web app frontend, and here it would be necessary for the REST API to be able to quickly reload the object state between steps in a form flow. And then it could limit reading the audio file to the steps that actually needed it.

I probably won’t build the frontend, but I encourage anyone else to do it. It should include some visual tool to test the prefix and BPM settings. The Songtwister class can generate peaks of the audio, which might be visualized as seen in the HTML template.
