# Songtwister
> Twist songs by removing or transforming beats.

Author: Sigfred Nielsen (persondoingstuff.com)

If you do anything with this, please drop me a note at sn@persondoingstuff.com

Songtwister is a Python script that takes an audio file, the beats per minute count and the position of the first beat, and uses it to apply various effects presets to the audio.

The main use case is to remove specific parts of each bar, transforming the time signature. Most pop songs are in 4/4 time. Removing one beat per bar makes it 3/4, giving it a waltz feel. Removing two half beats per bar, on the other hand, makes it 6/8 and gives it a boogie or folksy feel. And removing the last 16th note from each beat gives it a swing feel.

The script handles the calculations, cuts up the audio, and adds a crossfade to smoothen the cutpoints. It utilizes the PyDub library.

The primary effect is ‘remove’, but other effects are available. These include a very basic downpitching, speed changes, panning, reversing, etc.

## Installation and Getting  Started

Dependencies:

- A recent version of Python 3, ffmpeg and git.
- The python packages PyDub, PyYAML and Jinja2, along with their dependencies.

It is recommended to install the script in a virtual environment.

Linux, and probably Mac OS:

```
git clone https://github.com/persondoingstuff/Songtwister.git
cd Songtwister
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run.py -s example
```

Windows:

```
git clone https://github.com/persondoingstuff/Songtwister.git
cd Songtwister
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
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
usage: run.py [-h] -s SONG [-p PRESET] [-c CROSSFADE] [-n VERSION_NAME] [-g] [-l] [-a] [-y] [-v]

options:
  -h, --help            show this help message and exit
  -s SONG, --song SONG  The name of the song definition, set in the songs file. Required.
  -p PRESET, --preset PRESET
                        The preset to use. Presets may be defined in the presets file and called here by name.
                        If this is not defined, the main set of presets from the config will be generated.
                        If * is passed, every preset will be generated. Optional.
  -c CROSSFADE, --crossfade CROSSFADE
                        Length of the crossfade between cuts in the song, overriding values set elsewhere.
                        May be an int of the number of milliseconds, or a fraction of a bar length.
                        Eg. 1/4, corresponding to a full beat in a four beat bar. Default is 1/128.
                        Optional. Global setting controlled in config.yml
  -n VERSION_NAME, --version-name VERSION_NAME
                        Give the processing of the song a version name. Optional.
  -g, --guess-prefix    Guess the number of milliseconds of prefix in the song before the beat starts,
                        and generate files with single bars to evaluate the guessed value. Optional.
  -l, --make-html       Generate an HTML page with audio players for the exported audio files. Optional.
  -a, --all             Generate the default set of presets as defined in the config file.
                        Optional and implied if --preset is not defined.
  -y, --overwrite       Overwrite existing files with same name.
                        Optional. Global setting controlled in config.yml
  -v, --verbose         Output detailed logging.
                        Optional. Global setting controlled in config.yml
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
- 'reverserepeat': first play the audio in reverse, then regularly
- 'repeatreverse': first play the audio regularly, then in reverse
- 'bounceback': the audio reverses back on top of itself
- 'speed': increase or decrease the speed of a section, changing pitch accordingly
  - Usage: `speed 1.1`
- 'speedup': increase the playback speed of a section, preserving pitch
  - Usage: `speedup 2`
- 'speedupfill': fill the same time as the original audio, but speed it up a certain amount.
  - Usage: `speedupfill 1.5`
- 'pitchdown': lower the pitch a number of semitones, very roughly.
  - Usage: `pitchdown 12`
- 'pingpong': make alternating hard pans of the audio a specified number of times (default 2)
  - Usage: `pingpong 2`
- 'left' and 'right': pan the audio to one of the sides
- 'across left' and 'across right': start the audio panned to one side, and move it to the other
- TODO: 'move': move a section of the audio to a new placement. Does not exist, but it should

Except for “remove” and “silence”, effects may be layered on top of each other. However, it will not always work, especially if different numbers of beats per bar are used.

Shorter or longer crossfades may be used to create interesting effects

For “bars” and “beats” in the effect preset, the selection may be made like this:

- A number: select this specific bar
- `"even"` or `"odd"`: select alternating items.
- `"first"` or `"last"`: select the first or last item.
- `None`, `0`, `"none"` or `False`: select no items – a simple pass-through that is not terribly useful.
- `"all"` or `True`: select all items.
- `"random X"` (where X is a number):  select X random items. No number results in a random number of random items.
- `"every X of Y"` (where X and Y  are numbers): select item X in each batch of Y, for as many batches as can be made.

Non-existing items are ignored.

See various examples of presets in the files in the `effect_presets` folder.

## Importing and exporting song state

When a Songtwister object is instantiated, it carries state information about bpm, prefix, effects to be applied and more.

If you want to save this state, you can of course pickle the object.

However, in case you want it as json, you can use `song_object.export_state()` to get all but the audio data itself. You can then recreate the object using `song_object = Songtwister(**data)`.

It is also possible to instantiate the object without reading in the audio file itself.

I did this because I considered building a web app frontend, and here it would be necessary for the REST API to be able to quickly reload the object state between steps in a form flow. And then it could limit reading the audio file to the steps that actually needed it.

I probably won’t build the frontend, but I encourage anyone else to do it. It should include some visual tool to test the prefix and BPM settings. The Songtwister class can generate peaks of the audio, which might be visualized as seen in the HTML template.
