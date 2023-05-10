# croaker

> make sound go _croak_

<img width="899" alt="croaker, a plugin that makes sound go croak" src="https://github.com/renzol2/croaker/assets/55109467/3028d5ca-27b5-4c69-9fb1-c21d42aabd2c">

## About

**croaker** is a VST3/CLAP plugin that will make your audio sound like a croaking frog 🐸 (it's actually just a multi effects plugin).

**croaker** is made with the following:

- dsp with my own library, [fx](https://github.com/renzol2/fx)
- plugin boilerplate using [nih-plug](https://github.com/robbert-vdh/nih-plug), and UI using [Vizia](https://github.com/vizia/vizia)

this project is being completed partially in fulfillment of the senior capstone requirement for the [ Computer Science + Music degree program ](https://music.illinois.edu/admission/undergraduate-programs-and-application/undergraduate-degrees/bachelor-of-science-cs-music/) at the University of Illinois at Urbana-Champaign.

## Download

A temporary download link to standalone, VST3, and CLAP versions can be found here: [Google Drive](https://drive.google.com/drive/folders/1MUqEvY7qZCM-ucpszpEqc7d3vIfdQkVw?usp=sharing)

I have only tested plugin on macOS Monterey 12.5 using Audacity and Ableton Live 11. I haven't been able to run this on Reason 12. Also, since I don't have an Apple Developer account (yet), I haven't codesigned these applications. So, you'll need to codesign yourself. Here are some useful links:

- <https://www.osirisguitar.com/how-to-make-unsigned-vsts-work-in-macos-catalina/>
- <https://forum.cabbageaudio.com/t/how-to-codesign-osx-plugins/3079>
- <https://melatonin.dev/blog/how-to-code-sign-and-notarize-macos-audio-plugins-in-ci/>

## Building

After installing [Rust](https://rustup.rs/), you can compile croaker as follows:

```shell
cargo xtask bundle croaker --release
```

## License

This project is licensed under the GNU General Public License Version 3 or later.

Code that is copied or translated from other sources will be noted in the source code.

(If there is an issue with licensing, please contact me at renzomledesma@gmail.com)

