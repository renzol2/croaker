# croaker

> makes sound go _croak_

**croaker** is a VST3/CLAP plugin that will make your audio sound like a croaking frog 🐸, or worse (i.e. it's a really badly tuned multi effects plugin).

**croaker** is made with the following:

- dsp with my own library, [fx](https://github.com/renzol2/fx)
- plugin boilerplate using [nih-plug](https://github.com/robbert-vdh/nih-plug), and UI using [Vizia](https://github.com/vizia/vizia)

this project is being completed partially in fulfillment of the senior capstone requirement for the [ Computer Science + Music degree program ](https://music.illinois.edu/admission/undergraduate-programs-and-application/undergraduate-degrees/bachelor-of-science-cs-music/) at the University of Illinois at Urbana-Champaign.

## Building

After installing [Rust](https://rustup.rs/), you can compile croaker as follows:

```shell
cargo xtask bundle croaker --release
```

## License

This project is licensed under the GNU General Public License Version 3 or later.

Code that is copied or translated from other sources will be noted in the source code.

(If there is an issue with licensing, please contact me at renzomledesma@gmail.com)
