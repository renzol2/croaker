use nih_plug_vizia::vizia;
use vizia::prelude::*;

use crate::fonts;

pub const FANTASQUE_SANS_MONO: &str = "Fantasque Sans Mono";
// pub const FANTASQUE_SANS_MONO_ITALIC: &str = "Fantasque Sans Mono Italic";
// pub const FANTASQUE_SANS_MONO_BOLD: &str = "Fantasque Sans Mono Bold";
// pub const FANTASQUE_SANS_MONO_BOLD_ITALIC: &str = "Fantasque Sans Mono Bold Italic";

pub fn register_fantasque_sans_mono(cx: &mut Context) {
    cx.add_fonts_mem(&[fonts::FANTASQUE_SANS_MONO]);
}

// pub fn register_fantasque_sans_mono_italic(cx: &mut Context) {
//     cx.add_fonts_mem(&[fonts::FANTASQUE_SANS_MONO_ITALIC]);
// }

// pub fn register_fantasque_sans_mono_bold(cx: &mut Context) {
//     cx.add_fonts_mem(&[fonts::FANTASQUE_SANS_MONO_BOLD]);
// }

// pub fn register_fantasque_sans_mono_bold_italic(cx: &mut Context) {
//     cx.add_fonts_mem(&[fonts::FANTASQUE_SANS_MONO_BOLD_ITALIC]);
// }
