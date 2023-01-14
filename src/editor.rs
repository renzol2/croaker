use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext, Param, ParamPtr};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::CroakerParams;

const STYLE: &str = include_str!("style.css");
pub const WINDOW_WIDTH: u32 = 512;
pub const WINDOW_HEIGHT: u32 = 256;

#[derive(Lens)]
struct Data {
    pub gui_context: Arc<dyn GuiContext>,
    params: Arc<CroakerParams>,
    peak_meter: Arc<AtomicF32>,
}

// `ParamChangeEvent` enum credits to Fredemus and geom3trik
// https://github.com/Fredemus/va-filter/blob/b3a3b2ae4004738fb7ae5a114e372fdda6213fb1/src/ui.rs#L36
// Under GPLv3 License
#[derive(Debug)]
pub enum ParamChangeEvent {
    BeginSet(ParamPtr),
    EndSet(ParamPtr),
    SetParam(ParamPtr, f32),
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|event, _| match event {
            ParamChangeEvent::SetParam(param_ptr, new_value) => {
                unsafe {
                    self.gui_context
                        .raw_set_parameter_normalized(*param_ptr, *new_value)
                };
            }
            ParamChangeEvent::BeginSet(param_ptr) => {
                unsafe { self.gui_context.raw_begin_set_parameter(*param_ptr) };
            }
            ParamChangeEvent::EndSet(param_ptr) => {
                unsafe { self.gui_context.raw_end_set_parameter(*param_ptr) };
            }
        })
    }
}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::from_size(WINDOW_WIDTH, WINDOW_HEIGHT)
}

pub(crate) fn create(
    params: Arc<CroakerParams>,
    peak_meter: Arc<AtomicF32>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, context| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);
        cx.add_theme(STYLE);

        Data {
            gui_context: context.clone(),
            params: params.clone(),
            peak_meter: peak_meter.clone(),
        }
        .build(cx);

        ResizeHandle::new(cx);

        VStack::new(cx, |cx| {
            Label::new(cx, "renzofrog croaker")
                .font(assets::NOTO_SANS_BOLD_ITALIC)
                .font_size(30.0)
                .height(Pixels(50.0))
                .child_top(Stretch(1.0))
                .child_bottom(Pixels(0.0));

            HStack::new(cx, |cx| {
                // Gain control
                // Label::new(cx, "Gain").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.gain);
                make_knob(cx, params.gain.as_ptr(), |params| &params.gain);

                // Saturation control
                // Label::new(cx, "Saturation").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.saturation);
                make_knob(cx, params.saturation.as_ptr(), |params| &params.saturation);
            }).class("knobs");

            PeakMeter::new(
                cx,
                Data::peak_meter
                    .map(|peak_meter| util::gain_to_db(peak_meter.load(Ordering::Relaxed))),
                Some(Duration::from_millis(600)),
            )
            // This is how adding padding works in vizia
            .top(Pixels(10.0))
            .bottom(Pixels(10.0));
        })
        .row_between(Pixels(0.0))
        .child_left(Stretch(1.0))
        .child_right(Stretch(1.0))
        .class("container");
    })
}

// Knob helper function credits to Fredemus and geom3trik
// https://github.com/Fredemus/va-filter/blob/b3a3b2ae4004738fb7ae5a114e372fdda6213fb1/src/ui.rs#L192
// Under GPLv3 License
fn make_knob<P, F>(cx: &mut Context, param_ptr: ParamPtr, params_to_param: F) -> Handle<VStack>
where
    P: Param,
    F: 'static + Fn(&Arc<CroakerParams>) -> &P + Copy,
{
    VStack::new(cx, move |cx| {
        // Param name
        Label::new(
            cx,
            Data::params.map(move |params| params_to_param(params).name().to_owned()),
        );

        // Knob
        Knob::custom(
            cx,
            0.5,
            Data::params.map(move |params| params_to_param(params).modulated_normalized_value()),
            move |cx, lens| {
                TickKnob::new(
                    cx,
                    Percentage(80.0),
                    Pixels(4.),
                    Percentage(50.0),
                    270.0,
                    KnobMode::Continuous,
                )
                .value(lens.clone())
                .class("tick");
                ArcTrack::new(
                    cx,
                    false,
                    Percentage(100.0),
                    Percentage(10.),
                    -135.,
                    135.,
                    KnobMode::Continuous,
                )
                .value(lens)
                .class("track")
            },
        )
        .on_changing(move |cx, val| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                // ParamChangeEvent::AllParams(param_index, val),
                ParamChangeEvent::SetParam(param_ptr, val),
            )
        })
        .on_mouse_down(move |cx, _| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::BeginSet(param_ptr),
            )
        })
        .on_mouse_up(move |cx, _| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::EndSet(param_ptr),
            )
        });

        // Units
        Label::new(
            cx,
            Data::params.map(move |params| params_to_param(params).to_string()),
        )
        .width(Pixels(100.));
    })
    .child_space(Stretch(0.2))
}
