use gpui::{
    Capslock, KeyDownEvent, KeyUpEvent, Keystroke, Modifiers, ModifiersChangedEvent, MouseButton,
    MouseDownEvent, MouseExitEvent, MouseMoveEvent, MousePressureEvent, MouseUpEvent,
    NavigationDirection, PinchEvent, Pixels, PlatformInput, PressureStage, ScrollDelta,
    ScrollEvent, ScrollSource, TouchPhase, point, px,
};

use crate::{
    LMGetKbdType, NSStringExt, TISCopyCurrentKeyboardLayoutInputSource, TISGetInputSourceProperty,
    UCKeyTranslate, kTISPropertyUnicodeKeyLayoutData,
};
use core_foundation::data::{CFDataGetBytePtr, CFDataRef};
use core_graphics::event::CGKeyCode;
use objc2::{msg_send, runtime::AnyObject};
use objc2_app_kit::{
    NSDeleteFunctionKey, NSDownArrowFunctionKey, NSEndFunctionKey, NSEvent, NSEventModifierFlags,
    NSEventPhase, NSEventType, NSF1FunctionKey, NSF2FunctionKey, NSF3FunctionKey, NSF4FunctionKey,
    NSF5FunctionKey, NSF6FunctionKey, NSF7FunctionKey, NSF8FunctionKey, NSF9FunctionKey,
    NSF10FunctionKey, NSF11FunctionKey, NSF12FunctionKey, NSF13FunctionKey, NSF14FunctionKey,
    NSF15FunctionKey, NSF16FunctionKey, NSF17FunctionKey, NSF18FunctionKey, NSF19FunctionKey,
    NSF20FunctionKey, NSF21FunctionKey, NSF22FunctionKey, NSF23FunctionKey, NSF24FunctionKey,
    NSF25FunctionKey, NSF26FunctionKey, NSF27FunctionKey, NSF28FunctionKey, NSF29FunctionKey,
    NSF30FunctionKey, NSF31FunctionKey, NSF32FunctionKey, NSF33FunctionKey, NSF34FunctionKey,
    NSF35FunctionKey, NSHelpFunctionKey, NSHomeFunctionKey, NSLeftArrowFunctionKey,
    NSModeSwitchFunctionKey, NSPageDownFunctionKey, NSPageUpFunctionKey, NSRightArrowFunctionKey,
    NSUpArrowFunctionKey,
};
use std::{
    borrow::Cow,
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) const ESCAPE_KEY: u16 = 0x1b;

const BACKSPACE_KEY: u32 = 0x7f;
const SPACE_KEY: u32 = b' ' as u32;
const ENTER_KEY: u32 = 0x0d;
const NUMPAD_ENTER_KEY: u32 = 0x03;
const TAB_KEY: u32 = 0x09;
const SHIFT_TAB_KEY: u32 = 0x19;

// Menu key equivalents use different character values than the ones NSEvent
// reports for the same physical keys.
//
// See https://developer.apple.com/documentation/appkit/nsmenuitem/keyequivalent#Discussion
const BACKSPACE_KEY_EQUIVALENT: u32 = 0x08;
const DELETE_KEY_EQUIVALENT: u32 = 0x7f;

// ── 키패드 구분 (kVK_ANSI_Keypad* — Carbon `Events.h`) ──
//
// macOS 에는 NumLock 이 없어 키패드 숫자가 **문자로는 메인 숫자열과 구분되지 않는다**
// (`charactersIgnoringModifiers` 가 똑같이 "1" 을 준다). 구분이 남아 있는 곳은 `keyCode`
// 뿐이라 [`keypad_key`] 가 여기서 가른다.
const KEYPAD_DECIMAL: u16 = 0x41;
const KEYPAD_MULTIPLY: u16 = 0x43;
/// PC 배열 키패드의 NumLock 자리 (애플 키보드에는 `clear` 로 각인돼 있다).
const KEYPAD_CLEAR: u16 = 0x47;
const KEYPAD_ENTER: u16 = 0x4c;
const KEYPAD_0: u16 = 0x52;
const KEYPAD_1: u16 = 0x53;
const KEYPAD_2: u16 = 0x54;
const KEYPAD_3: u16 = 0x55;
const KEYPAD_4: u16 = 0x56;
const KEYPAD_5: u16 = 0x57;
const KEYPAD_6: u16 = 0x58;
const KEYPAD_7: u16 = 0x59;
const KEYPAD_8: u16 = 0x5b;
const KEYPAD_9: u16 = 0x5c;

/// 키패드 내비게이션 모드 — 켜면 키패드 숫자·Enter 가 **NumLock 을 끈 것처럼** 방향키·편집키로
/// 들어온다. macOS 에 NumLock 이 없어 키패드를 리모컨처럼 쓰는 앱이 직접 흉내내야 한다.
///
/// 이 값이 모드의 **단일 원천**이다 — [`parse_keystroke`] 는 `App` 을 볼 수 없어 앱 쪽 상태를
/// 읽어 올 수 없으므로, 앱이 [`set_keypad_nav_mode`] 로 여기에 쓰고 [`keypad_nav_mode`] 로 읽는다.
/// (선례: `window.rs` 의 `DRAG_MOVE_IS_DEFAULT`)
static KEYPAD_NAV_MODE: AtomicBool = AtomicBool::new(false);

/// 키패드 내비게이션 모드를 켜고 끈다. 꺼진 상태(기본)의 키 이름은 예전과 완전히 같다.
pub fn set_keypad_nav_mode(on: bool) {
    KEYPAD_NAV_MODE.store(on, Ordering::Relaxed);
}

/// 키패드 내비게이션 모드가 켜져 있는가.
pub fn keypad_nav_mode() -> bool {
    KEYPAD_NAV_MODE.load(Ordering::Relaxed)
}

/// 키패드 키의 이름. 키패드가 아니거나 번역할 게 없으면 `None` (기존 경로로 넘어간다).
///
/// `kp_clear` · `kp_multiply` 는 **모드와 무관하게 항상** 갈라 낸다 — 모드를 켜는 키 자체라
/// 꺼져 있을 때도 구분되지 않으면 켤 방법이 없다. 나머지는 모드가 켜져 있을 때만 번역해서
/// 꺼진 상태의 동작을 예전 그대로 남긴다 (키패드 `+ - /` 는 어느 모드에서도 건드리지 않는다 —
/// 문자 그대로 들어가는 게 맞는 키들이다).
///
/// 0 · 5 · . 는 NumLock 을 꺼도 방향키가 되지 않는 자리라 `kp_*` 이름 그대로 내보낸다.
/// 기본 바인딩이 없어 아무 일도 하지 않고(= 삼켜지고), 원하는 사용자는 직접 할당할 수 있다.
pub(crate) fn keypad_key(code: CGKeyCode) -> Option<&'static str> {
    match code {
        KEYPAD_CLEAR => return Some("kp_clear"),
        KEYPAD_MULTIPLY => return Some("kp_multiply"),
        _ => {}
    }
    if !keypad_nav_mode() {
        return None;
    }
    Some(match code {
        KEYPAD_8 => "up",
        KEYPAD_2 => "down",
        KEYPAD_4 => "left",
        KEYPAD_6 => "right",
        KEYPAD_7 => "home",
        KEYPAD_1 => "end",
        KEYPAD_9 => "pageup",
        KEYPAD_3 => "pagedown",
        KEYPAD_ENTER => "kp_enter",
        KEYPAD_0 => "kp_0",
        KEYPAD_5 => "kp_5",
        KEYPAD_DECIMAL => "kp_decimal",
        _ => return None,
    })
}

pub(crate) fn key_to_menu_equivalent(key: &str) -> Cow<'_, str> {
    let code = match key {
        "space" => SPACE_KEY,
        "backspace" => BACKSPACE_KEY_EQUIVALENT,
        "escape" => u32::from(ESCAPE_KEY),
        "up" => NSUpArrowFunctionKey,
        "down" => NSDownArrowFunctionKey,
        "left" => NSLeftArrowFunctionKey,
        "right" => NSRightArrowFunctionKey,
        "pageup" => NSPageUpFunctionKey,
        "pagedown" => NSPageDownFunctionKey,
        "home" => NSHomeFunctionKey,
        "end" => NSEndFunctionKey,
        "delete" => DELETE_KEY_EQUIVALENT,
        "insert" => NSHelpFunctionKey,
        "f1" => NSF1FunctionKey,
        "f2" => NSF2FunctionKey,
        "f3" => NSF3FunctionKey,
        "f4" => NSF4FunctionKey,
        "f5" => NSF5FunctionKey,
        "f6" => NSF6FunctionKey,
        "f7" => NSF7FunctionKey,
        "f8" => NSF8FunctionKey,
        "f9" => NSF9FunctionKey,
        "f10" => NSF10FunctionKey,
        "f11" => NSF11FunctionKey,
        "f12" => NSF12FunctionKey,
        "f13" => NSF13FunctionKey,
        "f14" => NSF14FunctionKey,
        "f15" => NSF15FunctionKey,
        "f16" => NSF16FunctionKey,
        "f17" => NSF17FunctionKey,
        "f18" => NSF18FunctionKey,
        "f19" => NSF19FunctionKey,
        "f20" => NSF20FunctionKey,
        "f21" => NSF21FunctionKey,
        "f22" => NSF22FunctionKey,
        "f23" => NSF23FunctionKey,
        "f24" => NSF24FunctionKey,
        "f25" => NSF25FunctionKey,
        "f26" => NSF26FunctionKey,
        "f27" => NSF27FunctionKey,
        "f28" => NSF28FunctionKey,
        "f29" => NSF29FunctionKey,
        "f30" => NSF30FunctionKey,
        "f31" => NSF31FunctionKey,
        "f32" => NSF32FunctionKey,
        "f33" => NSF33FunctionKey,
        "f34" => NSF34FunctionKey,
        "f35" => NSF35FunctionKey,
        _ => return Cow::Borrowed(key),
    };
    Cow::Owned(String::from_utf16(&[code as u16]).unwrap())
}

fn read_modifiers(native_event: &NSEvent) -> Modifiers {
    let modifiers = native_event.modifierFlags();
    let control = modifiers.contains(NSEventModifierFlags::Control);
    let alt = modifiers.contains(NSEventModifierFlags::Option);
    let shift = modifiers.contains(NSEventModifierFlags::Shift);
    let command = modifiers.contains(NSEventModifierFlags::Command);
    let function = modifiers.contains(NSEventModifierFlags::Function);

    Modifiers {
        control,
        alt,
        shift,
        platform: command,
        function,
    }
}

pub(crate) unsafe fn platform_input_from_native(
    native_event: *mut AnyObject,
    window_height: Option<Pixels>,
) -> Option<PlatformInput> {
    unsafe {
        let native_event = native_event.cast::<NSEvent>().as_ref()?;
        let event_type = native_event.r#type();

        // Filter out event types that aren't in the NSEventType enum.
        // See https://github.com/servo/cocoa-rs/issues/155#issuecomment-323482792 for details.
        match event_type.0 {
            0 | 21 | 32 | 33 | 35 | 36 | 37 => {
                return None;
            }
            _ => {}
        }

        match event_type {
            NSEventType::FlagsChanged => {
                Some(PlatformInput::ModifiersChanged(ModifiersChangedEvent {
                    modifiers: read_modifiers(native_event),
                    capslock: Capslock {
                        on: native_event
                            .modifierFlags()
                            .contains(NSEventModifierFlags::CapsLock),
                    },
                }))
            }
            NSEventType::KeyDown => Some(PlatformInput::KeyDown(KeyDownEvent {
                keystroke: parse_keystroke(native_event),
                is_held: native_event.isARepeat(),
                prefer_character_input: false,
            })),
            NSEventType::KeyUp => Some(PlatformInput::KeyUp(KeyUpEvent {
                keystroke: parse_keystroke(native_event),
            })),
            NSEventType::LeftMouseDown
            | NSEventType::RightMouseDown
            | NSEventType::OtherMouseDown => {
                let button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };
                window_height.map(|window_height| {
                    PlatformInput::MouseDown(MouseDownEvent {
                        button,
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            // MacOS screen coordinates are relative to bottom left
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                        click_count: native_event.clickCount() as usize,
                        first_mouse: false,
                    })
                })
            }
            NSEventType::LeftMouseUp | NSEventType::RightMouseUp | NSEventType::OtherMouseUp => {
                let button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };

                window_height.map(|window_height| {
                    PlatformInput::MouseUp(MouseUpEvent {
                        button,
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                        click_count: native_event.clickCount() as usize,
                    })
                })
            }
            NSEventType::Pressure => {
                let stage = native_event.stage();
                let pressure = native_event.pressure();

                window_height.map(|window_height| {
                    PlatformInput::MousePressure(MousePressureEvent {
                        stage: match stage {
                            1 => PressureStage::Normal,
                            2 => PressureStage::Force,
                            _ => PressureStage::Zero,
                        },
                        pressure,
                        modifiers: read_modifiers(native_event),
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                    })
                })
            }
            // Some mice (like Logitech MX Master) send navigation buttons as swipe events
            NSEventType::Swipe => {
                let navigation_direction = match native_event.phase() {
                    NSEventPhase::Ended => match native_event.deltaX() {
                        x if x > 0.0 => Some(NavigationDirection::Back),
                        x if x < 0.0 => Some(NavigationDirection::Forward),
                        _ => return None,
                    },
                    _ => return None,
                };

                match navigation_direction {
                    Some(direction) => window_height.map(|window_height| {
                        PlatformInput::MouseDown(MouseDownEvent {
                            button: MouseButton::Navigate(direction),
                            position: point(
                                px(native_event.locationInWindow().x as f32),
                                window_height - px(native_event.locationInWindow().y as f32),
                            ),
                            modifiers: read_modifiers(native_event),
                            click_count: 1,
                            first_mouse: false,
                        })
                    }),
                    _ => None,
                }
            }
            NSEventType::Magnify => window_height.map(|window_height| {
                let phase = match native_event.phase() {
                    NSEventPhase::MayBegin | NSEventPhase::Began => TouchPhase::Started,
                    NSEventPhase::Ended => TouchPhase::Ended,
                    _ => TouchPhase::Moved,
                };

                let magnification = native_event.magnification() as f32;

                PlatformInput::Pinch(PinchEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    delta: magnification,
                    modifiers: read_modifiers(native_event),
                    phase,
                })
            }),
            NSEventType::ScrollWheel => window_height.map(|window_height| {
                let phase = match native_event.phase() {
                    NSEventPhase::MayBegin | NSEventPhase::Began => TouchPhase::Started,
                    NSEventPhase::Ended => TouchPhase::Ended,
                    _ => TouchPhase::Moved,
                };

                let raw_data = point(
                    native_event.scrollingDeltaX() as f32,
                    native_event.scrollingDeltaY() as f32,
                );

                let delta = if native_event.hasPreciseScrollingDeltas() {
                    ScrollDelta::Pixels(raw_data.map(px))
                } else {
                    ScrollDelta::Lines(raw_data)
                };

                PlatformInput::Scroll(ScrollEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    delta,
                    touch_phase: phase,
                    modifiers: read_modifiers(native_event),
                    source: ScrollSource::Pointer,
                })
            }),
            NSEventType::LeftMouseDragged
            | NSEventType::RightMouseDragged
            | NSEventType::OtherMouseDragged => {
                let pressed_button = match native_event.buttonNumber() {
                    0 => MouseButton::Left,
                    1 => MouseButton::Right,
                    2 => MouseButton::Middle,
                    3 => MouseButton::Navigate(NavigationDirection::Back),
                    4 => MouseButton::Navigate(NavigationDirection::Forward),
                    // Other mouse buttons aren't tracked currently
                    _ => return None,
                };

                window_height.map(|window_height| {
                    PlatformInput::MouseMove(MouseMoveEvent {
                        pressed_button: Some(pressed_button),
                        position: point(
                            px(native_event.locationInWindow().x as f32),
                            window_height - px(native_event.locationInWindow().y as f32),
                        ),
                        modifiers: read_modifiers(native_event),
                    })
                })
            }
            NSEventType::MouseMoved => window_height.map(|window_height| {
                PlatformInput::MouseMove(MouseMoveEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),
                    pressed_button: None,
                    modifiers: read_modifiers(native_event),
                })
            }),
            NSEventType::MouseExited => window_height.map(|window_height| {
                PlatformInput::MouseExited(MouseExitEvent {
                    position: point(
                        px(native_event.locationInWindow().x as f32),
                        window_height - px(native_event.locationInWindow().y as f32),
                    ),

                    pressed_button: None,
                    modifiers: read_modifiers(native_event),
                })
            }),
            _ => None,
        }
    }
}

/// 이 키다운을 [`keypad_key`] 가 다른 이름으로 바꿔 내보내는가.
///
/// `window.rs` 가 **IME 를 먼저 태울지** 정할 때 쓴다. 번역된 키의 원본 NSEvent 는 여전히
/// "8" · "*" 같은 **인쇄 가능한 문자**를 들고 있어서, 그대로 `NSTextInputContext` 에 넘기면
/// `insertText:` 로 소비되고(YES 반환) 키맵까지 오지 못한다. 진짜 방향키가 같은 경로를
/// 무사히 지나는 것은 문자가 비인쇄 문자라 `doCommandBySelector:` 로 빠지기 때문인데,
/// 번역된 키패드는 그 성질을 물려받지 못한다.
pub(crate) fn is_keypad_event(native_event: &NSEvent) -> bool {
    keypad_key(native_event.keyCode()).is_some()
}

fn parse_keystroke(native_event: &NSEvent) -> Keystroke {
    let characters = unsafe {
        let characters: *mut AnyObject = msg_send![native_event, charactersIgnoringModifiers];
        characters.to_str().to_string()
    };
    let mut key_char = None;
    let first_char = characters.chars().next().map(|ch| u32::from(ch as u16));
    let modifiers = native_event.modifierFlags();

    let control = modifiers.contains(NSEventModifierFlags::Control);
    let alt = modifiers.contains(NSEventModifierFlags::Option);
    let mut shift = modifiers.contains(NSEventModifierFlags::Shift);
    let command = modifiers.contains(NSEventModifierFlags::Command);
    let function = modifiers.contains(NSEventModifierFlags::Function)
        && first_char
            .is_none_or(|ch| !(NSUpArrowFunctionKey..=NSModeSwitchFunctionKey).contains(&ch));

    // 키패드는 문자로 구분되지 않으므로 keyCode 로 먼저 가른다 ([`keypad_key`]).
    // `key_char` 는 비운다 — 방향키로 번역된 키가 텍스트로도 들어가면 안 된다.
    // `function` 도 끈다: 키패드에는 fn 키가 없어 잃을 정보가 없고, Clear 키가
    // NSFunctionKeyMask 를 함께 켜 오면 바인딩이 `fn-kp_clear` 로 어긋난다.
    if let Some(key) = keypad_key(native_event.keyCode()) {
        return Keystroke {
            modifiers: Modifiers {
                control,
                alt,
                shift,
                platform: command,
                function: false,
            },
            key: key.to_string(),
            key_char: None,
        };
    }

    #[allow(non_upper_case_globals)]
    let key = match first_char {
        Some(SPACE_KEY) => {
            key_char = Some(" ".to_string());
            "space".to_string()
        }
        Some(TAB_KEY) => {
            key_char = Some("\t".to_string());
            "tab".to_string()
        }
        Some(ENTER_KEY) | Some(NUMPAD_ENTER_KEY) => {
            key_char = Some("\n".to_string());
            "enter".to_string()
        }
        Some(BACKSPACE_KEY) => "backspace".to_string(),
        Some(key) if key == u32::from(ESCAPE_KEY) => "escape".to_string(),
        Some(SHIFT_TAB_KEY) => "tab".to_string(),
        Some(NSUpArrowFunctionKey) => "up".to_string(),
        Some(NSDownArrowFunctionKey) => "down".to_string(),
        Some(NSLeftArrowFunctionKey) => "left".to_string(),
        Some(NSRightArrowFunctionKey) => "right".to_string(),
        Some(NSPageUpFunctionKey) => "pageup".to_string(),
        Some(NSPageDownFunctionKey) => "pagedown".to_string(),
        Some(NSHomeFunctionKey) => "home".to_string(),
        Some(NSEndFunctionKey) => "end".to_string(),
        Some(NSDeleteFunctionKey) => "delete".to_string(),
        // Observed Insert==NSHelpFunctionKey not NSInsertFunctionKey.
        Some(NSHelpFunctionKey) => "insert".to_string(),
        Some(NSF1FunctionKey) => "f1".to_string(),
        Some(NSF2FunctionKey) => "f2".to_string(),
        Some(NSF3FunctionKey) => "f3".to_string(),
        Some(NSF4FunctionKey) => "f4".to_string(),
        Some(NSF5FunctionKey) => "f5".to_string(),
        Some(NSF6FunctionKey) => "f6".to_string(),
        Some(NSF7FunctionKey) => "f7".to_string(),
        Some(NSF8FunctionKey) => "f8".to_string(),
        Some(NSF9FunctionKey) => "f9".to_string(),
        Some(NSF10FunctionKey) => "f10".to_string(),
        Some(NSF11FunctionKey) => "f11".to_string(),
        Some(NSF12FunctionKey) => "f12".to_string(),
        Some(NSF13FunctionKey) => "f13".to_string(),
        Some(NSF14FunctionKey) => "f14".to_string(),
        Some(NSF15FunctionKey) => "f15".to_string(),
        Some(NSF16FunctionKey) => "f16".to_string(),
        Some(NSF17FunctionKey) => "f17".to_string(),
        Some(NSF18FunctionKey) => "f18".to_string(),
        Some(NSF19FunctionKey) => "f19".to_string(),
        Some(NSF20FunctionKey) => "f20".to_string(),
        Some(NSF21FunctionKey) => "f21".to_string(),
        Some(NSF22FunctionKey) => "f22".to_string(),
        Some(NSF23FunctionKey) => "f23".to_string(),
        Some(NSF24FunctionKey) => "f24".to_string(),
        Some(NSF25FunctionKey) => "f25".to_string(),
        Some(NSF26FunctionKey) => "f26".to_string(),
        Some(NSF27FunctionKey) => "f27".to_string(),
        Some(NSF28FunctionKey) => "f28".to_string(),
        Some(NSF29FunctionKey) => "f29".to_string(),
        Some(NSF30FunctionKey) => "f30".to_string(),
        Some(NSF31FunctionKey) => "f31".to_string(),
        Some(NSF32FunctionKey) => "f32".to_string(),
        Some(NSF33FunctionKey) => "f33".to_string(),
        Some(NSF34FunctionKey) => "f34".to_string(),
        Some(NSF35FunctionKey) => "f35".to_string(),
        _ => {
            // Cases to test when modifying this:
            //
            //           qwerty key | none | cmd   | cmd-shift
            // * Armenian         s | ս    | cmd-s | cmd-shift-s  (layout is non-ASCII, so we use cmd layout)
            // * Dvorak+QWERTY    s | o    | cmd-s | cmd-shift-s  (layout switches on cmd)
            // * Ukrainian+QWERTY s | с    | cmd-s | cmd-shift-s  (macOS reports cmd-s instead of cmd-S)
            // * Czech            7 | ý    | cmd-ý | cmd-7        (layout has shifted numbers)
            // * Norwegian        7 | 7    | cmd-7 | cmd-/        (macOS reports cmd-shift-7 instead of cmd-/)
            // * Russian          7 | 7    | cmd-7 | cmd-&        (shift-7 is . but when cmd is down, should use cmd layout)
            // * German QWERTZ    ; | ö    | cmd-ö | cmd-Ö        (Zed's shift special case only applies to a-z)
            //
            let mut chars_ignoring_modifiers =
                chars_for_modified_key(native_event.keyCode(), NO_MOD);
            let mut chars_with_shift = chars_for_modified_key(native_event.keyCode(), SHIFT_MOD);
            let always_use_cmd_layout = always_use_command_layout();

            // Handle Dvorak+QWERTY / Russian / Armenian
            if command || always_use_cmd_layout {
                let chars_with_cmd = chars_for_modified_key(native_event.keyCode(), CMD_MOD);
                let chars_with_both =
                    chars_for_modified_key(native_event.keyCode(), CMD_MOD | SHIFT_MOD);

                // We don't do this in the case that the shifted command key generates
                // the same character as the unshifted command key (Norwegian, e.g.)
                if chars_with_both != chars_with_cmd {
                    chars_with_shift = chars_with_both;

                // Handle edge-case where cmd-shift-s reports cmd-s instead of
                // cmd-shift-s (Ukrainian, etc.)
                } else if chars_with_cmd.to_ascii_uppercase() != chars_with_cmd {
                    chars_with_shift = chars_with_cmd.to_ascii_uppercase();
                }
                chars_ignoring_modifiers = chars_with_cmd;
            }

            if !control && !command && !function {
                let mut mods = NO_MOD;
                if shift {
                    mods |= SHIFT_MOD;
                }
                if alt {
                    mods |= OPTION_MOD;
                }

                key_char = Some(chars_for_modified_key(native_event.keyCode(), mods));
            }

            if shift
                && chars_ignoring_modifiers
                    .chars()
                    .all(|c| c.is_ascii_lowercase())
            {
                chars_ignoring_modifiers
            } else if shift {
                shift = false;
                chars_with_shift
            } else {
                chars_ignoring_modifiers
            }
        }
    };

    Keystroke {
        modifiers: Modifiers {
            control,
            alt,
            shift,
            platform: command,
            function,
        },
        key,
        key_char,
    }
}

fn always_use_command_layout() -> bool {
    if chars_for_modified_key(0, NO_MOD).is_ascii() {
        return false;
    }

    chars_for_modified_key(0, CMD_MOD).is_ascii()
}

const NO_MOD: u32 = 0;
const CMD_MOD: u32 = 1;
const SHIFT_MOD: u32 = 2;
const OPTION_MOD: u32 = 8;

fn chars_for_modified_key(code: CGKeyCode, modifiers: u32) -> String {
    // Values from: https://github.com/phracker/MacOSX-SDKs/blob/master/MacOSX10.6.sdk/System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/Headers/Events.h#L126
    // shifted >> 8 for UCKeyTranslate
    const CG_SPACE_KEY: u16 = 49;
    // https://github.com/phracker/MacOSX-SDKs/blob/master/MacOSX10.6.sdk/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/CarbonCore.framework/Versions/A/Headers/UnicodeUtilities.h#L278
    #[allow(non_upper_case_globals)]
    const kUCKeyActionDown: u16 = 0;
    #[allow(non_upper_case_globals)]
    const kUCKeyTranslateNoDeadKeysMask: u32 = 0;

    let keyboard_type = unsafe { LMGetKbdType() as u32 };
    const BUFFER_SIZE: usize = 4;
    let mut dead_key_state = 0;
    let mut buffer: [u16; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut buffer_size: usize = 0;

    let keyboard = unsafe { TISCopyCurrentKeyboardLayoutInputSource() };
    if keyboard.is_null() {
        return "".to_string();
    }
    let layout_data = unsafe {
        TISGetInputSourceProperty(keyboard, kTISPropertyUnicodeKeyLayoutData as *const c_void)
            as CFDataRef
    };
    if layout_data.is_null() {
        unsafe {
            let _: () = msg_send![keyboard, release];
        }
        return "".to_string();
    }
    let keyboard_layout = unsafe { CFDataGetBytePtr(layout_data) };

    unsafe {
        UCKeyTranslate(
            keyboard_layout as *const c_void,
            code,
            kUCKeyActionDown,
            modifiers,
            keyboard_type,
            kUCKeyTranslateNoDeadKeysMask,
            &mut dead_key_state,
            BUFFER_SIZE,
            &mut buffer_size as *mut usize,
            &mut buffer as *mut u16,
        );
        if dead_key_state != 0 {
            UCKeyTranslate(
                keyboard_layout as *const c_void,
                CG_SPACE_KEY,
                kUCKeyActionDown,
                modifiers,
                keyboard_type,
                kUCKeyTranslateNoDeadKeysMask,
                &mut dead_key_state,
                BUFFER_SIZE,
                &mut buffer_size as *mut usize,
                &mut buffer as *mut u16,
            );
        }
        let _: () = msg_send![keyboard, release];
    }
    String::from_utf16(&buffer[..buffer_size]).unwrap_or_default()
}
