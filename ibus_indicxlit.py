#!/usr/bin/python3
import glob
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
venv_sites = glob.glob(str(BASE_DIR / "venv/lib/python*/site-packages"))
if venv_sites:
    sys.path.insert(0, venv_sites[0])
sys.path.insert(0, str(BASE_DIR))

import gi

gi.require_version("IBus", "1.0")
from gi.repository import GLib, IBus

from kn_onnx_runtime import KannadaOnnxTransliterator

DEBOUNCE_MS = 70
MAX_CANDIDATES = 5
IGNORE_MODS = (
    IBus.ModifierType.CONTROL_MASK
    | IBus.ModifierType.MOD1_MASK
    | IBus.ModifierType.MOD4_MASK
    | IBus.ModifierType.SUPER_MASK
)

_SHARED_XLIT: KannadaOnnxTransliterator | None = None


def get_xlit() -> KannadaOnnxTransliterator:
    global _SHARED_XLIT
    if _SHARED_XLIT is None:
        _SHARED_XLIT = KannadaOnnxTransliterator()
    return _SHARED_XLIT


class IndicXlitKannadaEngine(IBus.Engine):
    __gtype_name__ = "IndicXlitKannadaEngine"

    def __init__(self):
        super().__init__()
        self.xlit = get_xlit()
        self.buffer = ""
        self.candidates: list[str] = []
        self.debounce_id: int | None = None
        self.lookup_table = IBus.LookupTable.new(
            page_size=MAX_CANDIDATES, cursor_pos=0, cursor_visible=True, round=True
        )

    def _cancel_debounce(self):
        if self.debounce_id is not None:
            GLib.source_remove(self.debounce_id)
            self.debounce_id = None

    def _clear_state(self):
        self._cancel_debounce()
        self.buffer = ""
        self.candidates = []
        self.lookup_table.clear()
        self.lookup_table.set_cursor_pos(0)
        self.hide_preedit_text()
        self.hide_lookup_table()

    def _render_ui(self):
        idx = self.lookup_table.get_cursor_pos()
        current = self.candidates[idx] if idx < len(self.candidates) else self.buffer
        preedit = IBus.Text.new_from_string(current)
        preedit.append_attribute(
            IBus.AttrType.UNDERLINE, IBus.AttrUnderline.SINGLE, 0, len(current)
        )
        self.update_preedit_text(preedit, len(current), True)
        self.update_lookup_table(self.lookup_table, True)

    def _run_translit_now(self):
        self.debounce_id = None
        self.lookup_table.clear()
        self.lookup_table.set_cursor_pos(0)
        if not self.buffer:
            self.candidates = []
            self.hide_preedit_text()
            self.hide_lookup_table()
            return False

        self.candidates = list(self.xlit.translit_word(self.buffer, beam_width=4))
        for cand in self.candidates:
            self.lookup_table.append_candidate(IBus.Text.new_from_string(cand))

        self._render_ui()
        return False

    def _schedule_update(self):
        self._cancel_debounce()
        self.lookup_table.set_cursor_pos(0)
        if not self.buffer:
            self._run_translit_now()
            return

        preedit = IBus.Text.new_from_string(self.buffer)
        preedit.append_attribute(
            IBus.AttrType.UNDERLINE, IBus.AttrUnderline.SINGLE, 0, len(self.buffer)
        )
        self.update_preedit_text(preedit, len(self.buffer), True)
        self.debounce_id = GLib.timeout_add(DEBOUNCE_MS, self._run_translit_now)

    def _commit_current(self, suffix: str = "", learn: bool = False):
        if self.debounce_id is not None:
            self._cancel_debounce()
            if self.buffer:
                self.candidates = list(self.xlit.translit_word(self.buffer, beam_width=4))
                self.lookup_table.set_cursor_pos(0)

        if self.candidates:
            idx = self.lookup_table.get_cursor_pos()
            chosen = self.candidates[idx] if idx < len(self.candidates) else self.buffer
            if learn and idx > 0:
                self.xlit.record_choice(self.buffer, chosen)
        else:
            chosen = self.buffer

        self.commit_text(IBus.Text.new_from_string(chosen + suffix))
        self._clear_state()

    def do_process_key_event(self, keyval, keycode, state):
        if state & IBus.ModifierType.RELEASE_MASK:
            return False
        if state & IGNORE_MODS:
            return False

        if self.buffer and IBus.KEY_1 <= keyval <= (IBus.KEY_1 + MAX_CANDIDATES - 1):
            if self.debounce_id is not None:
                self._run_translit_now()
            choice = keyval - IBus.KEY_1
            if choice < len(self.candidates):
                self.lookup_table.set_cursor_pos(choice)
                self._commit_current(learn=True)
                return True

        if self.buffer and keyval == IBus.KEY_Down:
            if self.debounce_id is not None:
                self._run_translit_now()
            self.lookup_table.cursor_down()
            self._render_ui()
            return True
        if self.buffer and keyval == IBus.KEY_Up:
            if self.debounce_id is not None:
                self._run_translit_now()
            self.lookup_table.cursor_up()
            self._render_ui()
            return True

        if self.buffer and keyval == IBus.KEY_BackSpace:
            self.buffer = self.buffer[:-1]
            self._schedule_update()
            return True

        if self.buffer and keyval == IBus.KEY_Escape:
            self._cancel_debounce()
            self.candidates = []
            self._commit_current()
            return True

        if self.buffer and keyval in (IBus.KEY_space, IBus.KEY_Return):
            self._commit_current(" " if keyval == IBus.KEY_space else "", learn=True)
            return True

        if (IBus.KEY_a <= keyval <= IBus.KEY_z) or (IBus.KEY_A <= keyval <= IBus.KEY_Z):
            self.buffer += chr(keyval)
            self._schedule_update()
            return True

        if self.buffer and 33 <= keyval <= 126:
            self._commit_current(chr(keyval), learn=True)
            return True

        return False

    def do_focus_out(self):
        if self.buffer:
            self._commit_current()

    def do_reset(self):
        self._clear_state()

    def do_disable(self):
        self._clear_state()


def main():
    IBus.init()
    bus = IBus.Bus()
    if not bus.is_connected():
        sys.exit("IBus daemon is not running")

    factory = IBus.Factory.new(bus.get_connection())
    factory.add_engine("indicxlit-kn", IndicXlitKannadaEngine)

    if "--ibus" in sys.argv:
        bus.request_name("org.freedesktop.IBus.IndicXlitKn", 0)
    else:
        desc = IBus.EngineDesc.new(
            "indicxlit-kn",
            "Kannada (IndicXlit)",
            "Neural Kannada Transliteration",
            "kn",
            "MIT",
            "Anish",
            "",
            "us",
        )
        comp = IBus.Component.new(
            "org.freedesktop.IBus.IndicXlitKn",
            "IndicXlit Kannada",
            "1.0",
            "MIT",
            "Anish",
            "",
            "",
            "indicxlit-kn",
        )
        comp.add_engine(desc)
        bus.register_component(comp)
        bus.set_global_engine("indicxlit-kn")

    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
