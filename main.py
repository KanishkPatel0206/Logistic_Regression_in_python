# -*- coding: utf-8 -*-
"""
Breast Cancer Classifier — Kivy GUI
A polished dark-theme desktop app with:
  • Hyperparameter controls (sliders + dropdowns)
  • Live training with progress
  • Metrics dashboard
  • Tabbed charts: Confusion Matrix, Feature Importance, ROC Curve, CV Scores
  • Export results to PNG files
"""

import os
import io
import threading

os.environ["KIVY_NO_ENV_CONFIG"] = "1"

# ── Kivy environment tweaks (must come before any kivy import) ──────────────
os.environ.setdefault("KIVY_WINDOW", "sdl2")

import kivy
kivy.require("2.0.0")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.image import Image as CoreImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image as KivyImage
from kivy.uix.progressbar import ProgressBar
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.widget import Widget
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.metrics import dp
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.lang import Builder

# ── Dark theme palette ───────────────────────────────────────────────────────
BG        = (0.118, 0.118, 0.180, 1)   # #1e1e2e
SURFACE   = (0.165, 0.165, 0.243, 1)   # #2a2a3e
ACCENT    = (0.537, 0.706, 0.980, 1)   # #89b4fa  (blue)
GREEN     = (0.651, 0.890, 0.631, 1)   # #a6e3a1
RED       = (0.953, 0.545, 0.659, 1)   # #f38ba8
YELLOW    = (0.976, 0.886, 0.686, 1)   # #f9e2af
TEXT      = (0.804, 0.839, 0.957, 1)   # #cdd6f4
SUBTEXT   = (0.647, 0.678, 0.796, 1)   # #a6adc8
BORDER    = (0.333, 0.333, 0.467, 1)   # #555577

Window.clearcolor = BG
Window.size = (1180, 760)
Window.minimum_width  = 960
Window.minimum_height = 640


# ── KV language helper widgets ───────────────────────────────────────────────
KV = """
<RoundedBox@Widget>:
    canvas.before:
        Color:
            rgba: self.bg_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.radius]
    bg_color: (0.165, 0.165, 0.243, 1)
    radius: 12

<StyledLabel@Label>:
    color: (0.804, 0.839, 0.957, 1)
    font_size: '13sp'

<StyledButton@Button>:
    background_normal: ''
    background_down: ''
    background_color: (0.537, 0.706, 0.980, 1)
    color: (0.118, 0.118, 0.180, 1)
    font_size: '13sp'
    bold: True
    size_hint_y: None
    height: '40dp'
    canvas.before:
        Color:
            rgba: (0.400, 0.560, 0.820, 1) if self.state == 'down' else self.background_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [8]

<MetricCard@BoxLayout>:
    orientation: 'vertical'
    padding: '12dp'
    spacing: '4dp'
    canvas.before:
        Color:
            rgba: (0.165, 0.165, 0.243, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [10]
"""
Builder.load_string(KV)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_texture_from_bytes(png_bytes: bytes):
    """Convert raw PNG bytes → Kivy CoreImage texture."""
    buf = io.BytesIO(png_bytes)
    ci  = CoreImage(buf, ext="png")
    return ci.texture


def hex_to_rgba(h: str, a: float = 1.0):
    h = h.lstrip("#")
    r, g, b = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return (r, g, b, a)


# ── Custom Widgets ────────────────────────────────────────────────────────────

class MetricCard(BoxLayout):
    """A single KPI card showing a label + big number."""
    def __init__(self, title: str, value: str, color=GREEN, **kwargs):
        super().__init__(orientation="vertical", padding=dp(12),
                         spacing=dp(4), **kwargs)
        with self.canvas.before:
            Color(*SURFACE)
            self._rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
        self.bind(pos=self._upd, size=self._upd)

        self._title_lbl = Label(text=title, color=SUBTEXT, font_size="12sp",
                                size_hint_y=None, height=dp(20), halign="center")
        self._title_lbl.bind(size=self._title_lbl.setter("text_size"))

        self._val_lbl = Label(text=value, color=color, font_size="22sp",
                              bold=True, size_hint_y=None, height=dp(36),
                              halign="center")
        self._val_lbl.bind(size=self._val_lbl.setter("text_size"))

        self.add_widget(self._title_lbl)
        self.add_widget(self._val_lbl)

    def _upd(self, *_):
        self._rect.pos  = self.pos
        self._rect.size = self.size

    def update(self, value: str):
        self._val_lbl.text = value


class ChartImage(BoxLayout):
    """Wrapper that holds a KivyImage and overlays a placeholder label."""
    def __init__(self, placeholder="No chart yet — run training first.", **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*SURFACE)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)])
        self.bind(pos=self._upd, size=self._upd)

        self._img = KivyImage(allow_stretch=True, keep_ratio=True)
        self._ph  = Label(text=placeholder, color=SUBTEXT, font_size="13sp",
                          halign="center", valign="middle")
        self._ph.bind(size=self._ph.setter("text_size"))
        self.add_widget(self._ph)

    def _upd(self, *_):
        self._bg.pos  = self.pos
        self._bg.size = self.size

    def set_texture(self, texture):
        if self._ph in self.children:
            self.remove_widget(self._ph)
        if self._img not in self.children:
            self.add_widget(self._img)
        self._img.texture = texture


class SectionHeader(Label):
    def __init__(self, text, **kwargs):
        super().__init__(text=text, color=ACCENT, font_size="14sp", bold=True,
                         size_hint_y=None, height=dp(32),
                         halign="left", valign="middle", **kwargs)
        self.bind(size=self.setter("text_size"))


class HRule(Widget):
    def __init__(self, **kwargs):
        super().__init__(size_hint_y=None, height=dp(1), **kwargs)
        with self.canvas:
            Color(*BORDER)
            self._line = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self._line, "pos",  self.pos),
                  size=lambda *_: setattr(self._line, "size", self.size))


class ParamRow(BoxLayout):
    """A labelled slider row."""
    def __init__(self, label: str, min_v, max_v, step, value, fmt="{:.2f}", **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None,
                         height=dp(48), spacing=dp(8), **kwargs)
        self._fmt = fmt
        lbl = Label(text=label, color=TEXT, font_size="12sp",
                    size_hint_x=0.38, halign="right", valign="middle")
        lbl.bind(size=lbl.setter("text_size"))

        self.slider = Slider(min=min_v, max=max_v, step=step, value=value,
                             size_hint_x=0.42,
                             cursor_image="", cursor_width=dp(16),
                             cursor_height=dp(16))
        # Kivy default slider colours via canvas override
        self.slider.bind(value=self._on_value)

        self._val_lbl = Label(text=fmt.format(value), color=YELLOW,
                              font_size="12sp", size_hint_x=0.20,
                              halign="left", valign="middle")
        self._val_lbl.bind(size=self._val_lbl.setter("text_size"))

        self.add_widget(lbl)
        self.add_widget(self.slider)
        self.add_widget(self._val_lbl)

    def _on_value(self, inst, val):
        self._val_lbl.text = self._fmt.format(val)

    @property
    def value(self):
        return self.slider.value


# ── Main App Layout ───────────────────────────────────────────────────────────

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", spacing=dp(12),
                         padding=dp(12), **kwargs)
        self._results = None
        self._build_ui()

    # ─── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        # LEFT PANEL — controls
        left = BoxLayout(orientation="vertical", size_hint_x=0.28,
                         spacing=dp(8), padding=dp(0))
        with left.canvas.before:
            Color(*SURFACE)
            self._left_bg = RoundedRectangle(pos=left.pos, size=left.size, radius=[dp(12)])
        left.bind(pos=lambda *_: setattr(self._left_bg, "pos",  left.pos),
                  size=lambda *_: setattr(self._left_bg, "size", left.size))

        scroll = ScrollView(size_hint=(1, 1))
        inner = BoxLayout(orientation="vertical", spacing=dp(6),
                          padding=dp(14), size_hint_y=None)
        inner.bind(minimum_height=inner.setter("height"))

        # Title
        title = Label(
            text="[b]Logistic Regression[/b]\n[color=a6adc8][size=11sp]Breast Cancer Classifier[/size][/color]",
            markup=True, color=ACCENT, font_size="15sp",
            size_hint_y=None, height=dp(52), halign="center"
        )
        title.bind(size=title.setter("text_size"))
        inner.add_widget(title)
        inner.add_widget(HRule())

        # — Hyperparameters section ——
        inner.add_widget(SectionHeader("Hyperparameters"))

        self._test_size  = ParamRow("Test Size",     0.10, 0.40, 0.05, 0.20, "{:.2f}")
        self._c_param    = ParamRow("Regularisation C", 0.01, 10.0, 0.01, 1.0,  "{:.2f}")
        self._max_iter   = ParamRow("Max Iterations",  500, 20000, 500, 10000, "{:.0f}")
        self._cv_folds   = ParamRow("CV Folds",          3,    10,   1,     5, "{:.0f}")
        self._rand_state = ParamRow("Random State",      0,   100,   1,    42, "{:.0f}")

        for w in [self._test_size, self._c_param, self._max_iter,
                  self._cv_folds, self._rand_state]:
            inner.add_widget(w)

        inner.add_widget(HRule())
        inner.add_widget(SectionHeader("Solver"))

        self._solver_spin = Spinner(
            text="lbfgs",
            values=["lbfgs", "liblinear", "saga", "sag", "newton-cg"],
            size_hint_y=None, height=dp(38),
            background_normal="", background_down="",
            background_color=SURFACE, color=TEXT, font_size="13sp",
        )
        inner.add_widget(self._solver_spin)
        inner.add_widget(HRule())

        # — Train button ———
        self._train_btn = Button(
            text="▶  Train Model",
            background_normal="", background_down="",
            background_color=ACCENT,
            color=BG,
            font_size="14sp", bold=True,
            size_hint_y=None, height=dp(46),
        )
        self._train_btn.bind(on_press=self._start_training)
        with self._train_btn.canvas.before:
            Color(*ACCENT)
            self._btn_rect = RoundedRectangle(
                pos=self._train_btn.pos, size=self._train_btn.size, radius=[dp(10)]
            )
        self._train_btn.bind(
            pos=lambda *_: setattr(self._btn_rect, "pos",  self._train_btn.pos),
            size=lambda *_: setattr(self._btn_rect, "size", self._train_btn.size),
        )
        inner.add_widget(self._train_btn)

        # Progress bar
        self._progress = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(8))
        inner.add_widget(self._progress)

        # Status label
        self._status_lbl = Label(
            text="Ready — adjust parameters and press Train.",
            color=SUBTEXT, font_size="11sp",
            size_hint_y=None, height=dp(48),
            halign="center", valign="top", text_size=(None, None),
        )
        self._status_lbl.bind(size=self._status_lbl.setter("text_size"))
        inner.add_widget(self._status_lbl)

        inner.add_widget(HRule())
        inner.add_widget(SectionHeader("Export"))

        export_btn = Button(
            text="💾  Save All Charts",
            background_normal="", background_down="",
            background_color=SURFACE, color=TEXT,
            font_size="13sp",
            size_hint_y=None, height=dp(40),
        )
        export_btn.bind(on_press=self._export_charts)
        with export_btn.canvas.before:
            Color(*SURFACE)
            self._exp_rect = RoundedRectangle(
                pos=export_btn.pos, size=export_btn.size, radius=[dp(8)]
            )
        export_btn.bind(
            pos=lambda *_: setattr(self._exp_rect, "pos",  export_btn.pos),
            size=lambda *_: setattr(self._exp_rect, "size", export_btn.size),
        )
        inner.add_widget(export_btn)
        inner.add_widget(Widget(size_hint_y=None, height=dp(8)))

        scroll.add_widget(inner)
        left.add_widget(scroll)
        self.add_widget(left)

        # RIGHT PANEL — metrics + charts
        right = BoxLayout(orientation="vertical", size_hint_x=0.72, spacing=dp(10))
        self.add_widget(right)

        # Metrics row
        metrics_row = GridLayout(cols=5, spacing=dp(8),
                                 size_hint_y=None, height=dp(80))
        self._cards = {}
        card_defs = [
            ("accuracy",  "Accuracy",  "--",   GREEN),
            ("precision", "Precision", "--",   ACCENT),
            ("recall",    "Recall",    "--",   YELLOW),
            ("f1",        "F1 Score",  "--",   RED),
            ("roc_auc",   "ROC AUC",   "--",   (0.800, 0.620, 0.980, 1)),
        ]
        for key, title, val, color in card_defs:
            card = MetricCard(title, val, color=color)
            self._cards[key] = card
            metrics_row.add_widget(card)
        right.add_widget(metrics_row)

        # CV info row
        cv_row = BoxLayout(orientation="horizontal", size_hint_y=None,
                           height=dp(32), spacing=dp(16))
        self._cv_lbl = Label(
            text="Cross-Validation: run training to see results",
            color=SUBTEXT, font_size="12sp", halign="left"
        )
        self._cv_lbl.bind(size=self._cv_lbl.setter("text_size"))
        cv_row.add_widget(self._cv_lbl)
        right.add_widget(cv_row)

        # Tabbed charts
        tabs = TabbedPanel(do_default_tab=False, tab_width=dp(180),
                           background_color=SURFACE)

        chart_tabs = [
            ("Confusion Matrix",    "_cm_chart"),
            ("Feature Importance",  "_fi_chart"),
            ("ROC Curve",           "_roc_chart"),
            ("CV Scores",           "_cv_chart"),
        ]
        for tab_text, attr in chart_tabs:
            tab = TabbedPanelItem(text=tab_text, font_size="12sp")
            chart = ChartImage(placeholder=f"{tab_text} — run training first.")
            setattr(self, attr, chart)
            tab.add_widget(chart)
            tabs.add_widget(tab)

        tabs.default_tab = tabs.tab_list[-1]  # select first tab
        right.add_widget(tabs)

    # ─── Training ────────────────────────────────────────────────────────────

    def _start_training(self, *_):
        self._train_btn.disabled = True
        self._train_btn.text = "Training…"
        self._progress.value = 0
        self._status_lbl.text = "Loading data…"
        Clock.schedule_once(lambda dt: self._animate_progress(0), 0.05)
        t = threading.Thread(target=self._run_training, daemon=True)
        t.start()

    def _animate_progress(self, step, *_):
        """Fake progress animation while training runs."""
        if self._progress.value < 85:
            self._progress.value += 5
            Clock.schedule_once(lambda dt: self._animate_progress(step + 1), 0.12)

    def _run_training(self):
        from src.logistic_regression import (
            train_and_evaluate,
            plot_confusion_matrix,
            plot_feature_importance,
            plot_roc_curve,
            plot_cv_scores,
        )
        try:
            results = train_and_evaluate(
                test_size=self._test_size.value,
                C=self._c_param.value,
                max_iter=int(self._max_iter.value),
                cv_folds=int(self._cv_folds.value),
                random_state=int(self._rand_state.value),
                solver=self._solver_spin.text,
            )
            # Generate plot PNGs in bytes
            results["_cm_bytes"]  = plot_confusion_matrix(results["confusion_matrix"])
            results["_fi_bytes"]  = plot_feature_importance(results["top_features"])
            results["_roc_bytes"] = plot_roc_curve(results["fpr"], results["tpr"], results["roc_auc"])
            results["_cv_bytes"]  = plot_cv_scores(results["cv_scores"])
            Clock.schedule_once(lambda dt: self._on_training_done(results), 0)
        except Exception as exc:
            import traceback
            msg = traceback.format_exc()
            Clock.schedule_once(lambda dt: self._on_training_error(str(exc), msg), 0)

    def _on_training_done(self, results):
        self._results = results
        self._progress.value = 100

        # Update metric cards
        self._cards["accuracy"].update(f"{results['accuracy']:.1f}%")
        self._cards["precision"].update(f"{results['precision']:.1f}%")
        self._cards["recall"].update(f"{results['recall']:.1f}%")
        self._cards["f1"].update(f"{results['f1']:.1f}%")
        self._cards["roc_auc"].update(f"{results['roc_auc']:.1f}%")

        # CV label
        self._cv_lbl.text = (
            f"Cross-Validation ({results['params']['cv_folds']}-fold):  "
            f"Mean = {results['cv_mean']:.1f}%   "
            f"Std = ±{results['cv_std']:.1f}%   "
            f"Scores: {', '.join(f'{s:.1f}' for s in results['cv_scores'])}"
        )

        # Update charts
        self._cm_chart.set_texture(make_texture_from_bytes(results["_cm_bytes"]))
        self._fi_chart.set_texture(make_texture_from_bytes(results["_fi_bytes"]))
        self._roc_chart.set_texture(make_texture_from_bytes(results["_roc_bytes"]))
        self._cv_chart.set_texture(make_texture_from_bytes(results["_cv_bytes"]))

        self._status_lbl.text = (
            f"✓ Done! Train: {results['train_samples']} | "
            f"Test: {results['test_samples']} samples"
        )
        self._train_btn.text = "▶  Train Model"
        self._train_btn.disabled = False

    def _on_training_error(self, short, full):
        self._progress.value = 0
        self._status_lbl.text = f"[ERROR] {short}"
        self._train_btn.text = "▶  Train Model"
        self._train_btn.disabled = False
        print(full)

    # ─── Export ──────────────────────────────────────────────────────────────

    def _export_charts(self, *_):
        if self._results is None:
            self._status_lbl.text = "Run training before exporting."
            return
        from src.logistic_regression import (
            plot_confusion_matrix,
            plot_feature_importance,
            plot_roc_curve,
            plot_cv_scores,
        )
        r = self._results
        out = "results"
        os.makedirs(out, exist_ok=True)
        plot_confusion_matrix(r["confusion_matrix"],   f"{out}/confusion_matrix.png")
        plot_feature_importance(r["top_features"],     f"{out}/feature_importance.png")
        plot_roc_curve(r["fpr"], r["tpr"], r["roc_auc"], f"{out}/roc_curve.png")
        plot_cv_scores(r["cv_scores"],                 f"{out}/cv_scores.png")
        self._status_lbl.text = f"✓ Charts saved to: {os.path.abspath(out)}/"


# ── App ───────────────────────────────────────────────────────────────────────

class LogisticRegressionApp(App):
    title = "Breast Cancer Classifier — Logistic Regression"

    def build(self):
        return MainLayout()

    def on_start(self):
        # Auto-run first training after short delay
        Clock.schedule_once(
            lambda dt: self.root._start_training(), 0.8
        )


if __name__ == "__main__":
    LogisticRegressionApp().run()
