"""
Cross-Platform Fake Account Detection Desktop Application
Supports Instagram and Facebook fake account detection using trained SVM-RNN hybrid models
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle
import threading
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
DARK_BG       = "#0D0D0D"
CARD_BG       = "#1A1A2E"
ACCENT_BLUE   = "#1F4FFF"
WHITE         = "#FFFFFF"
LIGHT_GREY    = "#A0A0B0"
SUCCESS_GREEN = "#00D084"
DANGER_RED    = "#FF4757"
WARN_YELLOW   = "#FFD700"
INSTA_PINK    = "#E1306C"
INSTA_ORANGE  = "#F77737"
FB_BLUE       = "#1877F2"
INPUT_BG      = "#12122A"
INPUT_ERROR   = "#2A0A10"
INPUT_BORDER  = "#2E2E5A"
BTN_HOVER     = "#2563EB"

# ─────────────────────────────────────────────
#  MODEL PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTA_PATHS = {
    "svm":       os.path.join(SCRIPT_DIR, "svm_model.pkl"),
    "rnn":       os.path.join(SCRIPT_DIR, "rnn_model.h5"),
    "scaler":    os.path.join(SCRIPT_DIR, "scaler.pkl"),
    "tokenizer": os.path.join(SCRIPT_DIR, "tokenizer.pkl"),
}

FB_PATHS = {
    "svm":       os.path.join(SCRIPT_DIR, "fb_svm_model.pkl"),
    "rnn":       os.path.join(SCRIPT_DIR, "fb_rnn_model.h5"),
    "scaler":    os.path.join(SCRIPT_DIR, "fb_scaler.pkl"),
    "tokenizer": os.path.join(SCRIPT_DIR, "fb_tokenizer.pkl"),
    "metadata":  os.path.join(SCRIPT_DIR, "fb_model_metadata.pkl"),
}

MAX_ITEMS = 10   # max captions / comments allowed

# ─────────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────────
class ModelCache:
    def __init__(self):
        self._insta = None
        self._fb    = None

    def get_insta(self):
        if self._insta is None:
            self._insta = self._load_insta()
        return self._insta

    def _load_insta(self):
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            with open(INSTA_PATHS["svm"], "rb") as f:
                svm = pickle.load(f)
            rnn = load_model(INSTA_PATHS["rnn"])
            with open(INSTA_PATHS["scaler"], "rb") as f:
                scaler = pickle.load(f)
            with open(INSTA_PATHS["tokenizer"], "rb") as f:
                tok = pickle.load(f)
            return {"svm": svm, "rnn": rnn, "scaler": scaler,
                    "tokenizer": tok, "pad_sequences": pad_sequences}
        except Exception as e:
            return {"error": str(e)}

    def get_fb(self):
        if self._fb is None:
            self._fb = self._load_fb()
        return self._fb

    def _load_fb(self):
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            with open(FB_PATHS["svm"], "rb") as f:
                svm = pickle.load(f)
            rnn = load_model(FB_PATHS["rnn"])
            with open(FB_PATHS["scaler"], "rb") as f:
                scaler = pickle.load(f)
            with open(FB_PATHS["tokenizer"], "rb") as f:
                tok = pickle.load(f)
            with open(FB_PATHS["metadata"], "rb") as f:
                meta = pickle.load(f)
            return {"svm": svm, "rnn": rnn, "scaler": scaler,
                    "tokenizer": tok, "pad_sequences": pad_sequences, "meta": meta}
        except Exception as e:
            return {"error": str(e)}


MODEL_CACHE = ModelCache()

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
MAX_LENGTH = 100

def _predict_hybrid(svm, rnn, scaler, tok, pad_sequences, csv_arr, text):
    csv_scaled = scaler.transform(csv_arr.reshape(1, -1))
    svm_proba  = svm.predict_proba(csv_scaled)[:, 1]
    seq        = tok.texts_to_sequences([text])
    padded     = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    rnn_proba  = rnn.predict(padded, verbose=0).flatten()
    combined   = 0.4 * svm_proba + 0.6 * rnn_proba
    pred       = int(combined[0] >= 0.5)
    conf       = combined[0] * 100 if pred == 1 else (1 - combined[0]) * 100
    return pred, conf


def predict_instagram(fields):
    m = MODEL_CACHE.get_insta()
    if "error" in m:
        raise RuntimeError(m["error"])
    csv_keys = ["profile pic", "nums/length username", "fullname words",
                "nums/length fullname", "name==username", "description length",
                "external URL", "private", "#posts", "#followers", "#following"]
    csv_arr = np.array([float(fields.get(k, 0)) for k in csv_keys])
    text = (f"{fields.get('username','')} {fields.get('fullname','')} "
            f"{fields.get('bio','')} {fields.get('captions','')} "
            f"{fields.get('comments','')}")
    return _predict_hybrid(m["svm"], m["rnn"], m["scaler"],
                           m["tokenizer"], m["pad_sequences"], csv_arr, text)


def predict_facebook(fields, account_type):
    m = MODEL_CACHE.get_fb()
    if "error" in m:
        raise RuntimeError(m["error"])
    meta          = m["meta"]
    all_f         = meta["all_csv_features"]
    personal_excl = meta["personal_exclude_features"]
    page_excl     = meta["page_exclude_features"]
    use = [f for f in all_f if f not in (personal_excl if account_type == 0 else page_excl)]
    csv_arr = np.array([float(fields.get(k, 0)) for k in use])
    text = (f"{fields.get('fullname','')} {fields.get('bio','')} "
            f"{fields.get('work','')} {fields.get('education','')} "
            f"{fields.get('categories','')} {fields.get('captions','')} "
            f"{fields.get('comments','')}")
    return _predict_hybrid(m["svm"], m["rnn"], m["scaler"],
                           m["tokenizer"], m["pad_sequences"], csv_arr, text)


# ─────────────────────────────────────────────
#  WIDGET HELPERS
# ─────────────────────────────────────────────
def styled_button(parent, text, command, color=ACCENT_BLUE,
                  text_color=WHITE, font_size=13, pady=12, padx=28):
    btn = tk.Button(parent, text=text, command=command,
                    bg=color, fg=text_color, activebackground=BTN_HOVER,
                    activeforeground=WHITE, relief="flat", cursor="hand2",
                    font=("Segoe UI", font_size, "bold"),
                    padx=padx, pady=pady, bd=0, highlightthickness=0)
    btn.bind("<Enter>", lambda e: btn.configure(bg=BTN_HOVER))
    btn.bind("<Leave>", lambda e: btn.configure(bg=color))
    return btn


def _reset_widget(w):
    w.config(highlightbackground=INPUT_BORDER, highlightcolor=ACCENT_BLUE, bg=INPUT_BG)


def _error_widget(w):
    w.config(highlightbackground=DANGER_RED, highlightcolor=DANGER_RED, bg=INPUT_ERROR)


def make_label_entry(parent, label_text, row, col_offset=0, width=28):
    """Returns (Entry widget, StringVar)."""
    tk.Label(parent, text=label_text, bg=CARD_BG, fg=LIGHT_GREY,
             font=("Segoe UI", 10)).grid(row=row, column=col_offset,
                                          sticky="w", padx=(0, 10), pady=4)
    var = tk.StringVar()
    ent = tk.Entry(parent, textvariable=var, width=width,
                   bg=INPUT_BG, fg=WHITE, insertbackground=WHITE,
                   relief="flat", font=("Segoe UI", 11),
                   highlightthickness=1, highlightcolor=ACCENT_BLUE,
                   highlightbackground=INPUT_BORDER)
    ent.grid(row=row, column=col_offset + 1, sticky="ew", pady=4)
    ent.bind("<FocusIn>", lambda e: _reset_widget(ent))
    ent.bind("<Key>",     lambda e: _reset_widget(ent))
    return ent, var


def make_text_area(parent, label_text, row, height=2, col_offset=0, colspan=1):
    """Plain multi-line text area. Returns Text widget."""
    tk.Label(parent, text=label_text, bg=CARD_BG, fg=LIGHT_GREY,
             font=("Segoe UI", 10)).grid(row=row, column=col_offset,
                                          sticky="nw", padx=(0, 10), pady=4)
    txt = tk.Text(parent, height=height, bg=INPUT_BG, fg=WHITE,
                  insertbackground=WHITE, relief="flat", font=("Segoe UI", 11),
                  highlightthickness=1, highlightcolor=ACCENT_BLUE,
                  highlightbackground=INPUT_BORDER, wrap="word")
    txt.grid(row=row, column=col_offset + 1, columnspan=colspan,
             sticky="ew", pady=4)
    txt.bind("<FocusIn>", lambda e: _reset_widget(txt))
    txt.bind("<Key>",     lambda e: _reset_widget(txt))
    return txt


def make_multiline_box(parent, label_text, row,
                       max_lines=10, col_offset=0, colspan=1):
    """
    Text box where each line = one caption/comment (Enter adds a new line).
    Enforces max_lines. Shows a live counter. Returns (Text widget, counter Label).
    """
    # Label + hint
    lbl_frame = tk.Frame(parent, bg=CARD_BG)
    lbl_frame.grid(row=row, column=col_offset, sticky="nw", padx=(0, 10), pady=4)

    tk.Label(lbl_frame, text=label_text, bg=CARD_BG, fg=LIGHT_GREY,
             font=("Segoe UI", 10), anchor="w").pack(anchor="w")
    tk.Label(lbl_frame, text="(press Enter for next)",
             bg=CARD_BG, fg="#606080", font=("Segoe UI", 8), anchor="w").pack(anchor="w")

    counter_lbl = tk.Label(lbl_frame,
                           text=f"0 / {max_lines}",
                           bg=CARD_BG, fg=LIGHT_GREY,
                           font=("Segoe UI", 8, "bold"))
    counter_lbl.pack(anchor="w", pady=(2, 0))

    txt = tk.Text(parent, height=max_lines, bg=INPUT_BG, fg=WHITE,
                  insertbackground=WHITE, relief="flat", font=("Segoe UI", 11),
                  highlightthickness=1, highlightcolor=ACCENT_BLUE,
                  highlightbackground=INPUT_BORDER, wrap="none")
    txt.grid(row=row, column=col_offset + 1, columnspan=colspan,
             sticky="ew", pady=4)

    def _update_counter(event=None):
        _reset_widget(txt)
        lines = [ln for ln in txt.get("1.0", "end-1c").split("\n") if ln.strip()]
        count = len(lines)
        counter_lbl.config(
            text=f"{count} / {max_lines}",
            fg=DANGER_RED if count >= max_lines else LIGHT_GREY
        )
        # Block new lines when at max
        if event and event.keysym == "Return":
            if count >= max_lines:
                return "break"

    txt.bind("<KeyRelease>", _update_counter)
    txt.bind("<Return>",     _update_counter)
    txt.bind("<FocusIn>",    lambda e: _reset_widget(txt))
    return txt, counter_lbl


def get_multiline_text(widget):
    """Join non-empty lines from a multiline box with a space (for NLP)."""
    lines = [ln.strip() for ln in widget.get("1.0", "end-1c").split("\n") if ln.strip()]
    return " ".join(lines)


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate_fields(entry_list, text_list, result_frame):
    """
    entry_list : list of (Entry widget, StringVar, display_label)
    text_list  : list of (Text widget,  display_label)   — REQUIRED text areas
    Returns True if all filled, else highlights empties and shows warning banner.
    """
    missing = []

    for ent, var, lbl in entry_list:
        if not var.get().strip():
            _error_widget(ent)
            missing.append(lbl)
        else:
            _reset_widget(ent)

    for txt, lbl in text_list:
        if not txt.get("1.0", "end-1c").strip():
            _error_widget(txt)
            missing.append(lbl)
        else:
            _reset_widget(txt)

    if missing:
        for w in result_frame.winfo_children():
            w.destroy()

        banner = tk.Frame(result_frame, bg=CARD_BG,
                          highlightthickness=2, highlightbackground=WARN_YELLOW,
                          padx=22, pady=14)
        banner.pack(fill="x", pady=8)

        tk.Label(banner,
                 text="⚠️   Please fill all the necessary fields",
                 font=("Segoe UI", 13, "bold"),
                 bg=CARD_BG, fg=WARN_YELLOW).pack(anchor="w")

        tk.Label(banner,
                 text="Missing:  " + "  •  ".join(missing),
                 font=("Segoe UI", 10),
                 bg=CARD_BG, fg=LIGHT_GREY,
                 wraplength=800, justify="left").pack(anchor="w", pady=(4, 0))
        return False

    return True


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class FakeAccountApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Platform Fake Account Identification Centre")
        self.configure(bg=DARK_BG)
        self.state("zoomed")
        self.minsize(800, 600)
        self._build_welcome()

    # ── HELPERS ──────────────────────────────
    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _scrollable_frame(self):
        outer  = tk.Frame(self, bg=DARK_BG)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=DARK_BG, bd=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner  = tk.Frame(canvas, bg=DARK_BG)
        wid    = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(wid, width=canvas.winfo_width())
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        return outer, inner

    # ═════════════════════════════════════════
    #  1. WELCOME SCREEN
    # ═════════════════════════════════════════
    def _build_welcome(self):
        self._clear()
        _, inner = self._scrollable_frame()
        tk.Frame(inner, bg=ACCENT_BLUE, height=5).pack(fill="x")

        c = tk.Frame(inner, bg=DARK_BG)
        c.pack(expand=True, fill="both")

        tk.Frame(c, bg=DARK_BG, height=60).pack()
        tk.Label(c, text="🛡️", font=("Segoe UI Emoji", 64),
                 bg=DARK_BG, fg=WHITE).pack(pady=(30, 0))
        tk.Label(c,
                 text="Welcome to Multi-Platform\nFake Account Identification Centre",
                 font=("Segoe UI", 28, "bold"), bg=DARK_BG, fg=WHITE,
                 justify="center", wraplength=900).pack(pady=(18, 0))
        tk.Frame(c, bg=ACCENT_BLUE, height=3, width=440).pack(pady=8)
        tk.Label(c, text="Powered by Hybrid SVM-RNN Deep Learning Models",
                 font=("Segoe UI", 13, "italic"),
                 bg=DARK_BG, fg=LIGHT_GREY).pack(pady=(0, 40))

        bf = tk.Frame(c, bg=DARK_BG)
        bf.pack(pady=(0, 50))
        for icon, name, color in [("📸", "Instagram", INSTA_PINK),
                                   ("👤", "Facebook",  FB_BLUE)]:
            b = tk.Frame(bf, bg=CARD_BG, padx=24, pady=14)
            b.pack(side="left", padx=20)
            tk.Label(b, text=icon, font=("Segoe UI Emoji", 28),
                     bg=CARD_BG, fg=color).pack()
            tk.Label(b, text=name, font=("Segoe UI", 12, "bold"),
                     bg=CARD_BG, fg=color).pack()

        tk.Label(c, text="Press any key or click the button below to continue",
                 font=("Segoe UI", 13), bg=DARK_BG, fg=LIGHT_GREY).pack(pady=(0, 20))
        styled_button(c, "▶   Continue",
                      self._build_platform_select,
                      color=ACCENT_BLUE, font_size=14).pack(pady=10)
        tk.Frame(c, bg=DARK_BG, height=60).pack()
        self.bind("<Key>", lambda e: self._build_platform_select())

    # ═════════════════════════════════════════
    #  2. PLATFORM SELECT
    # ═════════════════════════════════════════
    def _build_platform_select(self, _=None):
        self.unbind("<Key>")
        self._clear()
        _, inner = self._scrollable_frame()
        tk.Frame(inner, bg=ACCENT_BLUE, height=5).pack(fill="x")

        c = tk.Frame(inner, bg=DARK_BG)
        c.pack(expand=True, fill="both")
        tk.Frame(c, bg=DARK_BG, height=60).pack()
        tk.Label(c, text="🔍  Choose a Platform",
                 font=("Segoe UI", 30, "bold"),
                 bg=DARK_BG, fg=WHITE).pack(pady=(20, 6))
        tk.Label(c,
                 text="Select the social media platform for fake account analysis",
                 font=("Segoe UI", 13), bg=DARK_BG, fg=LIGHT_GREY).pack(pady=(0, 50))

        cf = tk.Frame(c, bg=DARK_BG)
        cf.pack()
        self._platform_card(cf, "📸", "Instagram",
                            "Detect fake accounts using\nprofile stats & NLP analysis",
                            INSTA_PINK, self._build_instagram_screen)
        self._platform_card(cf, "👤", "Facebook",
                            "Detect fake personal accounts\n& pages using hybrid AI models",
                            FB_BLUE, self._build_facebook_screen)
        tk.Frame(c, bg=DARK_BG, height=60).pack()

    def _platform_card(self, parent, icon, name, desc, color, command):
        card = tk.Frame(parent, bg=CARD_BG, padx=40, pady=40,
                        highlightthickness=2, highlightbackground=color)
        card.pack(side="left", padx=30, pady=10)
        tk.Label(card, text=icon, font=("Segoe UI Emoji", 52),
                 bg=CARD_BG, fg=color).pack()
        tk.Label(card, text=name, font=("Segoe UI", 20, "bold"),
                 bg=CARD_BG, fg=color).pack(pady=6)
        tk.Label(card, text=desc, font=("Segoe UI", 11),
                 bg=CARD_BG, fg=LIGHT_GREY, justify="center").pack(pady=(0, 20))
        styled_button(card, f"Analyse with {name}", command,
                      color=color, font_size=12).pack()
        card.bind("<Enter>", lambda e: card.config(highlightbackground=WHITE))
        card.bind("<Leave>", lambda e: card.config(highlightbackground=color))

    # ═════════════════════════════════════════
    #  3. INSTAGRAM SCREEN
    # ═════════════════════════════════════════
    def _build_instagram_screen(self):
        self._clear()
        _, inner = self._scrollable_frame()

        hdr = tk.Frame(inner, bg=INSTA_PINK, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📸  Instagram Fake Account Detector",
                 font=("Segoe UI", 18, "bold"), bg=INSTA_PINK, fg=WHITE).pack()

        body = tk.Frame(inner, bg=DARK_BG, padx=40, pady=30)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # ── LEFT: numeric features ────────────
        left = tk.LabelFrame(body, text="  📊 Profile Statistics  ",
                             bg=CARD_BG, fg=INSTA_ORANGE,
                             font=("Segoe UI", 12, "bold"), labelanchor="n",
                             padx=20, pady=20, relief="flat",
                             highlightthickness=1, highlightbackground=INSTA_PINK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 20), pady=10)
        left.columnconfigure(1, weight=1)

        insta_csv = [
            ("profile pic",          "Has Profile Picture (0/1)"),
            ("nums/length username", "Nums / Length of Username"),
            ("fullname words",       "Full Name Word Count"),
            ("nums/length fullname", "Nums / Length of Full Name"),
            ("name==username",       "Name Equals Username (0/1)"),
            ("description length",   "Bio / Description Length"),
            ("external URL",         "Has External URL (0/1)"),
            ("private",              "Private Account (0/1)"),
            ("#posts",               "Number of Posts"),
            ("#followers",           "Number of Followers"),
            ("#following",           "Number of Following"),
        ]

        self._insta_entries       = []   # (Entry, StringVar, label) for validation
        self._insta_vars          = {}   # key → var  for prediction
        self._insta_entry_widgets = {}   # key → Entry for clear/reset

        for i, (key, label) in enumerate(insta_csv):
            ent, var = make_label_entry(left, label, i, width=22)
            self._insta_vars[key]          = var
            self._insta_entry_widgets[key] = ent
            self._insta_entries.append((ent, var, label))

        # ── RIGHT: NLP features ───────────────
        right = tk.LabelFrame(body, text="  💬 Text / NLP Features  ",
                              bg=CARD_BG, fg=INSTA_ORANGE,
                              font=("Segoe UI", 12, "bold"), labelanchor="n",
                              padx=20, pady=20, relief="flat",
                              highlightthickness=1, highlightbackground=INSTA_PINK)
        right.grid(row=0, column=1, sticky="nsew", pady=10)
        right.columnconfigure(1, weight=1)

        self._insta_text_widgets = {}

        # Required
        for i, (key, label) in enumerate([("username", "Username *"),
                                           ("fullname", "Full Name *")]):
            w = make_text_area(right, label, i, height=2, colspan=1)
            self._insta_text_widgets[key] = w

        # Optional plain text
        self._insta_text_widgets["bio"] = make_text_area(
            right, "Bio / Description", 2, height=3, colspan=1)

        # Captions — max 10, Enter-separated
        cap_w, _ = make_multiline_box(
            right, f"Post Captions  (max {MAX_ITEMS})", 3,
            max_lines=MAX_ITEMS, colspan=1)
        self._insta_text_widgets["captions"] = cap_w

        # Comments — max 10, Enter-separated
        com_w, _ = make_multiline_box(
            right, f"Comments  (max {MAX_ITEMS})", 4,
            max_lines=MAX_ITEMS, colspan=1)
        self._insta_text_widgets["comments"] = com_w

        # Required text areas for validation
        self._insta_required_text = [
            (self._insta_text_widgets["username"], "Username"),
            (self._insta_text_widgets["fullname"], "Full Name"),
        ]

        # ── Result + Buttons ──────────────────
        self._insta_result_frame = tk.Frame(body, bg=DARK_BG)
        self._insta_result_frame.grid(row=1, column=0, columnspan=2,
                                      pady=(10, 0), sticky="ew")

        bf = tk.Frame(body, bg=DARK_BG)
        bf.grid(row=2, column=0, columnspan=2, pady=20)
        styled_button(bf, "🔍  Analyse Account",
                      self._run_instagram_prediction,
                      color=INSTA_PINK, font_size=13).pack(side="left", padx=12)
        styled_button(bf, "🔄  Clear Fields",
                      self._clear_insta_fields,
                      color="#555", font_size=13).pack(side="left", padx=12)
        styled_button(bf, "⬅  Back",
                      self._build_platform_select,
                      color=ACCENT_BLUE, font_size=13).pack(side="left", padx=12)

    def _clear_insta_fields(self):
        for v in self._insta_vars.values():
            v.set("")
        for ent in self._insta_entry_widgets.values():
            _reset_widget(ent)
        for w in self._insta_text_widgets.values():
            w.delete("1.0", "end")
            _reset_widget(w)
        for w in self._insta_result_frame.winfo_children():
            w.destroy()

    def _run_instagram_prediction(self):
        ok = validate_fields(
            self._insta_entries,
            self._insta_required_text,
            self._insta_result_frame,
        )
        if not ok:
            return

        for w in self._insta_result_frame.winfo_children():
            w.destroy()
        loading = tk.Label(self._insta_result_frame,
                           text="⏳ Analysing… please wait",
                           font=("Segoe UI", 13, "italic"),
                           bg=DARK_BG, fg=WARN_YELLOW)
        loading.pack(pady=10)
        self.update()

        def run():
            try:
                fields = {k: v.get() for k, v in self._insta_vars.items()}
                for k, w in self._insta_text_widgets.items():
                    if k in ("captions", "comments"):
                        fields[k] = get_multiline_text(w)
                    else:
                        fields[k] = w.get("1.0", "end-1c").strip()
                pred, conf = predict_instagram(fields)
                self.after(0, lambda: self._show_result(
                    self._insta_result_frame, pred, conf, "Instagram", loading))
            except Exception as e:
                self.after(0, lambda: self._show_error(
                    self._insta_result_frame, str(e), loading))

        threading.Thread(target=run, daemon=True).start()

    # ═════════════════════════════════════════
    #  4. FACEBOOK SCREEN
    # ═════════════════════════════════════════
    _FB_ALL_CSV = [
        ("personal(0)/page(1)", "Personal(0) / Page(1)"),
        ("private",             "Private Account (0/1)"),
        ("#friends",            "Number of Friends"),
        ("friends visibility",  "Friends Visible (0/1)"),
        ("#followers",          "Number of Followers"),
        ("#following",          "Number of Following"),
        ("category",            "Category / Type"),
        ("#posts",              "Number of Posts"),
        ("#likes",              "Number of Likes"),
        ("#check-ins",          "Number of Check-ins"),
        ("#reviews",            "Number of Reviews"),
    ]
    _FB_NLP = [
        ("fullname",   "Full Name *"),
        ("bio",        "Bio / About"),
        ("work",       "Work / Employment"),
        ("education",  "Education"),
        ("categories", "Categories"),
    ]
    _FB_PERSONAL_HIDE = {"#followers", "#following", "category",
                         "#likes", "#check-ins", "#reviews"}
    _FB_PAGE_HIDE     = {"private", "#friends", "friends visibility"}

    def _build_facebook_screen(self):
        self._clear()
        _, inner = self._scrollable_frame()

        hdr = tk.Frame(inner, bg=FB_BLUE, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="👤  Facebook Fake Account Detector",
                 font=("Segoe UI", 18, "bold"), bg=FB_BLUE, fg=WHITE).pack()

        body = tk.Frame(inner, bg=DARK_BG, padx=40, pady=30)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Account type toggle
        tf = tk.Frame(body, bg=DARK_BG)
        tf.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        tk.Label(tf, text="Account Type:", bg=DARK_BG, fg=WHITE,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 14))
        self._fb_account_type = tk.IntVar(value=0)
        for val, lbl in [(0, "👤  Personal Account"), (1, "📄  Page")]:
            tk.Radiobutton(tf, text=lbl, variable=self._fb_account_type,
                           value=val, bg=DARK_BG, fg=WHITE,
                           selectcolor=ACCENT_BLUE, activebackground=DARK_BG,
                           activeforeground=WHITE, font=("Segoe UI", 11),
                           cursor="hand2",
                           command=self._update_fb_fields).pack(side="left", padx=14)

        self._fb_left = tk.LabelFrame(body, text="  📊 Profile Statistics  ",
                                      bg=CARD_BG, fg="#5B9BD5",
                                      font=("Segoe UI", 12, "bold"), labelanchor="n",
                                      padx=20, pady=20, relief="flat",
                                      highlightthickness=1, highlightbackground=FB_BLUE)
        self._fb_left.grid(row=1, column=0, sticky="nsew", padx=(0, 20), pady=10)
        self._fb_left.columnconfigure(1, weight=1)

        self._fb_right = tk.LabelFrame(body, text="  💬 Text / NLP Features  ",
                                       bg=CARD_BG, fg="#5B9BD5",
                                       font=("Segoe UI", 12, "bold"), labelanchor="n",
                                       padx=20, pady=20, relief="flat",
                                       highlightthickness=1, highlightbackground=FB_BLUE)
        self._fb_right.grid(row=1, column=1, sticky="nsew", pady=10)
        self._fb_right.columnconfigure(1, weight=1)

        self._fb_vars          = {}
        self._fb_entry_widgets = {}
        self._fb_entries       = []
        self._fb_text_widgets  = {}
        self._init_fb_fields()

        self._fb_result_frame = tk.Frame(body, bg=DARK_BG)
        self._fb_result_frame.grid(row=2, column=0, columnspan=2,
                                   pady=(10, 0), sticky="ew")

        bf = tk.Frame(body, bg=DARK_BG)
        bf.grid(row=3, column=0, columnspan=2, pady=20)
        styled_button(bf, "🔍  Analyse Account",
                      self._run_facebook_prediction,
                      color=FB_BLUE, font_size=13).pack(side="left", padx=12)
        styled_button(bf, "🔄  Clear Fields",
                      self._clear_fb_fields,
                      color="#555", font_size=13).pack(side="left", padx=12)
        styled_button(bf, "⬅  Back",
                      self._build_platform_select,
                      color=ACCENT_BLUE, font_size=13).pack(side="left", padx=12)

    def _init_fb_fields(self):
        for child in self._fb_left.winfo_children():
            child.destroy()
        for child in self._fb_right.winfo_children():
            child.destroy()

        self._fb_vars          = {}
        self._fb_entry_widgets = {}
        self._fb_entries       = []
        self._fb_text_widgets  = {}

        acct = self._fb_account_type.get()
        hide = self._FB_PERSONAL_HIDE if acct == 0 else self._FB_PAGE_HIDE

        row_idx = 0
        for key, label in self._FB_ALL_CSV:
            if key in hide:
                continue
            ent, var = make_label_entry(self._fb_left, label, row_idx, width=22)
            self._fb_vars[key]          = var
            self._fb_entry_widgets[key] = ent
            self._fb_entries.append((ent, var, label))
            row_idx += 1

        # Plain NLP fields
        for i, (key, label) in enumerate(self._FB_NLP):
            w = make_text_area(self._fb_right, label, i, height=2, colspan=1)
            self._fb_text_widgets[key] = w

        # Captions
        cap_w, _ = make_multiline_box(
            self._fb_right, f"Post Captions  (max {MAX_ITEMS})",
            len(self._FB_NLP), max_lines=MAX_ITEMS, colspan=1)
        self._fb_text_widgets["captions"] = cap_w

        # Comments
        com_w, _ = make_multiline_box(
            self._fb_right, f"Comments  (max {MAX_ITEMS})",
            len(self._FB_NLP) + 1, max_lines=MAX_ITEMS, colspan=1)
        self._fb_text_widgets["comments"] = com_w

        self._fb_required_text = [
            (self._fb_text_widgets["fullname"], "Full Name"),
        ]

    def _update_fb_fields(self):
        self._init_fb_fields()

    def _clear_fb_fields(self):
        for v in self._fb_vars.values():
            v.set("")
        for ent in self._fb_entry_widgets.values():
            _reset_widget(ent)
        for w in self._fb_text_widgets.values():
            w.delete("1.0", "end")
            _reset_widget(w)
        for w in self._fb_result_frame.winfo_children():
            w.destroy()

    def _run_facebook_prediction(self):
        ok = validate_fields(
            self._fb_entries,
            self._fb_required_text,
            self._fb_result_frame,
        )
        if not ok:
            return

        for w in self._fb_result_frame.winfo_children():
            w.destroy()
        loading = tk.Label(self._fb_result_frame,
                           text="⏳ Analysing… please wait",
                           font=("Segoe UI", 13, "italic"),
                           bg=DARK_BG, fg=WARN_YELLOW)
        loading.pack(pady=10)
        self.update()

        def run():
            try:
                acct_type = self._fb_account_type.get()
                fields    = {k: v.get() for k, v in self._fb_vars.items()}
                for k, w in self._fb_text_widgets.items():
                    if k in ("captions", "comments"):
                        fields[k] = get_multiline_text(w)
                    else:
                        fields[k] = w.get("1.0", "end-1c").strip()
                pred, conf = predict_facebook(fields, acct_type)
                acct_label = "Personal" if acct_type == 0 else "Page"
                self.after(0, lambda: self._show_result(
                    self._fb_result_frame, pred, conf,
                    f"Facebook {acct_label}", loading))
            except Exception as e:
                self.after(0, lambda: self._show_error(
                    self._fb_result_frame, str(e), loading))

        threading.Thread(target=run, daemon=True).start()

    # ═════════════════════════════════════════
    #  5. RESULT / ERROR DISPLAY
    # ═════════════════════════════════════════
    def _show_result(self, parent, pred, confidence, platform, loading_lbl):
        loading_lbl.destroy()
        is_fake = pred == 1
        color   = DANGER_RED if is_fake else SUCCESS_GREEN
        verdict = "⚠️  FAKE ACCOUNT DETECTED" if is_fake else "✅  REAL ACCOUNT"
        icon    = "🚨" if is_fake else "🟢"

        card = tk.Frame(parent, bg=CARD_BG,
                        highlightthickness=2, highlightbackground=color,
                        padx=30, pady=20)
        card.pack(fill="x", pady=10)

        tk.Label(card, text=icon, font=("Segoe UI Emoji", 40),
                 bg=CARD_BG, fg=color).pack()
        tk.Label(card, text=verdict, font=("Segoe UI", 20, "bold"),
                 bg=CARD_BG, fg=color).pack(pady=4)
        tk.Label(card, text=f"Platform: {platform}",
                 font=("Segoe UI", 12), bg=CARD_BG, fg=LIGHT_GREY).pack()

        bar_outer = tk.Frame(card, bg=INPUT_BG, width=400, height=16)
        bar_outer.pack(pady=12)
        bar_outer.pack_propagate(False)
        tk.Frame(bar_outer, bg=color, height=16,
                 width=int(4 * confidence)).pack(side="left")

        tk.Label(card, text=f"Confidence: {confidence:.1f}%",
                 font=("Segoe UI", 13, "bold"), bg=CARD_BG, fg=color).pack()

        note = ("This account shows strong indicators of being fake.\n"
                "Verify before engaging or trusting this profile."
                if is_fake
                else "This account appears to be authentic based on the provided data.")
        tk.Label(card, text=note, font=("Segoe UI", 11, "italic"),
                 bg=CARD_BG, fg=WARN_YELLOW if is_fake else LIGHT_GREY,
                 justify="center").pack(pady=6)

    def _show_error(self, parent, error_msg, loading_lbl):
        loading_lbl.destroy()
        ef = tk.Frame(parent, bg=CARD_BG,
                      highlightthickness=2, highlightbackground=DANGER_RED,
                      padx=24, pady=16)
        ef.pack(fill="x", pady=10)
        tk.Label(ef, text="❌  Error During Analysis",
                 font=("Segoe UI", 14, "bold"), bg=CARD_BG, fg=DANGER_RED).pack()
        tk.Label(ef, text=error_msg, font=("Segoe UI", 10),
                 bg=CARD_BG, fg=WARN_YELLOW,
                 wraplength=700, justify="left").pack(pady=6)
        tk.Label(ef,
                 text="Ensure all model files are in the same folder as this script,\n"
                      "and that TensorFlow / scikit-learn are installed.",
                 font=("Segoe UI", 10, "italic"),
                 bg=CARD_BG, fg=LIGHT_GREY, justify="center").pack()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = FakeAccountApp()
    app.mainloop()