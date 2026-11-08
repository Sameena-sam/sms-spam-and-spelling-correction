"""
spell_gui.py
============
Graphical front-end for the spelling-correction system (Part A, Half 1).

Implements the GUI required by the brief:
  * A text editor limited to 500 characters.
  * A scrollable, searchable list of every word in the corpus dictionary.
  * A "Check Spelling" button that highlights misspelled words (in red).
  * Clicking a highlighted word shows correction suggestions together with
    their minimum edit distance from the wrong word, and lets you replace it.

Run:
    pip install nltk
    python spelling/spell_gui.py
"""
import tkinter as tk
from tkinter import ttk, messagebox

from spell_core import build_checker, edit_distance

MAX_CHARS = 500


class SpellApp(tk.Tk):
    def __init__(self, corpus_path: str | None = None):
        super().__init__()
        self.title("NLP Spelling Correction System")
        self.geometry("1000x640")
        self.minsize(900, 560)

        self.issues = []          # current list of detected issues
        self.active = None        # index of the issue selected for replacement

        self._build_widgets()

        # Build the language model (may take a few seconds on first run).
        self._set_status("Loading corpus and building language model...")
        self.update_idletasks()
        self.checker, self.tokens = build_checker(corpus_path)
        self._populate_dictionary()
        self._set_status(
            f"Ready - corpus: {len(self.tokens):,} tokens, "
            f"dictionary: {len(self.checker.vocab):,} words")

    # ------------------------------------------------------------------ layout
    def _build_widgets(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=3)
        root.rowconfigure(1, weight=2)

        # ---- left: editor -------------------------------------------------
        left = ttk.LabelFrame(root, text="Text Editor (max 500 characters)", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.editor = tk.Text(left, wrap="word", font=("Segoe UI", 11),
                              height=10, undo=True)
        self.editor.grid(row=0, column=0, sticky="nsew")
        ed_scroll = ttk.Scrollbar(left, command=self.editor.yview)
        ed_scroll.grid(row=0, column=1, sticky="ns")
        self.editor.config(yscrollcommand=ed_scroll.set)
        self.editor.bind("<KeyRelease>", self._enforce_limit)

        # colour tag for highlighting misspelled (non-word) tokens
        self.editor.tag_configure("nonword", foreground="#c0392b",
                                  underline=True)

        bar = ttk.Frame(left)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Check Spelling",
                   command=self.check_spelling).pack(side="left")
        ttk.Button(bar, text="Clear",
                   command=self.clear_editor).pack(side="left", padx=6)
        self.charvar = tk.StringVar(value=f"0/{MAX_CHARS} characters")
        ttk.Label(bar, textvariable=self.charvar).pack(side="right")

        # ---- right: dictionary -------------------------------------------
        right = ttk.LabelFrame(root, text="Dictionary", padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Search:").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._populate_dictionary())
        ttk.Entry(right, textvariable=self.search_var).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self.dict_list = tk.Listbox(right, activestyle="none")
        self.dict_list.grid(row=2, column=0, sticky="nsew")
        d_scroll = ttk.Scrollbar(right, command=self.dict_list.yview)
        d_scroll.grid(row=2, column=1, sticky="ns")
        self.dict_list.config(yscrollcommand=d_scroll.set)
        self.dict_count = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.dict_count).grid(
            row=3, column=0, sticky="w", pady=(4, 0))

        # ---- bottom: suggestions -----------------------------------------
        bottom = ttk.LabelFrame(
            root, text="Suggestions (click a highlighted word above)", padding=6)
        bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        bottom.rowconfigure(1, weight=1)
        bottom.columnconfigure(0, weight=1)

        self.sugg_info = tk.StringVar(
            value="No word selected. Run 'Check Spelling', then click a "
                  "highlighted (red) word.")
        ttk.Label(bottom, textvariable=self.sugg_info).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.sugg_list = tk.Listbox(bottom, height=5)
        self.sugg_list.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.sugg_list.bind("<Double-Button-1>", lambda e: self.replace_word())
        ttk.Button(bottom, text="Replace", command=self.replace_word).grid(
            row=1, column=1, sticky="n", padx=6, pady=(4, 0))

        # ---- status bar ---------------------------------------------------
        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w", padding=4).pack(fill="x", side="bottom")

    # ----------------------------------------------------------------- helpers
    def _set_status(self, msg):
        self.status.set(msg)

    def _enforce_limit(self, event=None):
        text = self.editor.get("1.0", "end-1c")
        if len(text) > MAX_CHARS:
            self.editor.delete(f"1.0+{MAX_CHARS}c", "end")
        n = len(self.editor.get("1.0", "end-1c"))
        self.charvar.set(f"{n}/{MAX_CHARS} characters")

    def _populate_dictionary(self):
        if not hasattr(self, "checker"):
            return
        words = self.checker.search_dictionary(self.search_var.get(), limit=1000)
        self.dict_list.delete(0, "end")
        for w in words:
            self.dict_list.insert("end", w)
        total = len(self.checker.vocab)
        self.dict_count.set(f"showing {len(words)} of {total:,} words")

    def clear_editor(self):
        self.editor.delete("1.0", "end")
        self._enforce_limit()
        self._clear_highlights()
        self.sugg_list.delete(0, "end")
        self.sugg_info.set("No word selected.")

    def _clear_highlights(self):
        for tag in self.editor.tag_names():
            if tag.startswith("issue") or tag == "nonword":
                self.editor.tag_remove(tag, "1.0", "end")
        self.issues = []
        self.active = None

    # --------------------------------------------------------------- checking
    def check_spelling(self):
        self._clear_highlights()
        text = self.editor.get("1.0", "end-1c")
        self.issues = self.checker.check_text(text)

        for idx, iss in enumerate(self.issues):
            start = f"1.0+{iss['start']}c"
            end = f"1.0+{iss['end']}c"
            base_tag = iss["type"]                  # "nonword" / "realword"
            click_tag = f"issue{idx}"
            self.editor.tag_add(base_tag, start, end)
            self.editor.tag_add(click_tag, start, end)
            self.editor.tag_bind(
                click_tag, "<Button-1>",
                lambda e, i=idx: self._show_suggestions(i))

        if not self.issues:
            self._set_status("No spelling issues found.")
            self.sugg_info.set("No spelling issues found.")
        else:
            self._set_status(
                f"Found {len(self.issues)} spelling error(s). "
                f"Click a highlighted word to see suggestions.")

    def _show_suggestions(self, idx):
        self.active = idx
        iss = self.issues[idx]
        self.sugg_info.set(f"'{iss['word']}'  -  spelling error. "
                           f"Double-click a suggestion (or select + Replace).")
        self.sugg_list.delete(0, "end")
        if not iss["suggestions"]:
            self.sugg_list.insert("end", "(no suggestion found)")
            return
        for word, dist in iss["suggestions"]:
            self.sugg_list.insert("end",
                                  f"{word}    (edit distance {dist})")

    def replace_word(self):
        if self.active is None:
            messagebox.showinfo("Replace", "Click a highlighted word first.")
            return
        sel = self.sugg_list.curselection()
        iss = self.issues[self.active]
        if not iss["suggestions"]:
            return
        choice = sel[0] if sel else 0
        new_word = iss["suggestions"][choice][0]

        start = f"1.0+{iss['start']}c"
        end = f"1.0+{iss['end']}c"
        self.editor.delete(start, end)
        self.editor.insert(start, new_word)

        # Re-check so offsets and highlights stay consistent.
        self.sugg_list.delete(0, "end")
        self.sugg_info.set("Replaced. Re-checked the text.")
        self.check_spelling()


if __name__ == "__main__":
    # To use your own corpus file instead of Reuters:
    #   SpellApp(corpus_path="my_corpus.txt").mainloop()
    SpellApp().mainloop()
