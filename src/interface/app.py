"""
Issue #11 — Selfie upload & results display UI.

A simple local desktop/kiosk app: pick an event, upload a selfie, see
confident and possible matches as thumbnails. Built with Tkinter
(Python's standard-library GUI toolkit) specifically so the project adds
zero new dependencies for this — no Flask/Streamlit server to run,
matching the proposal's "desktop/local kiosk workflow only" scope.

Run with:
    python -m src.interface.app
or:
    from src.interface import launch_app; launch_app()
"""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from config import EVENTS_DIR
from src.matching import match_selfie, NoFaceDetectedError, EventNotIndexedError

THUMBNAIL_SIZE = (160, 160)
SELFIE_PREVIEW_SIZE = (96, 96)
RESULTS_PER_ROW = 4
SUPPORTED_IMAGE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG *.BMP *.WEBP"),
)


class PhotoMatchApp:
    """
    Main application window. Kept as a plain class (not a bigger
    framework) so it's easy to follow: __init__ builds the widgets,
    the rest are event handlers.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PhotoMatch — Find Your Photos")
        self.root.geometry("900x650")

        self.selfie_path: Path | None = None
        self._selfie_preview_ref = None
        self._thumbnail_refs = []  # keep PhotoImage references alive

        self._build_widgets()
        self._refresh_event_list()

    # -- widget construction -------------------------------------------------

    def _build_widgets(self):
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Event:").pack(side="left")
        self.event_var = tk.StringVar()
        self.event_dropdown = ttk.Combobox(
            top_frame, textvariable=self.event_var, state="readonly", width=30
        )
        self.event_dropdown.pack(side="left", padx=(6, 20))

        self.selfie_preview = ttk.Label(top_frame, text="No selfie\nselected", anchor="center")
        self.selfie_preview.pack(side="left", padx=(0, 10))

        ttk.Button(top_frame, text="Choose Selfie...", command=self._choose_selfie).pack(
            side="left", padx=(0, 10)
        )
        self.search_button = ttk.Button(top_frame, text="Search", command=self._run_search)
        self.search_button.pack(side="left")

        self.status_var = tk.StringVar(value="Select an event and a selfie to begin.")
        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 0)).pack(fill="x")

        # Scrollable results area
        results_container = ttk.Frame(self.root)
        results_container.pack(fill="both", expand=True, padx=12, pady=12)

        canvas = tk.Canvas(results_container, borderwidth=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
        self.results_frame = ttk.Frame(canvas)

        self.results_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        results_window = canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw"
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(results_window, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # -- event handlers -------------------------------------------------------

    def _refresh_event_list(self):
        """List event_ids by scanning data/events/ for subfolders."""
        if not EVENTS_DIR.exists():
            events = []
        else:
            events = sorted(p.name for p in EVENTS_DIR.iterdir() if p.is_dir())

        self.event_dropdown["values"] = events
        if events:
            self.event_dropdown.current(0)

    def _choose_selfie(self):
        path = filedialog.askopenfilename(
            title="Choose a selfie",
            filetypes=SUPPORTED_IMAGE_TYPES,
        )
        if path:
            selfie_path = Path(path)
            preview = self._load_photo_image(selfie_path, SELFIE_PREVIEW_SIZE)
            if preview is None:
                messagebox.showerror(
                    "Invalid selfie", "That file could not be opened as an image."
                )
                return

            self.selfie_path = selfie_path
            self._selfie_preview_ref = preview
            self.selfie_preview.config(image=preview, text=self.selfie_path.name, compound="top")
            self.status_var.set("Selfie selected. Choose an event, then press Search.")

    def _run_search(self):
        event_id = self.event_var.get()
        if not event_id:
            messagebox.showwarning("No event selected", "Please choose an event first.")
            return
        if self.selfie_path is None:
            messagebox.showwarning("No selfie chosen", "Please choose a selfie photo first.")
            return

        self.search_button.config(state="disabled")
        self.status_var.set("Searching...")
        self._clear_results()

        # Run the (potentially slow) matching pipeline off the UI thread so
        # the window doesn't freeze while MTCNN/deepface/FAISS run.
        thread = threading.Thread(
            target=self._search_worker, args=(event_id, self.selfie_path), daemon=True
        )
        thread.start()

    def _search_worker(self, event_id: str, selfie_path: Path):
        try:
            selfie_image = cv2.imread(str(selfie_path))
            if selfie_image is None:
                raise ValueError(f"Could not read image: {selfie_path}")

            results = match_selfie(selfie_image, event_id)
            self.root.after(0, self._show_results, results)

        except NoFaceDetectedError:
            self.root.after(
                0,
                self._show_error,
                "No face was found in that selfie. Please retake it with your "
                "face clearly visible and try again.",
            )
        except EventNotIndexedError:
            self.root.after(
                0,
                self._show_error,
                f"Event '{event_id}' isn't ready for search yet — it hasn't "
                "been indexed. Ask staff to run indexing for this event.",
            )
        except Exception as exc:  # noqa: BLE001 — show *something* rather than freeze
            self.root.after(0, self._show_error, f"Something went wrong: {exc}")

    # -- results rendering ------------------------------------------------------

    def _clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._thumbnail_refs.clear()

    def _show_error(self, message: str):
        self.status_var.set(message)
        self.search_button.config(state="normal")

    def _show_results(self, results: dict):
        confident = results["confident"]
        possible = results["possible"]

        self.status_var.set(
            f"Found {len(confident)} confident match(es) and "
            f"{len(possible)} possible match(es)."
        )
        self.search_button.config(state="normal")

        self._render_tier("Confident Matches", confident)
        self._render_tier("Possible Matches", possible)

        if not confident and not possible:
            ttk.Label(
                self.results_frame,
                text="No matches found in this event. Try a different selfie or event.",
                padding=20,
            ).pack()

    def _render_tier(self, title: str, matches: list):
        if not matches:
            return

        ttk.Label(
            self.results_frame, text=f"{title} ({len(matches)})", font=("", 13, "bold")
        ).pack(anchor="w", pady=(10, 4))

        grid = ttk.Frame(self.results_frame)
        grid.pack(anchor="w")

        for i, match in enumerate(matches):
            row, col = divmod(i, RESULTS_PER_ROW)
            cell = ttk.Frame(grid, padding=6)
            cell.grid(row=row, column=col)

            thumb = self._load_thumbnail(match.photo_path)
            if thumb is not None:
                label = ttk.Label(cell, image=thumb)
                label.pack()
                self._thumbnail_refs.append(thumb)  # prevent garbage collection
            else:
                ttk.Label(cell, text="Preview unavailable", width=20, anchor="center").pack()

            ttk.Label(cell, text=f"{match.score:.2f}").pack()

    def _load_thumbnail(self, photo_path: str):
        return self._load_photo_image(photo_path, THUMBNAIL_SIZE)

    @staticmethod
    def _load_photo_image(photo_path: str | Path, size: tuple[int, int]):
        """Load an EXIF-corrected Tk image without keeping the file handle open."""
        try:
            with Image.open(photo_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail(size)
                return ImageTk.PhotoImage(image.copy())
        except (OSError, ValueError):
            return None  # broken/missing file — skip the thumbnail, not the whole result


def launch_app():
    """Entry point: creates the root Tk window and starts the app."""
    root = tk.Tk()
    PhotoMatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
