import os
import sys
from pathlib import Path


def _ensure_workdir() -> Path:
    """
    Set working directory to the folder containing this script
    so all relative paths (runs_plane/, test/, etc.) stay valid
    no matter where main.py is launched from.
    """
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    return base_dir


def _ask_file_path(initial_dir: Path | None = None) -> str:
    """
    Open a Windows file dialog to choose an image or video.
    If it fails, ask for the path in the console.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        initialdir = str(initial_dir) if initial_dir is not None else None
        filetypes = [
            (
                "Images and videos",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp "
                "*.mp4 *.avi *.mov *.mkv *.wmv *.mpg *.mpeg *.m4v",
            ),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Choose an image or video to analyze",
            filetypes=filetypes,
            initialdir=initialdir,
        )
        root.destroy()
        return path or ""
    except Exception as exc:
        print(f"Could not open file dialog ({exc}).")
        return input("Enter the path of the image or video to analyze: ").strip()


def run_image_audio_mode() -> None:
    """Run the image+audio script (image_audio_monitor)."""
    from image_audio_monitor import main as image_audio_main

    print("\n[Mode 1] Image + audio recognition (image_audio_monitor)\n")
    image_audio_main()


def run_text_mode() -> None:
    """Run the text-filtered recognition script (predict_keywords)."""
    from predict_keywords import main as keywords_main

    print("\n[Mode 2] Text-based recognition (predict_keywords)\n")
    # Let the user choose an input (default folder: test)
    source_path = _ask_file_path(initial_dir=Path("test"))
    if not source_path:
        print("No file selected. Returning to menu.")
        return

    # Temporarily override sys.argv so predict_keywords uses this source
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--source", source_path]
        keywords_main()
    finally:
        sys.argv = old_argv


def run_predict_mode() -> None:
    """Ask for a file via dialog, then run predict.py on it."""
    import predict as predict_module

    print("\n[Mode 3] Simple predict (predict.py)\n")
    # Default folder: test
    path = _ask_file_path(initial_dir=Path("test"))
    if not path:
        print("No file selected. Returning to menu.")
        return

    # Update the source used by predict.py then run its main
    predict_module.SOURCE = path
    if hasattr(predict_module, "main"):
        predict_module.main()
    else:
        print("Error: predict.py does not define a main() function.")


def run_debug_diagnostics() -> None:
    """
    Hidden debug option (choice '9').
    Tries to import key modules and checks important resources,
    printing any errors encountered.
    """
    import importlib
    import traceback

    print("\n=== Debug diagnostics (hidden option 9) ===\n")

    def check_import(name: str) -> None:
        print(f"Checking import: {name} ...", end=" ")
        try:
            importlib.import_module(name)
            print("OK")
        except Exception as exc:
            print("ERROR")
            traceback.print_exc()

    # External libraries
    for mod in ("cv2", "ultralytics", "torch", "speech_recognition", "tkinter"):
        check_import(mod)

    # Local modules
    for mod in ("image_audio_monitor", "predict", "predict_keywords"):
        check_import(mod)

    # Check YOLO weights path
    weights_path = Path("runs_plane") / "yolov8n_plane" / "weights" / "best.pt"
    print("\nChecking YOLO weights path:")
    print(f"  {weights_path} -> {'EXISTS' if weights_path.is_file() else 'MISSING'}")


def main() -> None:
    _ensure_workdir()

    print("=== Recognition main menu ===")
    print("1 - Image + audio recognition (image_audio_monitor)")
    print("2 - Text-based recognition (predict_keywords)")
    print("3 - Simple predict (predict.py)")
    choice = input("Your choice (1 / 2 / 3): ").strip()

    if choice == "1":
        run_image_audio_mode()
    elif choice == "2":
        run_text_mode()
    elif choice == "3":
        run_predict_mode()
    elif choice == "9":  # hidden debug option
        run_debug_diagnostics()
    else:
        print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Keep the cmd window open if launched by double-click
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass

