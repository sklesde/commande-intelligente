import importlib
import sys
from pathlib import Path
from types import ModuleType


def import_speech_recognition() -> ModuleType:
    """Evite le conflit de nom avec ce script (speech_recognition.py)."""
    module_name = "speech_recognition"
    module = sys.modules.get(module_name)
    if module and getattr(module, "__file__", None) != __file__:
        return module

    script_dir = Path(__file__).resolve().parent
    script_dir_lower = str(script_dir).lower()

    def _resolved_lower(path: str) -> str:
        try:
            return str(Path(path or ".").resolve()).lower()
        except OSError:
            return (path or "").lower()

    backup = list(sys.path)
    try:
        sys.path = [p for p in sys.path if _resolved_lower(p) != script_dir_lower]
        module = importlib.import_module(module_name)
    finally:
        sys.path[:] = backup

    return module


speech_rec = import_speech_recognition()

DEFAULT_SECONDS = 10
DEFAULT_OUTPUT = "output.wav"
DEFAULT_AMBIENT_DURATION = 0.0
LANGUAGE_CANDIDATES = ("fr-FR", "en-US")


def record_audio(
    duration: float = DEFAULT_SECONDS, ambient_duration: float = DEFAULT_AMBIENT_DURATION
) -> speech_rec.AudioData:
    """Capture du son via le micro par defaut."""
    if duration <= 0:
        raise ValueError("duration doit etre strictement positif.")
    if ambient_duration < 0:
        raise ValueError("ambient_duration ne peut pas etre negatif.")

    recognizer = speech_rec.Recognizer()
    with speech_rec.Microphone() as source:
        if ambient_duration > 0:
            recognizer.adjust_for_ambient_noise(source, duration=ambient_duration)
        print(f"Enregistrement en cours ({duration:.1f}s) ...")
        audio = recognizer.record(source, duration=duration)
    print("Enregistrement termine.")
    return audio


def save_audio(audio: speech_rec.AudioData, filename: str) -> None:
    """Sauvegarde l'audio dans un fichier WAV."""
    path = Path(filename)
    path.write_bytes(audio.get_wav_data())
    print(f"Fichier audio sauvegarde: {path.resolve()}")


def transcribe_audio(
    recognizer: speech_rec.Recognizer,
    audio: speech_rec.AudioData,
    languages: tuple[str, ...] = LANGUAGE_CANDIDATES,
) -> tuple[str, float, str] | None:
    """Transcrit l'audio pour plusieurs langues et affiche la meilleure correspondance."""

    def _best_alt(payload: object) -> tuple[str, float] | None:
        if not isinstance(payload, dict):
            return None
        alternatives = payload.get("alternative")
        if not isinstance(alternatives, list):
            return None
        best_transcript = None
        best_confidence = -1.0
        for candidate in alternatives:
            if not isinstance(candidate, dict):
                continue
            transcript = candidate.get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                continue
            confidence = candidate.get("confidence")
            try:
                conf_val = float(confidence)
            except (TypeError, ValueError):
                conf_val = 0.0
            if conf_val > best_confidence:
                best_confidence = conf_val
                best_transcript = transcript.strip()
        if best_transcript is None:
            return None
        return best_transcript, max(best_confidence, 0.0)

    best_choice: tuple[str, float, str] | None = None

    for language in languages:
        try:
            raw = recognizer.recognize_google(
                audio, language=language, show_all=True
            )
        except speech_rec.UnknownValueError:
            continue
        except speech_rec.RequestError as err:
            print(f"Service de reconnaissance indisponible: {err}")
            return

        best_alt = _best_alt(raw)
        transcript = None
        confidence = 0.0

        if best_alt is None:
            try:
                transcript = recognizer.recognize_google(audio, language=language)
            except speech_rec.UnknownValueError:
                continue
        else:
            transcript, confidence = best_alt

        if not transcript:
            continue

        if best_choice is None or confidence > best_choice[1]:
            best_choice = (transcript, confidence, language)

    if best_choice:
        result_text, conf, lang = best_choice
        confidence_suffix = f" (confiance: {conf:.2f})" if conf > 0 else ""
        print(f"[{lang}] {result_text}{confidence_suffix}")
        return result_text, conf, lang
    else:
        print("La reconnaissance a echoue pour toutes les langues testees.")
        return None


def main(
    duration: float = DEFAULT_SECONDS,
    filename: str = DEFAULT_OUTPUT,
    ambient_duration: float = DEFAULT_AMBIENT_DURATION,
    languages: tuple[str, ...] = LANGUAGE_CANDIDATES,
) -> None:
    recognizer = speech_rec.Recognizer()
    try:
        audio = record_audio(duration, ambient_duration=ambient_duration)
    except OSError as err:
        print(f"Erreur lors de la capture audio: {err}")
        return

    save_audio(audio, filename)
    transcribe_audio(recognizer, audio, languages)


if __name__ == "__main__":
    main()
