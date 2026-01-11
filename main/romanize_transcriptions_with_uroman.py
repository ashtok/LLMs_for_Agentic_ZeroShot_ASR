from __future__ import annotations

from pathlib import Path
import sys

# Import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    TRANSCRIPTIONS_UROMAN_FILE,
    REPO_ROOT,
)


def romanize_transcriptions_file(
    input_path: Path,
    output_path: Path,
    lang_code: str = "hin",
) -> int:
    """
    Romanize a UTF-8 transcription file line-by-line using uroman (Python).
    Expects input format: "filename text"
    Outputs format: "filename romanized_text"
    
    Args:
        input_path: Path to input transcriptions file
        output_path: Path to output romanized transcriptions file
        lang_code: Language code for uroman (default: hin for Hindi)
    
    Returns:
        Number of lines processed
    """
    # Point sys.path to the directory that contains uroman.py
    uroman_dir = REPO_ROOT / "uroman" / "uroman"
    
    if not uroman_dir.exists():
        raise FileNotFoundError(f"Uroman directory not found: {uroman_dir}")
    
    # Insert at beginning to ensure this uroman is used
    if str(uroman_dir) not in sys.path:
        sys.path.insert(0, str(uroman_dir))

    import uroman as ur  # now resolves to uroman/uroman/uroman.py

    u = ur.Uroman()  # loads data dir automatically

    lines_processed = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                fout.write("\n")
                continue

            # Split filename and text
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            filename, text = parts
            
            # Only romanize the text, not the filename
            rom_text = u.romanize_string(text, lcode=lang_code)
            
            # Write filename followed by romanized text
            fout.write(f"{filename} {rom_text.strip()}\n")
            lines_processed += 1
    
    return lines_processed


def main() -> None:
    # Use config paths
    data_dir = DATA_ROOT / LANGUAGE
    in_path = data_dir / TRANSCRIPTIONS_FILE
    out_path = data_dir / TRANSCRIPTIONS_UROMAN_FILE
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not in_path.exists():
        raise FileNotFoundError(f"Input transcriptions not found: {in_path}")
    
    print(f"Data directory: {data_dir}")
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"\nRomanizing transcriptions...")
    
    lines_processed = romanize_transcriptions_file(
        input_path=in_path,
        output_path=out_path,
        lang_code="hin",
    )
    
    print(f"✔ Done! Processed {lines_processed} lines.")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
