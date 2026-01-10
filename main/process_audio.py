import csv
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT, AUDIO_ROOT, TRANSCRIPTIONS_FILE


def create_transcriptions_txt(
    data_dir: Path,
    clips_dir: Path,
    output_name: str = "transcriptions.txt",
    tsv_name: str = "validated.tsv",
    delete_unvalidated: bool = True,
):
    """
    Converts Common Voice validated.tsv into transcriptions.txt
    and optionally deletes clips not present in the TSV file
    
    Args:
        data_dir: Directory containing the TSV files
        clips_dir: Directory containing audio clips
        output_name: Name of output transcriptions file
        tsv_name: Name of TSV file to process (default: validated.tsv)
        delete_unvalidated: Whether to delete clips not in TSV (default: True)
    
    Format per line:
        filename sentence
    """
    
    tsv_path = data_dir / tsv_name
    output_path = data_dir / output_name
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"{tsv_path} not found")
    
    if not clips_dir.exists():
        raise FileNotFoundError(f"{clips_dir} not found")
    
    # Read validated clips
    validated_clips = set()
    written = 0
    
    with open(tsv_path, "r", encoding="utf-8") as tsv, \
         open(output_path, "w", encoding="utf-8") as out:
        
        reader = csv.DictReader(tsv, delimiter="\t")
        
        for row in reader:
            filename = row.get("path")
            sentence = row.get("sentence", "").strip()
            
            if not filename or not sentence:
                continue
            
            audio_path = clips_dir / filename
            if not audio_path.exists():
                continue
            
            validated_clips.add(filename)
            out.write(f"{filename} {sentence}\n")
            written += 1
    
    print(f"✔ Wrote {written} entries to {output_path}")
    
    # Delete clips not in validated TSV
    if delete_unvalidated:
        deleted = 0
        kept = 0
        
        for clip_file in clips_dir.iterdir():
            if clip_file.is_file():
                if clip_file.name not in validated_clips:
                    try:
                        clip_file.unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"⚠ Failed to delete {clip_file.name}: {e}")
                else:
                    kept += 1
        
        print(f"✔ Kept {kept} validated clips")
        print(f"✔ Deleted {deleted} non-validated clips")
    else:
        print(f"ℹ Skipped deletion (delete_unvalidated=False)")


if __name__ == "__main__":
    create_transcriptions_txt(
        data_dir=DATA_ROOT / "hi",
        clips_dir=AUDIO_ROOT,
        output_name=TRANSCRIPTIONS_FILE,
        tsv_name="validated.tsv",
        delete_unvalidated=True,
    )
