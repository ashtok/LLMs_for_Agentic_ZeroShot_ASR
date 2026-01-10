import csv
from pathlib import Path


def create_transcriptions_txt(
    base_dir: Path,
    split: str = "train",
    output_name: str = "transcriptions.txt",
):
    """
    Converts Common Voice TSV (train/dev/test) into transcriptions.txt

    Format per line:
        filename sentence
    """

    tsv_path = base_dir / f"{split}.tsv"
    clips_dir = base_dir / "clips"
    output_path = base_dir / output_name

    if not tsv_path.exists():
        raise FileNotFoundError(f"{tsv_path} not found")

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

            out.write(f"{filename} {sentence}\n")
            written += 1

    print(f"✔ Wrote {written} entries to {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(
        r"D:\Masters In Germany\Computer Science\Semester 5\Thesis\NLP_Thesis\data\hi"
    )

    # Change split to: train / dev / test / validated
    create_transcriptions_txt(BASE_DIR, split="train")
