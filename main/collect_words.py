from pathlib import Path
from typing import Dict, Tuple
from collections import Counter
import re
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_ROOT,
    LANGUAGE,
    TRANSCRIPTIONS_FILE,
    WORDS_FILE,
    LEXICON_FILE,
    UROMAN_DIR,
    UROMAN_LANG_CODE,
)


def text_normalize(text: str) -> str:
    """Normalize text for Bulgarian (Cyrillic script)"""
    text = re.sub(r'^[^ ]+\.mp3\s+', '', text.strip())
    # Bulgarian Cyrillic range: \u0400-\u04FF (Cyrillic block)
    text = re.sub(r'[^\u0400-\u04FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def norm_uroman(text: str) -> str:
    text = text.lower().replace("'", "'")
    text = re.sub(r"([^a-z' ])", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def uromanize_python(words: list) -> Dict[str, str]:
    """Python uroman - split into character tokens for torchaudio."""
    if str(UROMAN_DIR) not in sys.path:
        sys.path.insert(0, str(UROMAN_DIR))
    
    import uroman as ur
    u = ur.Uroman()
    
    lexicon = {}
    for dev_word in words:
        rom_text = u.romanize_string(dev_word, lcode=UROMAN_LANG_CODE)
        uroman_clean = re.sub(r"\s+", "", norm_uroman(rom_text)).strip()
        
        # Split into character tokens with spaces
        char_tokens = " ".join(list(uroman_clean))
        lexicon[dev_word] = char_tokens + " |"
    
    return lexicon


def filter_lexicon(lexicon: Dict[str, str], word_counts: Dict[str, int]) -> Dict[str, str]:
    spelling_to_words = {}
    for dev_word, uroman_spell in lexicon.items():
        spelling_to_words.setdefault(uroman_spell, []).append(dev_word)
    
    filtered_lexicon = {}
    for uroman_spell, dev_words in spelling_to_words.items():
        if len(dev_words) > 1:
            dev_words.sort(key=lambda w: (-word_counts[w], len(w)))
        filtered_lexicon[dev_words[0]] = uroman_spell
    return filtered_lexicon


def create_word_list_and_lexicon(transcriptions_path: Path, words_path: Path, 
                                lexicon_path: Path, min_count: int = 2) -> Tuple[Dict[str, int], Dict[str, str]]:
    word_counts: Dict[str, int] = Counter()
    sentences = 0
    
    with open(transcriptions_path, 'r', encoding='utf-8') as f:
        for line in f:
            norm_line = text_normalize(line.strip())
            if not norm_line:
                continue
            words = norm_line.split()
            for word in words:
                if len(word) >= 2:
                    word_counts[word] += 1
            sentences += 1
    
    # Updated filter for Bulgarian Cyrillic characters
    filtered_counts = {w: c for w, c in word_counts.items() 
                      if c >= min_count and re.match(r'[\u0400-\u04FF]', w)}
    
    sorted_words = sorted(filtered_counts.items(), key=lambda x: (-x[1], x[0]))
    
    with open(words_path, 'w', encoding='utf-8') as f:
        for word, count in sorted_words:
            f.write(f"{word} {count}\n")
    
    print("Python uromanizing words...")
    lexicon = uromanize_python(list(filtered_counts.keys()))
    filtered_lexicon = filter_lexicon(lexicon, dict(filtered_counts))
    
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        for dev_word, uroman_spell in filtered_lexicon.items():
            f.write(f"{dev_word} {uroman_spell}\n")
    
    print(f"✅ words.txt: {len(sorted_words)} words")
    print(f"✅ lexicon.txt: {len(filtered_lexicon)} entries")
    print("Sample lexicon:", list(filtered_lexicon.items())[:5])
    
    return dict(sorted_words), filtered_lexicon


def main():
    data_dir = DATA_ROOT / LANGUAGE
    transcriptions_path = data_dir / TRANSCRIPTIONS_FILE
    words_output = data_dir / WORDS_FILE
    lexicon_output = data_dir / LEXICON_FILE
    
    word_counts, lexicon = create_word_list_and_lexicon(
        transcriptions_path, words_output, lexicon_output
    )
    print("\n✅ Ready for torchaudio ctc_decoder!")


if __name__ == "__main__":
    main()
