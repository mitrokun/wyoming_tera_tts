"""
Определяет границы предложений в потоке текста.
Поддерживает расстановку пауз для абзацев и диалогов.
"""
import regex as re
from collections.abc import Iterable

HARD_LIMIT = 350

ABBR_SET = {
    "г", "ул", "обл", "пер", "пр", "просп", "наб", "бул", "стр", "корп", "кв", "пос", "сел", "д",
    "акад", "проф", "доц", "канд", "тов", "гр", "ген", "лейт", "кап", "зам", "зав", "дир", "ред", "св", "дж",
    "см", "ср", "напр", "вып", "табл", "рис", "ил", "цит", "гл", "ст", "лат", "изд", "техн", "им", "руб", "коп", "мес",
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "inc"
}

ABBR_FOR_INTONATION = "|".join(sorted(ABBR_SET, key=len, reverse=True))

SENTENCE_BOUNDARY_RE = re.compile(
    rf"""
    (?<!\b(?i:{ABBR_FOR_INTONATION})) 
    (?<!\b\p{{Lu}})                    
    (?<!\p{{Ll}}\.\p{{Ll}})            
    (?<!(?:^|\n)[ \t]*\d+)             
    ([.!?])                            
    (?=
        \s+                        
        (?:[-—–]\s*)?                  
        [("«\[]*                       
        [\p{{Lu}}\d]                   
    )
    """,
    re.VERBOSE | re.UNICODE
)

BREAK_RE = re.compile(r'(\n\s*\n)|(\n[ \t]*(?=[-—–]))')

DASH_FALLBACK_RE = re.compile(r'.*[-—–][ \t]+', re.DOTALL)
COLON_FALLBACK_RE = re.compile(r'.*[;:][ \t]+', re.DOTALL)
COMMA_FALLBACK_RE = re.compile(r'.*[,][ \t]+', re.DOTALL)
SPACE_FALLBACK_RE = re.compile(r'.*[ \t]+', re.DOTALL)

PARENS_RE = re.compile(r"\s*\((.*?)\)")
LIST_ITEM_RE = re.compile(r"^\s*(?:(\d+)\.|([*-]))\s*(.*)", re.MULTILINE)
LEADING_PUNCT_RE = re.compile(r"^[.,\s]+")
MULTI_SPACE_RE = re.compile(r"\s+")
DOUBLE_PUNCT_RE = re.compile(r"\s*([,.]\s*){2,}")

REMOVE_CHARS = str.maketrans("", "", "*«»\"„“")


def _list_replacer(match) -> str:
    num, bullet, text = match.groups()
    return f"{num} — {text}" if num else text


def post_clean_sentence(sentence: str) -> str:
    sentence = LIST_ITEM_RE.sub(_list_replacer, sentence)
    sentence = sentence.replace('…', ' —').replace('\n', ' ').replace(';', ' —')
    sentence = sentence.translate(REMOVE_CHARS)
    sentence = PARENS_RE.sub(r", \1, ", sentence)
    sentence = LEADING_PUNCT_RE.sub("", sentence)
    sentence = MULTI_SPACE_RE.sub(" ", sentence)
    sentence = DOUBLE_PUNCT_RE.sub(r"\1 ", sentence).strip()
    return sentence


class SentenceBoundaryDetector:
    def __init__(self, emit_break_markers: bool = True) -> None:
        self.buffer = ""
        self.emit_break_markers = emit_break_markers

    def add_chunk(self, chunk: str) -> Iterable[str]:
        self.buffer += chunk

        while True:
            match_break = BREAK_RE.search(self.buffer)
            match_punc = SENTENCE_BOUNDARY_RE.search(self.buffer)
            
            if match_break and (not match_punc or match_break.start() <= match_punc.start()):
                sentence = self.buffer[:match_break.start()].strip()
                if sentence:
                    yield post_clean_sentence(sentence)
                
                if self.emit_break_markers:
                    if match_break.group(1):
                        yield "<PARAGRAPH_BREAK>"
                    else:
                        yield "<DIALOGUE_BREAK>"
                    
                self.buffer = self.buffer[match_break.end():]
                continue

            if not match_punc:
                if len(self.buffer) > HARD_LIMIT:
                    sub_buffer = self.buffer[:HARD_LIMIT]
                    
                    match = DASH_FALLBACK_RE.search(sub_buffer)
                    if not match:
                        match = COLON_FALLBACK_RE.search(sub_buffer)
                    if not match:
                        match = COMMA_FALLBACK_RE.search(sub_buffer)
                    if not match:
                        match = SPACE_FALLBACK_RE.search(sub_buffer)
                    
                    split_pos = match.end() if match else HARD_LIMIT
                    if split_pos <= 0:
                        split_pos = HARD_LIMIT

                    sentence = self.buffer[:split_pos].strip()
                    if sentence:
                        yield post_clean_sentence(sentence)
                    
                    self.buffer = self.buffer[split_pos:].lstrip()
                    continue

                break

            sep_char = match_punc.group(1)
            sep_end_pos = match_punc.end(1)
            sep_start_pos = match_punc.start(1)

            if (sep_char == '.' and 
                sep_end_pos == len(self.buffer) and 
                sep_start_pos > 0 and 
                self.buffer[sep_start_pos - 1].isdigit()):
                break

            sentence_end_pos = match_punc.end(1)
            sentence = self.buffer[:sentence_end_pos].strip()
            
            if sentence:
                yield post_clean_sentence(sentence)
            
            self.buffer = self.buffer[sentence_end_pos:]

    def finish(self) -> str:
        res = post_clean_sentence(self.buffer)
        self.buffer = ""
        return res