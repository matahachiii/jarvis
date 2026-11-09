"""Safety net: strip emojis and markdown from model output before it is shown or spoken."""

import re

# Emojis and pictographic symbols (the most common Unicode blocks)
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, pictographs, flags, ...
    "\u2600-\u27BF"          # misc symbols and dingbats (sun, check marks, ...)
    "\uFE0F\u200D"           # invisible emoji helpers (variation selector, joiner)
    "]+"
)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")                                # [text](url) -> text
_HEADING = re.compile(r"^[ \t]*#+[ \t]*", re.MULTILINE)                     # "# Title" -> "Title"
_LIST_PREFIX = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", re.MULTILINE)  # "- item", "1. item" -> "item"
_MARKDOWN_CHARS = re.compile(r"[*#`]")                                      # leftover bold/italic/code marks
_EXTRA_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")


def clean_for_speech(text: str) -> str:
    """Return text without emojis or markdown.

    Order matters: first structures (links, headings, list bullets), then loose characters.
    Leading/trailing spaces are NOT stripped, because later this also runs on streamed chunks
    where a space at the edge is meaningful ("thirty" + " six").
    """
    text = _LINK.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_PREFIX.sub("", text)
    text = _MARKDOWN_CHARS.sub("", text)
    text = _EXTRA_SPACES.sub(" ", text)
    text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
    return text


class StreamCleaner:
    """
    Cleans streamed text chunk by chunk.

    Holds back the last unfinished word until a space or newline arrives, so markdown
    split over two chunks ("**Cert" + "ainly**") is still caught. Remembers whether the
    previous piece ended a line, so "- " is only removed when it really is a list bullet.
    """

    def __init__(self):
        self._buffer = ""
        self._at_line_start = True

    def feed(self, chunk: str) -> str:
        """Add a chunk; return the part that is ready to show (complete words only)."""
        self._buffer += chunk
        cut = max(self._buffer.rfind(" "), self._buffer.rfind("\n"))
        if cut < 0:
            return ""  # no complete word yet, keep waiting
        ready, self._buffer = self._buffer[:cut + 1], self._buffer[cut + 1:]
        return self._clean(ready)

    def flush(self) -> str:
        """Return whatever is left at the end of the stream."""
        ready, self._buffer = self._buffer, ""
        return self._clean(ready)

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        # A fake previous character tells the regexes whether we are at the start of a line:
        # "\n" -> yes (bullets/headings can be stripped), "x" -> no (mid-sentence, leave "-" alone)
        prefix = "\n" if self._at_line_start else "x"
        cleaned = clean_for_speech(prefix + text)[1:]
        self._at_line_start = text.rstrip(" \t").endswith("\n")
        return cleaned
