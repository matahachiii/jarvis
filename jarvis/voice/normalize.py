"""
Speech normalization: turn written text into words a TTS engine reads correctly.

    "Set to 40%."            -> "Set to forty percent."
    "It ended 4-2."          -> "It ended four two."
    "It's 12°C at 14:30."    -> "It's twelve degrees Celsius at fourteen thirty."
    "€1,250 on 2026-11-27"   -> "one thousand two hundred and fifty euros on the twenty-seventh of November twenty twenty-six"

Runs on ONE sentence at a time (the Speaker splits first), right before synthesis.
Our own small function on purpose: the same result for every TTS engine, and testable.
British style, to match the voice: "one hundred and five", years as "twenty twenty-six".

Order matters: the specific patterns (dates, money, times, temperatures, %, ordinals, units)
run first; plain numbers last. Each step turns digits into words, so later steps never
see the same digits twice.
"""

import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                     "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

CURRENCIES = {"€": ("euro", "euros", "cents"), "$": ("dollar", "dollars", "cents"),
              "£": ("pound", "pounds", "pence")}
UNITS = {"km/h": ("kilometre per hour", "kilometres per hour"),
         "mph": ("mile per hour", "miles per hour"),
         "km": ("kilometre", "kilometres"),
         "kg": ("kilogram", "kilograms"),
         "GB": ("gigabyte", "gigabytes"),
         "MB": ("megabyte", "megabytes")}
OPERATORS = {"+": "plus", "=": "equals", "×": "times", "÷": "divided by", "x": "times"}

# Whole-word abbreviations. "etc." at the very end keeps its full stop (end of the sentence).
ABBREVIATIONS = [
    (re.compile(r"\be\.g\.,?"), "for example,"),
    (re.compile(r"\bi\.e\.,?"), "that is,"),
    (re.compile(r"\betc\.(?=\s*$)"), "et cetera."),
    (re.compile(r"\betc\."), "et cetera"),
    (re.compile(r"\bvs\.?(?=\s)"), "versus"),
    (re.compile(r"\bapprox\."), "approximately"),
    (re.compile(r"\bMr\.(?=\s)"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s)"), "Missus"),
    (re.compile(r"\bDr\.(?=\s)"), "Doctor"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"~(?=\d)"), "about "),
]

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"   # 1,234.5 / 1234 / 3.5 (no sign)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONEY_BEFORE = re.compile(rf"([€$£])\s?(-?)({NUM})(\s?(?:thousand|million|billion|k)\b)?")
_MONEY_AFTER = re.compile(rf"(?<![\w.])(-?)({NUM})\s?€")
_TIME_AMPM = re.compile(r"(?<![\w:])(\d{1,2})(?::([0-5]\d))?\s?([AaPp])\.?[Mm](\.(?=\s+[a-z]))?(?![\w])")
_TIME_24H = re.compile(r"(?<![\w:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_TEMPERATURE = re.compile(rf"(?<![\w.])(-?)({NUM})\s?°\s?([CF])(?![A-Za-z])")
_DEGREES = re.compile(rf"(?<![\w.])(-?)({NUM})\s?°")
_PERCENT = re.compile(rf"(?<![\w.])(-?)({NUM})\s?%")
_ORDINAL = re.compile(r"(?<![\w.])(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_UNIT = re.compile(rf"(?<![\w.])({NUM})\s?(km/h|mph|km|kg|GB|MB)(?![\w/])")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")   # "mp3" -> "mp 3"
_OPERATOR = re.compile(r"(?<=\d)\s*([+=×÷])\s*(?=-?\d)|(?<=\d)\s+(x)\s+(?=\d)")
_VERSION = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}")       # 2.5.1 -> two point five point one
_SCORE = re.compile(r"(?<![\w.,-])(\d+)[-–](\d+)(?![\w-])")
_NUMBER = re.compile(rf"(?<![\w.])(-?)({NUM})")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"[ \t]{2,}")


def number_to_words(n: int) -> str:
    """Whole number -> British English words: 105 -> 'one hundred and five', -4 -> 'minus four'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + (" and " + number_to_words(rest) if rest else "")
    parts = []
    for size, word in SCALES:
        if n >= size:
            group = (n // size) % 1000
            if group:
                parts.append(f"{number_to_words(group)} {word}")
    last = n % 1000
    if last:
        # British "and" before a last group below 100: 1005 -> "one thousand and five"
        parts.append(("and " if last < 100 else "") + number_to_words(last))
    return " ".join(parts)


def year_to_words(year: int) -> str:
    """1900-2099 as spoken: 2026 -> 'twenty twenty-six', 2005 -> 'two thousand and five', 1905 -> 'nineteen oh five'."""
    if 2000 <= year <= 2009:
        return number_to_words(year)
    high, low = divmod(year, 100)
    if low == 0:
        return number_to_words(high) + " hundred"
    if low < 10:
        return f"{number_to_words(high)} oh {ONES[low]}"
    return f"{number_to_words(high)} {number_to_words(low)}"


def ordinal_words(n: int) -> str:
    """22 -> 'twenty-second', 11 -> 'eleventh', 40 -> 'fortieth'."""
    words = number_to_words(n)
    cut = max(words.rfind(" "), words.rfind("-")) + 1
    head, last = words[:cut], words[cut:]
    if last in ORDINAL_IRREGULAR:
        last = ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + last


def digits_to_words(digits: str) -> str:
    """Read digit by digit: '007' -> 'zero zero seven'."""
    return " ".join(ONES[int(d)] for d in digits)


def spoken_number(sign: str, number: str, years: bool = True) -> str:
    """
    A written number (without sign) -> words. Handles 1,234 / 3.14 / 007 / years.
    years=True: a plain 4-digit number from 1900 to 2099 is read as a year.
    """
    whole, _, decimals = number.partition(".")
    if "," in whole:
        words = number_to_words(int(whole.replace(",", "")))
    elif len(whole) > 1 and whole.startswith("0"):
        words = digits_to_words(whole)                      # codes like 007, 0586
    elif len(whole) > 15:
        words = digits_to_words(whole)                      # too long to say as a number
    elif years and not sign and not decimals and len(whole) == 4 and 1900 <= int(whole) <= 2099:
        words = year_to_words(int(whole))
    else:
        words = number_to_words(int(whole))
    if decimals:
        words += " point " + digits_to_words(decimals)
    return ("minus " if sign else "") + words


def _is_one(sign: str, number: str) -> bool:
    return not sign and number == "1"


def _iso_date(m: re.Match) -> str:
    year, month, day = int(m[1]), int(m[2]), int(m[3])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return m[0]                                         # not a real date: leave it to later steps
    year_words = year_to_words(year) if 1900 <= year <= 2099 else number_to_words(year)
    return f"the {ordinal_words(day)} of {MONTHS[month - 1]} {year_words}"


def _money(symbol: str, sign: str, number: str, scale: str | None) -> str:
    singular, plural, cents_word = CURRENCIES[symbol]
    if scale:                                               # €1.5 million -> one point five million euros
        scale = scale.strip()
        scale = "thousand" if scale == "k" else scale
        return f"{spoken_number(sign, number, years=False)} {scale} {plural}"
    whole, _, cents = number.partition(".")
    if len(cents) == 2:                                     # €2.50 -> two euros fifty
        whole_value = int(whole.replace(",", ""))
        cents_words = number_to_words(int(cents))
        minus = "minus " if sign else ""
        if whole_value == 0:
            return f"{minus}{cents_words} {cents_word}"     # $0.99 -> ninety-nine cents
        name = singular if whole_value == 1 else plural
        words = f"{minus}{number_to_words(whole_value)} {name}"
        return words + (f" {cents_words}" if int(cents) else "")
    name = singular if _is_one(sign, number) else plural
    return f"{spoken_number(sign, number, years=False)} {name}"


def _time(hour: int, minute: int) -> str:
    """9:00 -> 'nine o'clock', 9:05 -> 'nine oh five', 14:30 -> 'fourteen thirty'."""
    if minute == 0:
        return f"{number_to_words(hour)} o'clock"
    if minute < 10:
        return f"{number_to_words(hour)} oh {ONES[minute]}"
    return f"{number_to_words(hour)} {number_to_words(minute)}"


def _time_ampm(m: re.Match) -> str:
    hour, minute = int(m[1]), int(m[2] or 0)
    if not 1 <= hour <= 12:
        return m[0]
    suffix = "ay em" if m[3].lower() == "a" else "pee em"   # spelled out: TTS engines read "A" as "uh"
    clock = number_to_words(hour) if minute == 0 else _time(hour, minute)
    return f"{clock} {suffix}"


def _temperature(m: re.Match) -> str:
    unit = "Celsius" if m[3] == "C" else "Fahrenheit"
    degrees = "degree" if _is_one(m[1], m[2]) else "degrees"
    return f"{spoken_number(m[1], m[2], years=False)} {degrees} {unit}"


def _unit(m: re.Match) -> str:
    singular, plural = UNITS[m[2]]
    return f"{spoken_number('', m[1], years=False)} {singular if m[1] == '1' else plural}"


def normalize_for_speech(text: str) -> str:
    """Return the sentence with numbers, symbols and simple abbreviations written out as words."""
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = _ISO_DATE.sub(_iso_date, text)
    text = _MONEY_BEFORE.sub(lambda m: _money(m[1], m[2], m[3], m[4]), text)
    text = _MONEY_AFTER.sub(lambda m: _money("€", m[1], m[2], None), text)
    text = _TIME_AMPM.sub(_time_ampm, text)
    text = _TIME_24H.sub(lambda m: _time(int(m[1]), int(m[2])), text)
    text = _TEMPERATURE.sub(_temperature, text)
    text = _DEGREES.sub(lambda m: f"{spoken_number(m[1], m[2], years=False)} "
                                  f"{'degree' if _is_one(m[1], m[2]) else 'degrees'}", text)
    text = _PERCENT.sub(lambda m: f"{spoken_number(m[1], m[2], years=False)} percent", text)
    text = _ORDINAL.sub(lambda m: ordinal_words(int(m[1])), text)
    text = _UNIT.sub(_unit, text)
    text = _LETTER_DIGIT.sub(" ", text)
    text = _OPERATOR.sub(lambda m: f" {OPERATORS[m[1] or m[2]]} ", text)
    text = _VERSION.sub(lambda m: " point ".join(number_to_words(int(p)) for p in m[0].split(".")), text)
    text = _SCORE.sub(lambda m: f"{spoken_number('', m[1])} {spoken_number('', m[2])}", text)
    text = _NUMBER.sub(lambda m: spoken_number(m[1], m[2]), text)
    text = _DIGITS.sub(lambda m: digits_to_words(m[0]), text)   # safety net: no digit is ever left
    return _SPACES.sub(" ", text).strip()
