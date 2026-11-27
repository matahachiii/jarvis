"""
Unit tests for jarvis/voice/normalize.py (speech normalization). Pure Python, no TTS, no sound.

    python -m unittest test_voice_normalize
"""

import unittest

from jarvis.voice.normalize import (normalize_for_speech, number_to_words, ordinal_words,
                                    spoken_number, year_to_words)


class TestNumberToWords(unittest.TestCase):
    def test_small(self):
        self.assertEqual(number_to_words(0), "zero")
        self.assertEqual(number_to_words(7), "seven")
        self.assertEqual(number_to_words(13), "thirteen")
        self.assertEqual(number_to_words(40), "forty")
        self.assertEqual(number_to_words(65), "sixty-five")

    def test_hundreds_british_and(self):
        self.assertEqual(number_to_words(100), "one hundred")
        self.assertEqual(number_to_words(105), "one hundred and five")
        self.assertEqual(number_to_words(999), "nine hundred and ninety-nine")

    def test_thousands_and_more(self):
        self.assertEqual(number_to_words(1005), "one thousand and five")
        self.assertEqual(number_to_words(1234), "one thousand two hundred and thirty-four")
        self.assertEqual(number_to_words(20000), "twenty thousand")
        self.assertEqual(number_to_words(2_000_000), "two million")
        self.assertEqual(number_to_words(3_000_000_050), "three billion and fifty")

    def test_negative(self):
        self.assertEqual(number_to_words(-4), "minus four")


class TestYearsOrdinals(unittest.TestCase):
    def test_years(self):
        self.assertEqual(year_to_words(2026), "twenty twenty-six")
        self.assertEqual(year_to_words(2010), "twenty ten")
        self.assertEqual(year_to_words(2000), "two thousand")
        self.assertEqual(year_to_words(2005), "two thousand and five")
        self.assertEqual(year_to_words(1905), "nineteen oh five")
        self.assertEqual(year_to_words(1900), "nineteen hundred")

    def test_ordinals(self):
        self.assertEqual(ordinal_words(1), "first")
        self.assertEqual(ordinal_words(2), "second")
        self.assertEqual(ordinal_words(11), "eleventh")
        self.assertEqual(ordinal_words(12), "twelfth")
        self.assertEqual(ordinal_words(22), "twenty-second")
        self.assertEqual(ordinal_words(40), "fortieth")
        self.assertEqual(ordinal_words(100), "one hundredth")

    def test_spoken_number(self):
        self.assertEqual(spoken_number("", "1,234"), "one thousand two hundred and thirty-four")
        self.assertEqual(spoken_number("", "3.14"), "three point one four")
        self.assertEqual(spoken_number("-", "2.5"), "minus two point five")
        self.assertEqual(spoken_number("", "007"), "zero zero seven")
        self.assertEqual(spoken_number("", "2026"), "twenty twenty-six")
        self.assertEqual(spoken_number("", "2026", years=False), "two thousand and twenty-six")
        self.assertEqual(spoken_number("", "1500"), "one thousand five hundred")   # not a year


class TestNormalize(unittest.TestCase):
    def check(self, text, expected):
        self.assertEqual(normalize_for_speech(text), expected)

    def test_benchmark_sentence(self):
        self.check("Certainly, sir. Spotify is paused, and the volume is now at 40 percent.",
                   "Certainly, sir. Spotify is paused, and the volume is now at forty percent.")

    def test_text_without_numbers_is_unchanged(self):
        self.check("Good evening, sir. How may I help?", "Good evening, sir. How may I help?")

    def test_percent(self):
        self.check("Set it to 40%.", "Set it to forty percent.")
        self.check("18% of 50 is 9.", "eighteen percent of fifty is nine.")
        self.check("-5% growth", "minus five percent growth")

    def test_score(self):
        self.check("It ended 4-2.", "It ended four two.")
        self.check("It ended 4–2.", "It ended four two.")   # en dash

    def test_temperature(self):
        self.check("It's 12°C.", "It's twelve degrees Celsius.")
        self.check("It's 12 °C.", "It's twelve degrees Celsius.")
        self.check("It's -3°C.", "It's minus three degrees Celsius.")
        self.check("It's 1°C.", "It's one degree Celsius.")
        self.check("It's 50°F.", "It's fifty degrees Fahrenheit.")
        self.check("Turn it 90°.", "Turn it ninety degrees.")
        self.check("It's 12 degrees.", "It's twelve degrees.")

    def test_times(self):
        self.check("It's 14:30.", "It's fourteen thirty.")
        self.check("At 9:05.", "At nine oh five.")
        self.check("At 9:00.", "At nine o'clock.")
        self.check("At 3 pm.", "At three pee em.")
        self.check("At 9:30 a.m. tomorrow.", "At nine thirty ay em tomorrow.")
        self.check("Wake me at 7 a.m.", "Wake me at seven ay em.")   # final full stop kept

    def test_dates_and_years(self):
        self.check("Today is 2026-11-27.", "Today is the twenty-seventh of November twenty twenty-six.")
        self.check("The 1st and 22nd of May.", "The first and twenty-second of May.")
        self.check("In 2026.", "In twenty twenty-six.")

    def test_money(self):
        self.check("It costs €1,250.", "It costs one thousand two hundred and fifty euros.")
        self.check("It costs $2.50.", "It costs two dollars fifty.")
        self.check("It costs $0.99.", "It costs ninety-nine cents.")
        self.check("It costs £1.", "It costs one pound.")
        self.check("It costs €1.5 million.", "It costs one point five million euros.")
        self.check("It costs 5€.", "It costs five euros.")

    def test_units(self):
        self.check("Wind at 15 km/h.", "Wind at fifteen kilometres per hour.")
        self.check("Just 1 km away.", "Just one kilometre away.")
        self.check("Over 100 GB.", "Over one hundred gigabytes.")

    def test_math(self):
        self.check("5 + 3 = 8", "five plus three equals eight")
        self.check("3 x 4 = 12", "three times four equals twelve")

    def test_numbers(self):
        self.check("1,234,567 people", "one million two hundred and thirty-four thousand five hundred and sixty-seven people")
        self.check("Pi is 3.14.", "Pi is three point one four.")
        self.check("Version 2.5.1.", "Version two point five point one.")
        self.check("COVID-19", "COVID-nineteen")
        self.check("an mp3 file", "an mp three file")

    def test_abbreviations(self):
        self.check("Mr. Smith & Dr. Who", "Mister Smith and Doctor Who")
        self.check("Cats vs. dogs.", "Cats versus dogs.")
        self.check("Apples, pears, etc.", "Apples, pears, et cetera.")
        self.check("Fruit, e.g. apples.", "Fruit, for example, apples.")
        self.check("~20 minutes", "about twenty minutes")

    def test_no_digit_ever_left(self):
        for text in ["Room 4B", "x1y2z3", "a.1.b", "27-11-2026", "16:9 screen", "Call 0586 883"]:
            self.assertNotRegex(normalize_for_speech(text), r"\d", text)


if __name__ == "__main__":
    unittest.main()
