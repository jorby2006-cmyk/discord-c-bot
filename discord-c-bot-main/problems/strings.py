import random
from .base import TestCase
from typing import Optional

def gen_strings(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Basic string exercises
    valid_kinds = [
        "palindrome",
        "vowel_count",
        "reverse_string"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "palindrome":
        s = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 10)))
        ans = "YES" if s == s[::-1] else "NO"

        sample_in = f"{s}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            ts = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(1, 12)))
            tout = "YES\n" if ts == ts[::-1] else "NO\n"
            tests.append(TestCase(inp=f"{ts}\n", out=tout))

        return {
            "id": "STR_PALIN",
            "title": "Palindrome Check",
            "family": "strings",
            "task": "Given a string S, print YES if it's a palindrome, NO otherwise.",
            "input_format": "One line: S",
            "output_format": "YES or NO",
            "constraints": "1 ≤ length of S ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "vowel_count":
        s = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 15)))
        ans = sum(1 for ch in s if ch in "aeiou")

        sample_in = f"{s}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            ts = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(1, 15)))
            tout = f"{sum(1 for ch in ts if ch in 'aeiou')}\n"
            tests.append(TestCase(inp=f"{ts}\n", out=tout))

        return {
            "id": "STR_VOWEL_CT",
            "title": "Count Vowels",
            "family": "strings",
            "task": "Given a string S, print the number of vowels (a, e, i, o, u).",
            "input_format": "One line: S",
            "output_format": "Integer: number of vowels",
            "constraints": "1 ≤ length of S ≤ 15",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "reverse_string":
        s = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 15)))
        ans = s[::-1]

        sample_in = f"{s}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            ts = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(1, 15)))
            tout = f"{ts[::-1]}\n"
            tests.append(TestCase(inp=f"{ts}\n", out=tout))

        return {
            "id": "STR_REV",
            "title": "Reverse String",
            "family": "strings",
            "task": "Given a string S, print the string reversed.",
            "input_format": "One line: S",
            "output_format": "Reversed string",
            "constraints": "1 ≤ length of S ≤ 15",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
