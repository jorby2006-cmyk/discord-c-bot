import random
from .base import TestCase
from typing import Optional

def gen_stl_intro(rng: random.Random, kind: Optional[str] = None) -> dict:
    # An advanced level of programming where you will start hitting yourself
    valid_kinds = [
        "vector_sum",
        "sort_vector",
        "word_count"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "vector_sum":
        # Easy: sum elements of a vector
        n = rng.randint(3, 10)
        arr = [rng.randint(1, 20) for _ in range(n)]
        ans = sum(arr)

        sample_in = f"{n}\n{' '.join(map(str, arr))}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 12)
            tarr = [rng.randint(1, 25) for _ in range(tn)]
            tout = sum(tarr)
            tests.append(TestCase(inp=f"{tn}\n{' '.join(map(str, tarr))}\n", out=f"{tout}\n"))

        return {
            "id": "STL_VEC_SUM",
            "title": "Vector Sum",
            "family": "STL_intro",
            "task": "Given n integers in a vector, compute the sum of its elements using STL.",
            "input_format": "First line: n\nSecond line: n integers separated by space",
            "output_format": "Single integer: sum of elements",
            "constraints": "1 ≤ n ≤ 12, 1 ≤ element ≤ 25",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "sort_vector":
        # Medium: count unique elements using set
        n = rng.randint(4, 12)
        arr = [rng.randint(1, 10) for _ in range(n)]
        ans = len(set(arr))

        sample_in = f"{n}\n{' '.join(map(str, arr))}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(4, 15)
            tarr = [rng.randint(1, 12) for _ in range(tn)]
            tout = len(set(tarr))
            tests.append(TestCase(inp=f"{tn}\n{' '.join(map(str, tarr))}\n", out=f"{tout}\n"))

        return {
            "id": "STL_UNIQUE",
            "title": "Unique Elements Count",
            "family": "STL_intro",
            "task": "Given n integers, print the number of unique elements using STL set.",
            "input_format": "First line: n\nSecond line: n integers separated by space",
            "output_format": "Single integer: count of unique numbers",
            "constraints": "1 ≤ n ≤ 15, 1 ≤ element ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "word_count":
        # Trickier: count occurrences of words using map
        words_list = ["apple", "banana", "cat", "dog", "egg"]
        n = rng.randint(3, 8)
        words = [rng.choice(words_list) for _ in range(n)]

        # prepare map output as "word count" per line sorted alphabetically
        from collections import Counter
        c = Counter(words)
        ans = "\n".join(f"{w} {c[w]}" for w in sorted(c.keys())) + "\n"

        sample_in = f"{n}\n{' '.join(words)}\n"
        sample_out = ans

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 10)
            twords = [rng.choice(words_list) for _ in range(tn)]
            tc = Counter(twords)
            tout = "\n".join(f"{w} {tc[w]}" for w in sorted(tc.keys())) + "\n"
            tests.append(TestCase(inp=f"{tn}\n{' '.join(twords)}\n", out=tout))

        return {
            "id": "STL_WORD_COUNT",
            "title": "Word Occurrences",
            "family": "STL_intro",
            "task": "Given n words, print each unique word with its frequency using STL map (alphabetically sorted).",
            "input_format": "First line: n\nSecond line: n words separated by space",
            "output_format": "Each line: word count, sorted alphabetically by word",
            "constraints": "1 ≤ n ≤ 10, words from a fixed small list",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
