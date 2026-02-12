import random
from .base import TestCase
from typing import Optional

def gen_recursion(rng: random.Random, kind: Optional[str] = None) -> dict:
    # call itself daw
    valid_kinds = [
        "factorial",
        "sum_n",
        "reverse_string"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "factorial":
        # Easy: factorial of n
        n = rng.randint(0, 10)
        ans = 1
        for i in range(1, n+1):
            ans *= i

        sample_in = f"{n}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(0, 12)
            tout = 1
            for i in range(1, tn+1):
                tout *= i
            tests.append(TestCase(inp=f"{tn}\n", out=f"{tout}\n"))

        return {
            "id": "REC_FACTORIAL",
            "title": "Factorial of N",
            "family": "recursion",
            "task": "Given n, print n! (factorial) using recursion.",
            "input_format": "One line: n",
            "output_format": "Single integer: factorial of n",
            "constraints": "0 ≤ n ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "sum_n":
        # Medium: sum of first n natural numbers
        n = rng.randint(1, 20)
        ans = n*(n+1)//2

        sample_in = f"{n}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 25)
            tout = tn*(tn+1)//2
            tests.append(TestCase(inp=f"{tn}\n", out=f"{tout}\n"))

        return {
            "id": "REC_SUM_N",
            "title": "Sum of First N Numbers",
            "family": "recursion",
            "task": "Given n, print the sum of first n natural numbers using recursion.",
            "input_format": "One line: n",
            "output_format": "Single integer: sum of first n numbers",
            "constraints": "1 ≤ n ≤ 25",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "reverse_string":
        # Trickier: reverse a string using recursion
        letters = "abcdefghijklmnopqrstuvwxyz"
        n = rng.randint(3, 8)
        s = "".join(rng.choice(letters) for _ in range(n))
        ans = s[::-1]

        sample_in = f"{s}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 10)
            ts = "".join(rng.choice(letters) for _ in range(tn))
            tout = ts[::-1]
            tests.append(TestCase(inp=f"{ts}\n", out=f"{tout}\n"))

        return {
            "id": "REC_REV_STR",
            "title": "Reverse String",
            "family": "recursion",
            "task": "Given a string s, print it reversed using recursion.",
            "input_format": "One line: s",
            "output_format": "Single line: reversed string",
            "constraints": "3 ≤ length of s ≤ 10",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
