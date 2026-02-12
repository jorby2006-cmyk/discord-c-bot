import random
from .base import TestCase
from typing import Optional

def gen_math_logic(rng: random.Random, kind: Optional[str] = None) -> dict:
    # To make your head explode
    valid_kinds = [
        "sum_digits",
        "prime_check",
        "fibonacci_n"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "sum_digits":
        # Easy: Sum of digits of a number
        n = rng.randint(10, 9999)
        ans = sum(int(d) for d in str(n))

        sample_in = f"{n}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 99999)
            tout = sum(int(d) for d in str(tn))
            tests.append(TestCase(inp=f"{tn}\n", out=f"{tout}\n"))

        return {
            "id": "ML_SUM_DIGITS",
            "title": "Sum of Digits",
            "family": "math_logic",
            "task": "Given an integer n, print the sum of its digits.",
            "input_format": "One line: n",
            "output_format": "Single integer: sum of digits",
            "constraints": "1 ≤ n ≤ 99999",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "prime_check":
        # Medium: Check if a number is prime
        n = rng.randint(2, 100)
        ans = "YES" if all(n % i != 0 for i in range(2, int(n**0.5)+1)) else "NO"

        sample_in = f"{n}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(2, 200)
            tout = "YES" if all(tn % i != 0 for i in range(2, int(tn**0.5)+1)) else "NO"
            tests.append(TestCase(inp=f"{tn}\n", out=f"{tout}\n"))

        return {
            "id": "ML_PRIME_CHECK",
            "title": "Prime Number Check",
            "family": "math_logic",
            "task": "Given an integer n, print YES if it is prime, NO otherwise.",
            "input_format": "One line: n",
            "output_format": "YES or NO",
            "constraints": "2 ≤ n ≤ 200",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "fibonacci_n":
        # Trickier: Find the nth Fibonacci number
        n = rng.randint(1, 20)
        a, b = 0, 1
        for _ in range(n-1):
            a, b = b, a+b
        ans = b if n > 1 else 0

        sample_in = f"{n}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 25)
            a, b = 0, 1
            for _ in range(tn-1):
                a, b = b, a+b
            tout = b if tn > 1 else 0
            tests.append(TestCase(inp=f"{tn}\n", out=f"{tout}\n"))

        return {
            "id": "ML_FIB_N",
            "title": "Nth Fibonacci Number",
            "family": "math_logic",
            "task": "Given n, print the nth Fibonacci number (0-based: F1=0, F2=1).",
            "input_format": "One line: n",
            "output_format": "Single integer: nth Fibonacci number",
            "constraints": "1 ≤ n ≤ 25",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
