import random
from .base import TestCase
from typing import Optional

def gen_arrays_basic(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Basic array warmups
    valid_kinds = [
        "max",
        "count_even",
        "sum",
        "minmax",
        "count_odd",
        "average",
        "count_positive"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "max":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 80) for _ in range(n)]
        ans = max(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{max(ta)}\n"))

        return {
            "id": "ARR_MAX",
            "title": "Find the Maximum",
            "family": "arrays_basic",
            "task": "Given n integers, print the maximum value.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: the maximum value",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_even":
        n = rng.randint(6, 30)
        arr = [rng.randint(-30, 60) for _ in range(n)]
        ans = sum(1 for x in arr if x % 2 == 0)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{sum(1 for x in ta if x % 2 == 0)}\n"
            ))

        return {
            "id": "ARR_EVEN_CT",
            "title": "Count Even Numbers",
            "family": "arrays_basic",
            "task": "Given n integers, count how many are even.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: count of even numbers",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "sum":
        n = rng.randint(5, 25)
        arr = [rng.randint(-20, 40) for _ in range(n)]
        ans = sum(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{sum(ta)}\n"))

        return {
            "id": "ARR_SUM",
            "title": "Sum of Array",
            "family": "arrays_basic",
            "task": "Given n integers, print their sum.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: the sum",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "minmax":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 80) for _ in range(n)]
        mn, mx = min(arr), max(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{mn} {mx}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{min(ta)} {max(ta)}\n"))

        return {
            "id": "ARR_MINMAX",
            "title": "Min and Max",
            "family": "arrays_basic",
            "task": "Given n integers, print the minimum and maximum values.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "Two integers: min then max (space-separated)",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_odd":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 80) for _ in range(n)]
        ans = sum(1 for x in arr if x % 2 != 0)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{sum(1 for x in ta if x % 2 != 0)}\n"
            ))

        return {
            "id": "ARR_ODD_CT",
            "title": "Count Odd Numbers",
            "family": "arrays_basic",
            "task": "Given n integers, count how many are odd.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: count of odd numbers",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "average":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 80) for _ in range(n)]
        ans = sum(arr) / n

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans:.2f}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            avg = sum(ta) / tn
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{avg:.2f}\n"
            ))

        return {
            "id": "ARR_AVG",
            "title": "Average of Array",
            "family": "arrays_basic",
            "task": "Given n integers, print their average (2 decimal places).",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One float: the average rounded to 2 decimals",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_positive":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 50) for _ in range(n)]
        ans = sum(1 for x in arr if x > 0)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{sum(1 for x in ta if x > 0)}\n"
            ))

        return {
            "id": "ARR_POS_CT",
            "title": "Count Positive Numbers",
            "family": "arrays_basic",
            "task": "Given n integers, count how many are positive.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: count of positive numbers",
            "constraints": "1 ≤ n ≤ 100, values between -1000 and 1000",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

