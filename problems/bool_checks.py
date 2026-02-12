import random
from .base import TestCase
from typing import Optional

def gen_bool_checks(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Simple “true/false” style checks
    valid_kinds = [
        "is_sorted",
        "all_equal",
        "majority_positive",
        "contains_negative",
        "alternating_signs",
        "all_even",
        "has_duplicate"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "is_sorted":
        n = rng.randint(5, 30)
        arr = [rng.randint(-20, 20) for _ in range(n)]
        if rng.random() < 0.5:
            arr.sort()
        ok = all(arr[i] <= arr[i + 1] for i in range(n - 1))

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-100, 100) for _ in range(tn)]
            if rng.random() < 0.5:
                ta.sort()
            tok = all(ta[i] <= ta[i + 1] for i in range(tn - 1))
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_SORTED",
            "title": "Check if Sorted",
            "family": "bool_checks",
            "task": "Print true if the array is non-decreasing, otherwise print false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "all_equal":
        n = rng.randint(5, 30)
        val = rng.randint(-5, 9)
        arr = [val for _ in range(n)]
        if rng.random() < 0.6:
            arr[rng.randrange(n)] = val + rng.choice([-2, -1, 1, 2])
        ok = all(x == arr[0] for x in arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            v = rng.randint(-10, 10)
            ta = [v for _ in range(tn)]
            if rng.random() < 0.6:
                ta[rng.randrange(tn)] = v + rng.choice([-3, -1, 1, 3])
            tok = all(x == ta[0] for x in ta)
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_ALL_EQUAL",
            "title": "All Elements Equal",
            "family": "bool_checks",
            "task": "Print true if all elements are equal, otherwise print false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "majority_positive":
        n = rng.randint(5, 30)
        arr = [rng.randint(-9, 9) for _ in range(n)]
        pos = sum(1 for x in arr if x > 0)
        ok = pos > (n // 2)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            tpos = sum(1 for x in ta if x > 0)
            tok = tpos > (tn // 2)
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_MAJ_POS",
            "title": "Majority Positive",
            "family": "bool_checks",
            "task": "Print true if more than half the numbers are positive (>0), otherwise print false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "contains_negative":
        n = rng.randint(5, 30)
        arr = [rng.randint(-10, 10) for _ in range(n)]
        ok = any(x < 0 for x in arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            tok = any(x < 0 for x in ta)
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_CONTAINS_NEG",
            "title": "Contains Negative",
            "family": "bool_checks",
            "task": "Print true if at least one number is negative, otherwise print false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
    
    elif kind == "alternating_signs":
        n = rng.randint(3, 30)
        arr = [rng.randint(-20, 20) for _ in range(n)]
        ok = all(arr[i] * arr[i+1] < 0 for i in range(n-1))

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 50)
            ta = [rng.randint(-25, 25) for _ in range(tn)]
            tok = all(ta[i]*ta[i+1] < 0 for i in range(tn-1))
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_ALT_SIGNS",
            "title": "Alternating Signs",
            "family": "bool_checks",
            "task": "Print true if the array alternates between positive and negative numbers, otherwise false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "3 ≤ n ≤ 50",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "all_even":
        n = rng.randint(5, 30)
        arr = [rng.randint(0, 50) for _ in range(n)]
        if rng.random() < 0.4:
            arr[rng.randint(0, n-1)] += 1  # make one number odd sometimes
        ok = all(x % 2 == 0 for x in arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(0, 100) for _ in range(tn)]
            if rng.random() < 0.3:
                ta[rng.randint(0, tn-1)] += 1
            tok = all(x%2==0 for x in ta)
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_ALL_EVEN",
            "title": "All Even",
            "family": "bool_checks",
            "task": "Print true if all numbers are even, otherwise false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100, 0 ≤ values ≤ 100",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "has_duplicate":
        n = rng.randint(5, 30)
        arr = [rng.randint(1, 20) for _ in range(n)]
        if rng.random() < 0.5:
            arr[rng.randint(0, n-1)] = arr[rng.randint(0, n-1)]  # force duplicate sometimes
        ok = len(set(arr)) < n

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(1, 50) for _ in range(tn)]
            if rng.random() < 0.5 and tn > 1:
                ta[rng.randint(0, tn-1)] = ta[rng.randint(0, tn-1)]
            tok = len(set(ta)) < tn
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=("true\n" if tok else "false\n")))

        return {
            "id": "BOOL_HAS_DUP",
            "title": "Has Duplicate",
            "family": "bool_checks",
            "task": "Print true if at least one number appears more than once, otherwise false.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "true or false (lowercase)",
            "constraints": "1 ≤ n ≤ 100, 1 ≤ values ≤ 50",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }


