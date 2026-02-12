import random
from .base import TestCase
from typing import Optional

def gen_functions(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Makes you wanna use functions.
    valid_kinds = [
        "sum_array_fn",
        "count_greater_fn",
        "product_array",
        "max_element_fn",
        "count_even_fn",
        "reverse_array_fn",
        "sum_squares_fn"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "sum_array_fn":
        n = rng.randint(5, 25)
        arr = [rng.randint(-20, 30) for _ in range(n)]
        ans = sum(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{sum(ta)}\n"))

        return {
            "id": "FN_SUM_ARRAY",
            "title": "Sum Array (Function Required)",
            "family": "functions",
            "task": "Write a function sumArray that returns the sum of the array. Then print the result.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: sum",
            "constraints": "1 ≤ n ≤ 100",
            "note": "Your solution MUST use a user-defined function (as required by your instructor).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_greater_fn":
        n = rng.randint(6, 25)
        arr = [rng.randint(-20, 40) for _ in range(n)]
        k = rng.randint(-10, 25)
        ans = sum(1 for x in arr if x > k)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + f"\n{k}\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tk = rng.randint(-1000, 1000)
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + f"\n{tk}\n",
                out=f"{sum(1 for x in ta if x > tk)}\n"
            ))

        return {
            "id": "FN_COUNT_GT",
            "title": "Count Greater Than K (Function Required)",
            "family": "functions",
            "task": "Write a function countGreater that counts how many array values are greater than K. Then print the count.",
            "input_format": "Line 1: n\nLine 2: n integers\nLine 3: K",
            "output_format": "One integer: count of values > K",
            "constraints": "1 ≤ n ≤ 100",
            "note": "Your solution MUST use a user-defined function (as required by your instructor).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "product_array_fn":
        n = rng.randint(3, 15)
        arr = [rng.randint(1, 10) for _ in range(n)]
        prod = 1
        for x in arr:
            prod *= x

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{prod}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 20)
            ta = [rng.randint(1, 15) for _ in range(tn)]
            tp = 1
            for x in ta: tp *= x
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{tp}\n"))

        return {
            "id": "FN_PRODUCT_ARRAY",
            "title": "Product of Array (Function Required)",
            "family": "functions",
            "task": "Write a function productArray that returns the product of all array elements. Then print the result.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: product",
            "constraints": "1 ≤ n ≤ 20, values between 1 and 15",
            "note": "Your solution MUST use a user-defined function.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "max_element_fn":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 50) for _ in range(n)]
        mx = max(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{mx}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 50)
            ta = [rng.randint(-100, 100) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{max(ta)}\n"))

        return {
            "id": "FN_MAX_ELEMENT",
            "title": "Maximum Element (Function Required)",
            "family": "functions",
            "task": "Write a function maxElement that returns the largest element in the array. Then print it.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: maximum value",
            "constraints": "1 ≤ n ≤ 50, values between -100 and 100",
            "note": "Your solution MUST use a user-defined function.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_even_fn":
        n = rng.randint(5, 30)
        arr = [rng.randint(0, 50) for _ in range(n)]
        ce = sum(1 for x in arr if x % 2 == 0)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ce}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 50)
            ta = [rng.randint(0, 100) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{sum(1 for x in ta if x%2==0)}\n"))

        return {
            "id": "FN_COUNT_EVEN",
            "title": "Count Even Numbers (Function Required)",
            "family": "functions",
            "task": "Write a function countEven that returns the number of even elements in the array. Then print it.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: count of even numbers",
            "constraints": "1 ≤ n ≤ 50, values between 0 and 100",
            "note": "Your solution MUST use a user-defined function.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "reverse_array_fn":
        n = rng.randint(5, 25)
        arr = [rng.randint(-50, 50) for _ in range(n)]
        rev = arr[::-1]

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = " ".join(map(str, rev)) + "\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 50)
            ta = [rng.randint(-100, 100) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=" ".join(map(str, ta[::-1]))+"\n"))

        return {
            "id": "FN_REVERSE_ARRAY",
            "title": "Reverse Array (Function Required)",
            "family": "functions",
            "task": "Write a function reverseArray that returns the array in reverse order. Then print it.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "n integers in reverse order",
            "constraints": "1 ≤ n ≤ 50, values between -100 and 100",
            "note": "Your solution MUST use a user-defined function.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "sum_squares_fn":
        n = rng.randint(3, 25)
        arr = [rng.randint(-10, 10) for _ in range(n)]
        ssum = sum(x*x for x in arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ssum}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 50)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{sum(x*x for x in ta)}\n"))

        return {
            "id": "FN_SUM_SQUARES",
            "title": "Sum of Squares (Function Required)",
            "family": "functions",
            "task": "Write a function sumSquares that returns the sum of squares of all array elements. Then print it.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: sum of squares",
            "constraints": "1 ≤ n ≤ 50, values between -50 and 50",
            "note": "Your solution MUST use a user-defined function.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
