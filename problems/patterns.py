import random
from .base import TestCase
from typing import Optional

def gen_patterns(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Printing patterns
    valid_kinds = [
        "rect",
        "triangle_num",
        "right_triangle",
        "inverted_num_triangle",
        "pyramid",
        "diamond",
        "hollow_square"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "rect":
        r = rng.randint(2, 8)
        c = rng.randint(2, 10)
        out = ("*" * c + "\n") * r

        sample_in = f"{r} {c}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tr = rng.randint(1, 12)
            tc = rng.randint(1, 20)
            tout = ("*" * tc + "\n") * tr
            tests.append(TestCase(inp=f"{tr} {tc}\n", out=tout))

        return {
            "id": "PAT_RECT",
            "title": "Print a Rectangle",
            "family": "patterns",
            "task": "Given rows R and columns C, print an R×C rectangle of '*' characters.",
            "input_format": "One line: R C",
            "output_format": "R lines, each with C '*' characters",
            "constraints": "1 ≤ R ≤ 12, 1 ≤ C ≤ 20",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "triangle_num":
        n = rng.randint(3, 9)
        out = ""
        for i in range(1, n + 1):
            out += "".join(str(x) for x in range(1, i + 1)) + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 12)
            tout = ""
            for i in range(1, tn + 1):
                tout += "".join(str(x) for x in range(1, i + 1)) + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_NUM_TRI",
            "title": "Number Triangle",
            "family": "patterns",
            "task": "Given n, print n lines. Line i contains the numbers 1..i (no spaces).",
            "input_format": "One line: n",
            "output_format": "n lines forming the triangle",
            "constraints": "1 ≤ n ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "right_triangle":
        n = rng.randint(3, 10)
        out = ""
        for i in range(1, n + 1):
            out += " " * (n - i) + "*" * i + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 12)
            tout = ""
            for i in range(1, tn + 1):
                tout += " " * (tn - i) + "*" * i + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_RIGHT_TRI",
            "title": "Right-Aligned Triangle",
            "family": "patterns",
            "task": "Given n, print a right-aligned triangle of '*' with n lines.",
            "input_format": "One line: n",
            "output_format": "n lines forming the triangle",
            "constraints": "1 ≤ n ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "inverted_num_triangle":
        n = rng.randint(3, 9)
        out = ""
        for i in range(n, 0, -1):
            out += "".join(str(x) for x in range(1, i + 1)) + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 12)
            tout = ""
            for i in range(tn, 0, -1):
                tout += "".join(str(x) for x in range(1, i + 1)) + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_INV_NUM_TRI",
            "title": "Inverted Number Triangle",
            "family": "patterns",
            "task": "Given n, print n lines. Line i contains numbers 1..(n-i+1).",
            "input_format": "One line: n",
            "output_format": "n lines forming the triangle",
            "constraints": "1 ≤ n ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "pyramid":
        n = rng.randint(3, 8)
        out = ""
        for i in range(1, n + 1):
            out += " " * (n - i) + "*" * (2 * i - 1) + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 10)
            tout = ""
            for i in range(1, tn + 1):
                tout += " " * (tn - i) + "*" * (2 * i - 1) + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_PYRAMID",
            "title": "Pyramid of Stars",
            "family": "patterns",
            "task": "Given n, print a pyramid of '*' with n lines. Each line centered with increasing stars.",
            "input_format": "One line: n",
            "output_format": "n lines forming the pyramid",
            "constraints": "1 ≤ n ≤ 10",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "diamond":
        n = rng.randint(3, 7)
        out = ""
        # top half
        for i in range(1, n + 1):
            out += " " * (n - i) + "*" * (2 * i - 1) + "\n"
        # bottom half
        for i in range(n-1, 0, -1):
            out += " " * (n - i) + "*" * (2 * i - 1) + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 8)
            tout = ""
            for i in range(1, tn + 1):
                tout += " " * (tn - i) + "*" * (2 * i - 1) + "\n"
            for i in range(tn-1, 0, -1):
                tout += " " * (tn - i) + "*" * (2 * i - 1) + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_DIAMOND",
            "title": "Diamond of Stars",
            "family": "patterns",
            "task": "Given n, print a diamond of '*' characters with 2*n-1 lines.",
            "input_format": "One line: n",
            "output_format": "2*n-1 lines forming the diamond",
            "constraints": "1 ≤ n ≤ 8",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "hollow_square":
        n = rng.randint(3, 10)
        out = ""
        for i in range(n):
            if i == 0 or i == n-1:
                out += "*" * n + "\n"
            else:
                out += "*" + " " * (n-2) + "*" + "\n"

        sample_in = f"{n}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 12)
            tout = ""
            for i in range(tn):
                if i == 0 or i == tn-1:
                    tout += "*" * tn + "\n"
                else:
                    tout += "*" + " " * (tn-2) + "*" + "\n"
            tests.append(TestCase(inp=f"{tn}\n", out=tout))

        return {
            "id": "PAT_HOLLOW_SQ",
            "title": "Hollow Square",
            "family": "patterns",
            "task": "Given n, print an n×n square with '*' on borders and spaces inside.",
            "input_format": "One line: n",
            "output_format": "n lines forming the hollow square",
            "constraints": "1 ≤ n ≤ 12",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
