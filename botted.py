import os
import re
import json
import random
import hashlib
import logging
import datetime
import asyncio
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import discord
from discord.ext import tasks, commands

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
TOKEN = os.getenv("DISCORD_TOKEN")  # required
CHANNEL_ID = int(os.getenv("DAILY_CHANNEL_ID", "0"))  # required
SUBMIT_CHANNEL_ID = int(os.getenv("SUBMIT_CHANNEL_ID", "0"))  # optional: restrict submissions

STATE_FILE = "state.json"

# Philippines time (UTC+8)
PH_TZ = datetime.timezone(datetime.timedelta(hours=8))
POST_TIME = datetime.time(hour=9, minute=0, tzinfo=PH_TZ)

# Judge limits
COMPILE_TIMEOUT_SEC = 12
RUN_TIMEOUT_SEC = 2
MAX_OUTPUT_BYTES = 64_000

# g++ command (adjust if needed)
GPP = os.getenv("GPP", "g++")

# =========================
# DISCORD BOT SETUP
# =========================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# =========================
# STATE HELPERS
# =========================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "day_index": 0,
            "last_posted_date": None,
            "problems_by_date": {}  # date_str -> problem dict (includes hidden tests)
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def today_str_ph() -> str:
    return datetime.datetime.now(PH_TZ).date().isoformat()

# =========================
# PROBLEM MODEL
# =========================
@dataclass
class TestCase:
    inp: str
    out: str

def stable_seed_for_day(day_index: int, date_str: str) -> int:
    h = hashlib.sha256(f"{day_index}|{date_str}|CS1JUDGE".encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def pick_family(day_index: int) -> str:
    families = [
        "arrays_basic",
        "arrays_nested",
        "bool_checks",
        "functions",
        "patterns",
    ]
    return families[day_index % len(families)]

def normalize_output(s: str) -> str:
    # Normalize: trim trailing spaces per line, normalize newlines, allow final newline differences
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    # remove trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")

# =========================
# CS1 PROBLEM GENERATORS
# =========================
def gen_arrays_basic(rng: random.Random) -> dict:
    kind = rng.choice(["max", "count_even", "sum", "minmax"])

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
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{max(ta)}\n"
            ))

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

    if kind == "count_even":
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

    if kind == "sum":
        n = rng.randint(5, 25)
        arr = [rng.randint(-20, 40) for _ in range(n)]
        ans = sum(arr)

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-1000, 1000) for _ in range(tn)]
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{sum(ta)}\n"
            ))

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

    # minmax
    n = rng.randint(5, 25)
    arr = [rng.randint(-50, 80) for _ in range(n)]
    mn, mx = min(arr), max(arr)

    sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
    sample_out = f"{mn} {mx}\n"

    tests = []
    for _ in range(7):
        tn = rng.randint(1, 100)
        ta = [rng.randint(-1000, 1000) for _ in range(tn)]
        tests.append(TestCase(
            inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
            out=f"{min(ta)} {max(ta)}\n"
        ))

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

def gen_arrays_nested(rng: random.Random) -> dict:
    kind = rng.choice(["pair_sum_count", "duplicate_values"])

    if kind == "pair_sum_count":
        n = rng.randint(6, 25)
        arr = [rng.randint(-10, 20) for _ in range(n)]
        i, j = rng.sample(range(n), 2)
        target = arr[i] + arr[j]

        # brute force answer
        ct = 0
        for a in range(n):
            for b in range(a + 1, n):
                if arr[a] + arr[b] == target:
                    ct += 1

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + f"\n{target}\n"
        sample_out = f"{ct}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 60)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            ai, aj = rng.sample(range(tn), 2)
            tt = ta[ai] + ta[aj]
            tct = 0
            for a in range(tn):
                for b in range(a + 1, tn):
                    if ta[a] + ta[b] == tt:
                        tct += 1
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + f"\n{tt}\n",
                out=f"{tct}\n"
            ))

        return {
            "id": "ARR_PAIR_CT",
            "title": "Count Pair Sums",
            "family": "arrays_nested",
            "task": "Given n integers and a target, count how many pairs (i<j) satisfy a[i] + a[j] = target.",
            "input_format": "Line 1: n\nLine 2: n integers\nLine 3: target",
            "output_format": "One integer: number of valid pairs",
            "constraints": "1 ≤ n ≤ 60, values between -50 and 50",
            "note": "Use nested loops (no maps).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    # duplicate_values: count how many distinct values appear more than once
    n = rng.randint(8, 30)
    arr = [rng.randint(1, 12) for _ in range(n)]

    def distinct_duplicates(a: List[int]) -> int:
        dd = 0
        seen = set()
        for x in a:
            if x in seen:
                continue
            # count occurrences using loop (CS1)
            c = 0
            for y in a:
                if y == x:
                    c += 1
            if c >= 2:
                dd += 1
            seen.add(x)
        return dd

    ans = distinct_duplicates(arr)
    sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
    sample_out = f"{ans}\n"

    tests = []
    for _ in range(7):
        tn = rng.randint(1, 80)
        ta = [rng.randint(1, 25) for _ in range(tn)]
        tests.append(TestCase(
            inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
            out=f"{distinct_duplicates(ta)}\n"
        ))

    return {
        "id": "ARR_DUP_DISTINCT",
        "title": "Count Distinct Duplicates",
        "family": "arrays_nested",
        "task": "Given n integers, count how many distinct values appear at least twice.",
        "input_format": "Line 1: n\nLine 2: n integers",
        "output_format": "One integer: count of distinct values with frequency ≥ 2",
        "constraints": "1 ≤ n ≤ 80, values between 1 and 25",
        "note": "Use loops/nested loops only (no maps).",
        "sample_in": sample_in,
        "sample_out": sample_out,
        "tests": [t.__dict__ for t in tests],
    }

def gen_bool_checks(rng: random.Random) -> dict:
    kind = rng.choice(["is_sorted", "all_equal", "majority_positive"])

    if kind == "is_sorted":
        n = rng.randint(5, 30)
        arr = [rng.randint(-20, 20) for _ in range(n)]
        # sometimes make it sorted
        if rng.random() < 0.5:
            arr.sort()
        ok = all(arr[i] <= arr[i+1] for i in range(n-1))

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = ("true\n" if ok else "false\n")

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 100)
            ta = [rng.randint(-100, 100) for _ in range(tn)]
            if rng.random() < 0.5:
                ta.sort()
            tok = all(ta[i] <= ta[i+1] for i in range(tn-1))
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=("true\n" if tok else "false\n")
            ))

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

    if kind == "all_equal":
        n = rng.randint(5, 30)
        val = rng.randint(-5, 9)
        arr = [val for _ in range(n)]
        if rng.random() < 0.6:
            # break equality sometimes
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
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=("true\n" if tok else "false\n")
            ))

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

    # majority_positive
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
        tests.append(TestCase(
            inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
            out=("true\n" if tok else "false\n")
        ))

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

def gen_functions(rng: random.Random) -> dict:
    # Still judged via stdin/stdout, but statement requires a function.
    kind = rng.choice(["sum_array_fn", "count_greater_fn"])

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
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{sum(ta)}\n"
            ))

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

    # count_greater_fn
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

def gen_patterns(rng: random.Random) -> dict:
    kind = rng.choice(["rect", "triangle_num"])

    if kind == "rect":
        r = rng.randint(2, 8)
        c = rng.randint(2, 10)

        # output r lines of c stars
        out = ""
        for _ in range(r):
            out += "*" * c + "\n"

        sample_in = f"{r} {c}\n"
        sample_out = out

        tests = []
        for _ in range(7):
            tr = rng.randint(1, 12)
            tc = rng.randint(1, 20)
            tout = ("*" * tc + "\n") * tr
            tests.append(TestCase(
                inp=f"{tr} {tc}\n",
                out=tout
            ))

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

    # triangle_num: 1..i each row
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
        tests.append(TestCase(
            inp=f"{tn}\n",
            out=tout
        ))

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

def generate_problem(day_index: int, date_str: str) -> dict:
    seed = stable_seed_for_day(day_index, date_str)
    rng = random.Random(seed)

    family = pick_family(day_index)
    if family == "arrays_basic":
        p = gen_arrays_basic(rng)
    elif family == "arrays_nested":
        p = gen_arrays_nested(rng)
    elif family == "bool_checks":
        p = gen_bool_checks(rng)
    elif family == "functions":
        p = gen_functions(rng)
    else:
        p = gen_patterns(rng)

    p["day"] = date_str
    p["seed"] = seed
    p["day_index"] = day_index
    return p

# =========================
# EMBED BUILDER
# =========================
def build_embed(problem: dict) -> discord.Embed:
    title = f"🧩 DAILY MACHINE PROBLEM • {problem['title']}"
    desc = (
        "```fix\n"
        f"CS1 DAILY MP • {problem['family'].upper()}\n"
        "```\n"
        f"**Task**\n{problem['task']}\n\n"
        f"**Input Format**\n{problem['input_format']}\n\n"
        f"**Output Format**\n{problem['output_format']}\n\n"
        f"**Constraints**\n{problem.get('constraints','-')}\n"
    )
    if problem.get("note"):
        desc += f"\n**Note**\n{problem['note']}\n"

    embed = discord.Embed(
        title=title,
        description=desc,
        color=0x5865F2,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    embed.add_field(
        name="Sample Input",
        value=f"```text\n{problem['sample_in']}```",
        inline=False,
    )
    embed.add_field(
        name="Sample Output",
        value=f"```text\n{problem['sample_out']}```",
        inline=False,
    )

    embed.add_field(
        name="How to Submit (C++ only)",
        value="Use `!submit` then paste your full C++ code in a ```cpp``` block (or attach a .cpp file). "
              "No prompts like `Enter n:` — output must match exactly.",
        inline=False,
    )
    embed.set_footer(text=f"Day: {problem['day']} • Seed: {problem['seed']}")
    return embed

# =========================
# JUDGE: COMPILE + RUN
# =========================
CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_cpp_from_message(content: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(content)
    if not m:
        return None
    code = m.group(1).strip()
    if not code:
        return None
    return code

async def read_attachment_cpp(message: discord.Message) -> Optional[str]:
    if not message.attachments:
        return None
    for att in message.attachments:
        if att.filename.lower().endswith(".cpp") or att.filename.lower().endswith(".cc") or att.filename.lower().endswith(".cxx"):
            data = await att.read()
            try:
                return data.decode("utf-8", errors="replace").strip()
            except Exception:
                return None
    return None

async def run_subprocess(cmd: List[str], stdin_data: Optional[bytes], timeout_sec: int) -> Tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout_sec)
        return proc.returncode, out[:MAX_OUTPUT_BYTES], err[:MAX_OUTPUT_BYTES]
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -999, b"", b"TIMEOUT"

async def compile_cpp(code: str, workdir: str) -> Tuple[bool, str]:
    src = os.path.join(workdir, "main.cpp")
    exe = os.path.join(workdir, "main.out")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code + "\n")

    cmd = [GPP, "-std=c++17", "-O2", "-pipe", src, "-o", exe]
    rc, out, err = await run_subprocess(cmd, stdin_data=None, timeout_sec=COMPILE_TIMEOUT_SEC)

    if rc == 0 and os.path.exists(exe):
        return True, ""
    if rc == -999:
        return False, "Compilation timed out."
    return False, (err.decode("utf-8", errors="replace") or "Compilation failed.")

async def run_tests(workdir: str, tests: List[dict]) -> Tuple[bool, int, str, Optional[dict]]:
    exe = os.path.join(workdir, "main.out")
    for idx, t in enumerate(tests, start=1):
        inp = t["inp"].encode("utf-8")
        expected = normalize_output(t["out"])

        rc, out, err = await run_subprocess([exe], stdin_data=inp, timeout_sec=RUN_TIMEOUT_SEC)
        if rc == -999:
            return False, idx, "Time Limit Exceeded", {"test": t}
        if rc != 0:
            msg = err.decode("utf-8", errors="replace").strip()
            if msg == "TIMEOUT":
                msg = "Time Limit Exceeded"
            return False, idx, f"Runtime Error (exit {rc})\n{msg}", {"test": t}

        got = normalize_output(out.decode("utf-8", errors="replace"))
        if got != expected:
            return False, idx, "Wrong Answer", {"test": t, "expected": expected, "got": got}

    return True, len(tests), "Accepted", None

# =========================
# DAILY LOOP
# =========================
@tasks.loop(time=POST_TIME)
async def post_daily_problem():
    if CHANNEL_ID == 0:
        logging.warning("DAILY_CHANNEL_ID not set.")
        return

    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        logging.warning("Channel not found. Check DAILY_CHANNEL_ID and bot permissions.")
        return

    state = load_state()
    date_str = today_str_ph()

    if state.get("last_posted_date") == date_str:
        logging.info("Already posted today. Skipping.")
        return

    day_index = int(state.get("day_index", 0))
    problem = generate_problem(day_index, date_str)

    await channel.send("⚙️ **DAILY MP DROP:** Solve it in C++ and submit with `!submit`.", embed=build_embed(problem))

    # Persist with hidden tests
    pb = state.get("problems_by_date", {})
    pb[date_str] = problem
    state["problems_by_date"] = pb
    state["day_index"] = day_index + 1
    state["last_posted_date"] = date_str
    save_state(state)

    logging.info(f"Posted MP for {date_str} (day_index={day_index}).")

@post_daily_problem.before_loop
async def before_post_daily():
    await bot.wait_until_ready()
    logging.info("CS1 Judge armed.")

@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user} (id: {bot.user.id})")
    if not post_daily_problem.is_running():
        post_daily_problem.start()

# =========================
# COMMANDS
# =========================
@bot.command(name="postnow")
@commands.has_permissions(administrator=True)
async def postnow(ctx: commands.Context):
    """Admin: post today's problem immediately and advance counter."""
    if CHANNEL_ID == 0:
        await ctx.send("❌ DAILY_CHANNEL_ID not set.")
        return

    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        await ctx.send("❌ Target channel not found.")
        return

    state = load_state()
    date_str = today_str_ph()
    day_index = int(state.get("day_index", 0))
    problem = generate_problem(day_index, date_str)

    await channel.send("⚙️ **DAILY MP DROP:** Solve it in C++ and submit with `!submit`.", embed=build_embed(problem))

    pb = state.get("problems_by_date", {})
    pb[date_str] = problem
    state["problems_by_date"] = pb
    state["day_index"] = day_index + 1
    state["last_posted_date"] = date_str
    save_state(state)

    await ctx.send("✅ Posted and advanced today’s MP.")

@bot.command(name="today")
async def today(ctx: commands.Context):
    """Show today's stored problem (embed only)."""
    state = load_state()
    date_str = today_str_ph()
    p = state.get("problems_by_date", {}).get(date_str)
    if not p:
        await ctx.send("❌ No problem stored for today yet. Ask admin to `!postnow` or wait for the schedule.")
        return
    await ctx.send(embed=build_embed(p))

@bot.command(name="submit")
async def submit(ctx: commands.Context):
    """Submit C++ code in a cpp code block or attach a .cpp file."""
    # Optional channel restriction
    if SUBMIT_CHANNEL_ID and ctx.channel.id != SUBMIT_CHANNEL_ID:
        await ctx.send(f"❌ Submit only in <#{SUBMIT_CHANNEL_ID}>.")
        return

    state = load_state()
    date_str = today_str_ph()
    problem = state.get("problems_by_date", {}).get(date_str)
    if not problem:
        await ctx.send("❌ No active problem for today. Ask admin to `!postnow`.")
        return

    msg: discord.Message = ctx.message

    code = await read_attachment_cpp(msg)
    if not code:
        code = extract_cpp_from_message(msg.content)

    if not code:
        await ctx.send("❌ I didn't find C++ code. Paste it inside a ```cpp``` block or attach a .cpp file.")
        return

    # Quick policy reminder (judge-style)
    if re.search(r'cout\s*<<\s*".*enter', code, flags=re.IGNORECASE):
        await ctx.send("⚠️ Heads up: prompts like `Enter n:` will usually cause Wrong Answer. Output should be answer only.")

    await ctx.send("🧪 Compiling and running tests...")

    with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
        ok, cerr = await compile_cpp(code, workdir)
        if not ok:
            # trim compile errors
            cerr = cerr.strip()
            if len(cerr) > 1800:
                cerr = cerr[:1800] + "\n... (truncated)"
            await ctx.send(f"❌ **Compilation Error**\n```text\n{cerr}\n```")
            return

        tests = problem.get("tests", [])
        passed, where, verdict, details = await run_tests(workdir, tests)

        if passed:
            await ctx.send(f"✅ **Accepted** — {where}/{where} tests passed.\n"
                           f"Problem: **{problem['title']}** (Day {problem['day']})")
            return

        # Failure detail
        if verdict == "Wrong Answer" and details:
            t = details["test"]
            exp = details["expected"]
            got = details["got"]
            await ctx.send(
                f"❌ **Wrong Answer** — failed test #{where}\n"
                f"**Input**\n```text\n{t['inp']}```"
                f"**Expected**\n```text\n{exp}```"
                f"**Got**\n```text\n{got}```"
            )
            return

        if verdict == "Time Limit Exceeded" and details:
            t = details["test"]
            await ctx.send(
                f"⏱️ **Time Limit Exceeded** — test #{where}\n"
                f"**Input**\n```text\n{t['inp']}```"
                f"Tip: check nested loops / infinite loops."
            )
            return

        # Runtime error
        await ctx.send(f"💥 **{verdict}** — test #{where}")

@bot.command(name="setup")
@commands.has_permissions(administrator=True)
async def setup_cmd(ctx: commands.Context):
    ch = bot.get_channel(CHANNEL_ID) if CHANNEL_ID else None
    await ctx.send(
        f"✅ Bot online.\n"
        f"- Daily channel ID: `{CHANNEL_ID}` (name: `{getattr(ch, 'name', None)}`)\n"
        f"- Submit channel ID: `{SUBMIT_CHANNEL_ID}` (0 means any channel)\n"
        f"- Post time: `{POST_TIME}` (PH time)"
    )

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN env var missing. Set DISCORD_TOKEN before running.")
if CHANNEL_ID == 0:
    raise RuntimeError("DAILY_CHANNEL_ID env var missing. Set DAILY_CHANNEL_ID before running.")

@bot.command()
async def ping(ctx):
    await ctx.send("pong")


bot.run(TOKEN)
