# botted.py — CS1 Daily MP + C++ Judge (Slash Commands Edition)
#
# ✅ Prefix commands enabled via ! (e.g. !submit)
# ✅ Keeps: daily MP posting, judging, cooldowns, hints, leaderboard, admin/dev tools
#
# Requirements:
#   pip install -U discord.py python-dotenv
#
# ENV:
#   DISCORD_TOKEN=...
#   DAILY_CHANNEL_ID=123
#   SUBMIT_CHANNEL_ID=123 (optional, 0 = any)
#   GPP=C:\msys64\ucrt64\bin\g++.exe   (Windows/MSYS2) or g++ (Linux)
#
# Notes:
# - Slash commands can’t read “message content” unless you also implement legacy prefix handling.
#   Here we go all-in on slash commands.
# - /submit takes either a code text input OR an attachment (.cpp/.cc/.cxx/.txt).

from __future__ import annotations

import os
import re
import json
import random
import hashlib
import logging
import datetime
import asyncio
import tempfile
import subprocess
import time
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import discord
from discord import app_commands
from discord.ext import tasks

from problems.arrays_basic import gen_arrays_basic
from problems.arrays_nested import gen_arrays_nested
from problems.bool_checks import gen_bool_checks
from problems.functions import gen_functions
from problems.patterns import gen_patterns
from problems.strings import gen_strings
from problems.math_logic import gen_math_logic
from problems.recursion import gen_recursion
from problems.stl_intro import gen_stl_intro

from family_kinds import family_kinds

# Optional: load .env for local runs (DON'T commit .env)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# CONFIG (ENV VARS ONLY)
# =========================
TOKEN = os.getenv("DISCORD_TOKEN")  # required
DAILY_CHANNEL_ID = int(os.getenv("DAILY_CHANNEL_ID", "0"))  # required
SUBMIT_CHANNEL_ID = int(os.getenv("SUBMIT_CHANNEL_ID", "0"))  # optional; 0 = any channel

STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Philippines time (UTC+8). Daily post is at 09:00 AM PH time.
PH_TZ = datetime.timezone(datetime.timedelta(hours=8))
POST_TIME = datetime.time(hour=9, minute=0, tzinfo=PH_TZ)

# Judge limits
COMPILE_TIMEOUT_SEC = int(os.getenv("COMPILE_TIMEOUT_SEC", "12"))
RUN_TIMEOUT_SEC = int(os.getenv("RUN_TIMEOUT_SEC", "2"))
MAX_OUTPUT_BYTES = int(os.getenv("MAX_OUTPUT_BYTES", "64000"))
MAX_RUN_MEMORY_MB = int(os.getenv("MAX_RUN_MEMORY_MB", "256"))
MAX_RUN_NPROC = int(os.getenv("MAX_RUN_NPROC", "64"))
MAX_RUN_CPU_SEC = int(os.getenv("MAX_RUN_CPU_SEC", "3"))

MAX_CODE_BYTES = int(os.getenv("MAX_CODE_BYTES", "100000"))  # ~100KB

SUBMIT_COOLDOWN_SEC = int(os.getenv("SUBMIT_COOLDOWN_SEC", "15"))
COOLDOWN_AFTER_ACCEPT_SEC = int(os.getenv("COOLDOWN_AFTER_ACCEPT_SEC", str(SUBMIT_COOLDOWN_SEC)))
COOLDOWN_AFTER_FAIL_SEC = int(os.getenv("COOLDOWN_AFTER_FAIL_SEC", str(SUBMIT_COOLDOWN_SEC)))

ENFORCE_SKILLS = os.getenv("ENFORCE_SKILLS", "true").strip().lower() in ("1", "true", "yes", "y", "on")
TUTOR_FULL_CODE = os.getenv("TUTOR_FULL_CODE", "false").strip().lower() in ("1", "true", "yes", "y", "on")

ADMIN_ROLES = [r.strip() for r in os.getenv("ADMIN_ROLES", "Root Admin").split(",") if r.strip()]

# g++ command
GPP = os.getenv("GPP", "g++")
IS_WINDOWS = (os.name == "nt")

SUBMIT_LOCK = asyncio.Lock()

BOT_START_MONO = time.monotonic()
JUDGE_METRICS: Dict[str, int] = {
    "submissions": 0,
    "accepted": 0,
    "compile_errors": 0,
    "wrong_answers": 0,
    "runtime_errors": 0,
    "timeouts": 0,
    "output_limit_exceeded": 0,
}

ERR_CONFIG = "CFG001"
ERR_NO_CODE = "SUB001"
ERR_AMBIGUOUS_CODE = "SUB002"
ERR_CODE_TOO_LARGE = "SUB003"
ERR_COMPILE_TIMEOUT = "JDG101"
ERR_COMPILE_FAIL = "JDG102"
ERR_RUN_TIMEOUT = "JDG201"
ERR_RUNTIME = "JDG202"
ERR_WRONG_ANSWER = "JDG203"
ERR_OUTPUT_LIMIT = "JDG204"
ERR_INTERNAL = "JDG500"

SKILL_HARD_FAIL_CONFIDENCE = float(os.getenv("SKILL_HARD_FAIL_CONFIDENCE", "0.75"))
SKILL_WARN_CONFIDENCE = float(os.getenv("SKILL_WARN_CONFIDENCE", "0.45"))
HINTS_PER_DAY_LIMIT = int(os.getenv("HINTS_PER_DAY_LIMIT", "5"))

BOT_UPDATES_VERSION = "2026-02-prefix-bang"
COMMAND_PREFIX = "!"

# =========================
# DISCORD BOT SETUP
# =========================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def is_admin_member(member: discord.Member) -> bool:
    try:
        if member.guild_permissions.administrator:
            return True
        names = {r.name for r in member.roles}
        return any(ar in names for ar in ADMIN_ROLES)
    except Exception:
        return False

def today_str_ph() -> str:
    return datetime.datetime.now(PH_TZ).date().isoformat()

def next_post_time_ph(now: Optional[datetime.datetime] = None) -> datetime.datetime:
    now = now or datetime.datetime.now(PH_TZ)
    today = now.date()
    candidate = datetime.datetime.combine(today, datetime.time(POST_TIME.hour, POST_TIME.minute, tzinfo=PH_TZ))
    if now >= candidate:
        candidate = candidate + datetime.timedelta(days=1)
    return candidate

# =========================
# STATE HELPERS
# =========================
def default_state() -> dict:
    return {
        "day_index": 0,
        "last_posted_date": None,
        "problems_by_date": {},
        "cooldowns": {},
        "compile_errors": {},
        "hint_usage": {},
        "scores": {},
        "audit_log": [],
    }

def _state_with_defaults(raw: dict) -> dict:
    base = default_state()
    base.update(raw or {})
    for k in ("problems_by_date", "cooldowns", "compile_errors", "hint_usage", "scores"):
        if not isinstance(base.get(k), dict):
            base[k] = {}
    if not isinstance(base.get("audit_log"), list):
        base["audit_log"] = []
    return base

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return default_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return _state_with_defaults(json.load(f))
    except Exception:
        return default_state()

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def append_audit(state: dict, entry: dict) -> None:
    logs = state.get("audit_log", [])
    logs.append({"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), **entry})
    state["audit_log"] = logs[-250:]

def cooldown_remaining_sec(state: dict, user_id: int) -> int:
    key = str(user_id)
    now = time.time()
    until = float(state.get("cooldowns", {}).get(key, 0.0))
    return max(0, int(until - now))

def set_cooldown(state: dict, user_id: int, sec: int) -> None:
    state.setdefault("cooldowns", {})[str(user_id)] = time.time() + max(0, sec)

def record_compile_error(state: dict, user_id: int, msg: str) -> None:
    state.setdefault("compile_errors", {})[str(user_id)] = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "msg": msg[:2000],
    }

def current_week_key_ph() -> str:
    now = datetime.datetime.now(PH_TZ).date()
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

def _score_row(state: dict, user_id: int, display: str) -> dict:
    scores = state.setdefault("scores", {})
    row = scores.get(str(user_id), {
        "name": display,
        "accepted": 0,
        "submissions": 0,
        "streak": 0,
        "last_accept_date": None,
        "week_key": current_week_key_ph(),
        "weekly_accepts": 0,
    })
    row["name"] = display
    if row.get("week_key") != current_week_key_ph():
        row["week_key"] = current_week_key_ph()
        row["weekly_accepts"] = 0
    scores[str(user_id)] = row
    return row

def score_submission(state: dict, user_id: int, display: str, accepted: bool, date_str: str) -> None:
    row = _score_row(state, user_id, display)
    row["submissions"] += 1
    if not accepted:
        return
    row["accepted"] += 1
    row["weekly_accepts"] += 1
    prev = row.get("last_accept_date")
    if prev:
        try:
            prev_d = datetime.date.fromisoformat(prev)
            cur_d = datetime.date.fromisoformat(date_str)
            delta = (cur_d - prev_d).days
            if delta == 1:
                row["streak"] = int(row.get("streak", 0)) + 1
            elif delta > 1:
                row["streak"] = 1
        except Exception:
            row["streak"] = 1
    else:
        row["streak"] = 1
    row["last_accept_date"] = date_str

def consume_hint(state: dict, user_id: int, date_str: str) -> Tuple[bool, int]:
    usage = state.setdefault("hint_usage", {})
    per_user = usage.setdefault(str(user_id), {})
    used = int(per_user.get(date_str, 0))
    if used >= HINTS_PER_DAY_LIMIT:
        return False, used
    per_user[date_str] = used + 1
    return True, used + 1

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
    families = ["arrays_basic", "arrays_nested", "bool_checks", "functions", "patterns",
                "strings", "math_logic", "recursion", "stl_intro"]
    return families[day_index % len(families)]

def normalize_output(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")

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
    elif family == "patterns":
        p = gen_patterns(rng)
    elif family == "strings":
        p = gen_strings(rng)
    elif family == "math_logic":
        p = gen_math_logic(rng)
    elif family == "recursion":
        p = gen_recursion(rng)
    else:
        p = gen_stl_intro(rng)

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
    embed.add_field(name="Sample Input", value=f"```text\n{problem['sample_in']}```", inline=False)
    embed.add_field(name="Sample Output", value=f"```text\n{problem['sample_out']}```", inline=False)

    embed.add_field(
        name="How to Submit (C++ only)",
        value="Use `!submit` and paste your full C++ code (or attach a .cpp file). "
              "No prompts like `Enter n:` — output must match exactly.",
        inline=False,
    )
    embed.set_footer(text=f"Day: {problem['day']} • Seed: {problem['seed']}")
    return embed

# =========================
# CODE EXTRACTION (SLASH)
# =========================
CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_cpp_blocks(content: str) -> List[str]:
    return [m.group(1).strip() for m in CODE_BLOCK_RE.finditer(content or "") if m.group(1).strip()]

async def get_code_from_slash_inputs(code_text: Optional[str], attachment: Optional[discord.Attachment]) -> Tuple[Optional[str], Optional[str]]:
    candidates: List[Tuple[str, str]] = []

    if code_text:
        blocks = extract_cpp_blocks(code_text)
        if blocks:
            for i, b in enumerate(blocks, start=1):
                candidates.append((f"code_text:block_{i}", b))
        else:
            candidates.append(("code_text:raw", code_text.strip()))

    if attachment is not None:
        fn = attachment.filename.lower()
        if fn.endswith((".cpp", ".cc", ".cxx", ".txt")):
            data = await attachment.read()
            text = data.decode("utf-8", errors="replace").strip()
            blocks = extract_cpp_blocks(text)
            if blocks:
                for i, b in enumerate(blocks, start=1):
                    candidates.append((f"attachment:{attachment.filename}:block_{i}", b))
            elif text:
                candidates.append((f"attachment:{attachment.filename}", text))

    if not candidates:
        return None, f"{ERR_NO_CODE}: I didn't find C++ code. Paste code or attach a .cpp file."
    if len(candidates) > 1:
        names = ", ".join(name for name, _ in candidates[:4])
        suffix = "..." if len(candidates) > 4 else ""
        return None, f"{ERR_AMBIGUOUS_CODE}: Found multiple code payloads ({names}{suffix}). Submit exactly one payload."
    return candidates[0][1], None

async def get_code_from_message(message: discord.Message, raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    code_text = raw_text.strip() if raw_text else None
    attachment = message.attachments[0] if message.attachments else None
    return await get_code_from_slash_inputs(code_text, attachment)

def exe_path(workdir: str) -> str:
    return os.path.join(workdir, "main.exe" if IS_WINDOWS else "main.out")

# =========================
# BETTER DIFF UTILITIES
# =========================
def first_mismatch_line(expected: str, got: str) -> Tuple[int, str, str]:
    e_lines = expected.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    g_lines = got.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = max(len(e_lines), len(g_lines))
    for i in range(n):
        e = e_lines[i] if i < len(e_lines) else "<missing>"
        g = g_lines[i] if i < len(g_lines) else "<missing>"
        if e != g:
            return i + 1, e, g
    return 0, "", ""

def clamp_block(s: str, limit: int = 1200) -> str:
    s = s if s.endswith("\n") else s + "\n"
    if len(s) <= limit:
        return s
    return s[:limit] + "\n... (truncated)\n"

# =========================
# TUTOR HELPERS
# =========================
def get_today_problem_from_state() -> Optional[dict]:
    st = load_state()
    date_str = today_str_ph()
    return st.get("problems_by_date", {}).get(date_str)

def _hint_bank(problem: dict) -> Dict[str, List[str]]:
    fam = str(problem.get("family", "")).lower()
    pid = str(problem.get("id", "")).upper()

    by_id: Dict[str, List[str]] = {}

    if pid in by_id:
        return {pid: by_id[pid]}

    by_family: Dict[str, List[str]] = {
        "arrays_basic": [
            "Read **n**, then read **n numbers**. Store them in an array/vector so you can process them using `a[i]`.",
            "Use a loop to fill `vector<long long> a(n)`, then another loop to compute the answer (count/sum/max/etc.).",
            "Structure: `vector<long long> a(n); for i: cin>>a[i];` then process with `a[i]`."
        ],
        "arrays_nested": [
            "This one usually needs **nested loops** (a loop inside a loop) + an array/matrix.",
            "If it's 2D: store inputs, then use `for (i) for (j)` to compute the required value.",
            "Outer loop = rows/first index, inner loop = columns/second index. Avoid `map`/`unordered_map`."
        ],
        "bool_checks": [
            "Use comparisons (`<`, `>`, `==`, `!=`, etc.) and output the required decision result.",
            "Compute a condition, store in `bool ok`, then print exactly as required by the statement.",
            "Common pattern: `bool ok = (condition);` then print `1/0` or `YES/NO`."
        ],
        "functions": [
            "Your instructor wants a **user-defined function** besides `main()`.",
            "Write a function that does the core work, then call it from `main()` and print the result.",
            "Skeleton: `returnType fn(params){ ... }` then `main`: read input → call `fn(...)` → print."
        ],
        "patterns": [
            "Patterns are about **loops + printing**. Print line by line.",
            "Identify rows/cols. Outer loop controls rows, inner loop prints characters.",
            "Simulate row 1..n and " + r"print `\n` per row."
        ],
        "strings": [
            "Use `string` / `getline` as required. Read input carefully (line vs token).",
            "If processing characters: loop through string and apply conditions.",
            "Remember: `getline(cin, s)` reads spaces; `cin >> s` does not."
        ],
        "math_logic": [
            "Use arithmetic ops and/or loops based on the task (digits, primes, gcd, etc.).",
            "Write the formula first, then implement carefully. Watch for integer division.",
            "Test with small numbers and edge cases."
        ],
        "recursion": [
            "This requires a **recursive function** (it calls itself).",
            "Define a clear base case that stops recursion. Then reduce problem size each call.",
            "Template: `f(n) = ... f(n-1)` with base case like `n==0`/`n==1`."
        ],
        "stl_intro": [
            "Use STL containers/algorithms (vector, set, map, sort, etc.) as required.",
            "Prefer `vector` + `sort` for ordering tasks. For frequency, consider `map`/`unordered_map` if allowed.",
            "Keep it standard: include headers and use `std::` properly."
        ],
    }

    return {fam: by_family.get(fam, ["Read the problem carefully.", "Follow the input/output format exactly.", "Test using the sample I/O."])}

def tutor_hints(problem: dict) -> List[str]:
    bank = _hint_bank(problem)
    return next(iter(bank.values()))

# =========================
# SKILL ENFORCEMENT (heuristics)
# =========================
def _strip_cpp_comments_and_strings(code: str) -> str:
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)
    return code

def _has_user_defined_function(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    fn_pat = re.compile(
        r"(?mx)^\s*(?:template\s*<[^>]+>\s*)?"
        r"(?:static\s+|inline\s+|constexpr\s+|friend\s+)?"
        r"(?:[\w:\<\>\,\&\*\s]+?)\s+"
        r"([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{"
    )
    names = [m.group(1) for m in fn_pat.finditer(code)]
    return any(n != "main" for n in names)

def _has_array_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    has_vector = bool(re.search(r"\b(?:std::)?vector\s*<", code))
    has_array = bool(re.search(r"\b(?:std::)?array\s*<", code))
    has_c_array_decl = bool(re.search(
        r"\b(?:bool|char|short|int|long|long\s+long|float|double|string|std::string)\s+\w+\s*\[\s*\w*\s*\]\s*;",
        code
    ))
    has_index = bool(re.search(r"\[[^\]]+\]", code))
    return (has_vector or has_array or has_c_array_decl) and has_index

def _has_nested_loops(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    pat = re.compile(r"(for|while)\s*\([^\)]*\)\s*\{[\s\S]{0,800}?(for|while)\s*\(", re.MULTILINE)
    if pat.search(code):
        return True
    return len(re.findall(r"\bfor\b|\bwhile\b", code)) >= 2

def _has_bool_logic(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\bbool\b|\btrue\b|\bfalse\b|==|!=|<=|>=|<|>", code))

def _has_string_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\b(?:std::)?string\b|getline\s*\(", code))

def _has_pattern_printing(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    has_loop = bool(re.search(r"\bfor\b|\bwhile\b", code))
    uses_cout = "cout" in code
    has_star_or_hash = ("*" in code) or ("#" in code)
    return has_loop and uses_cout and (has_star_or_hash or _has_nested_loops(code))

def _has_math_logic_ops(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"%|\bsqrt\s*\(|\babs\s*\(|/|\*|\+|-", code))

def _has_recursion(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    fn_pat = re.compile(r"(?mx)^\s*(?:[\w:\<\>\,\&\*\s]+?)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{")
    names = [m.group(1) for m in fn_pat.finditer(code)]
    names = [n for n in names if n != "main"]
    for n in names:
        if len(re.findall(rf"\b{re.escape(n)}\s*\(", code)) >= 2:
            return True
    return False

def _has_stl_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\b(?:std::)?vector\s*<|\b(?:std::)?set\s*<|\b(?:std::)?map\s*<|\bunordered_map\s*<|\b(?:std::)?sort\s*\(", code))

def enforce_skill(problem: dict, code: str) -> Tuple[bool, str, float]:
    fam = str(problem.get("family", "")).lower()

    if fam == "arrays_basic":
        if not _has_array_usage(code):
            return False, "❌ **ARRAYS (basic)**: I didn’t detect array/vector + indexing like `a[i]`.", 0.85
    elif fam == "arrays_nested":
        if not _has_array_usage(code):
            return False, "❌ **ARRAYS (nested)**: I didn’t detect array/vector + indexing like `a[i]`.", 0.85
        if not _has_nested_loops(code):
            return False, "❌ **ARRAYS (nested)**: I didn’t detect **nested loops** (e.g., `for` inside `for`).", 0.8
        if re.search(r"\bmap\b|\bunordered_map\b", _strip_cpp_comments_and_strings(code)):
            return False, "❌ **ARRAYS (nested)**: `map`/`unordered_map` not allowed. Use loops as required.", 0.95
    elif fam == "bool_checks":
        if not _has_bool_logic(code):
            return False, "❌ **BOOL CHECKS**: I didn’t detect boolean logic (`bool`, comparisons, true/false).", 0.55
    elif fam == "functions":
        if not _has_user_defined_function(code):
            return False, "❌ **FUNCTIONS**: Define at least one user-defined function (besides `main`).", 0.85
    elif fam == "patterns":
        if not _has_pattern_printing(code):
            return False, "❌ **PATTERNS**: Use loops to print the pattern (typically `cout` inside loops).", 0.5
    elif fam == "strings":
        if not _has_string_usage(code):
            return False, "❌ **STRINGS**: Use `string`/`std::string` or `getline`.", 0.6
    elif fam == "math_logic":
        if not _has_math_logic_ops(code):
            return False, "❌ **MATH/LOGIC**: I didn’t detect typical math ops (`%`, arithmetic, etc.).", 0.4
    elif fam == "recursion":
        if not _has_recursion(code):
            return False, "❌ **RECURSION**: I didn’t detect a recursive function (a function calling itself).", 0.9
    elif fam == "stl_intro":
        if not _has_stl_usage(code):
            return False, "❌ **STL INTRO**: Use STL (e.g., `vector`, `set`, `map`, or `sort`).", 0.75

    return True, "", 1.0

# =========================
# SUBPROCESS + JUDGE
# =========================
async def run_subprocess(
    cmd: List[str],
    stdin_data: Optional[bytes],
    timeout_sec: int,
    cwd: Optional[str] = None
) -> Tuple[int, bytes, bytes, bool]:
    creationflags = 0
    if IS_WINDOWS:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    preexec = None
    if not IS_WINDOWS:
        def _limit_resources() -> None:
            try:
                import resource
                mem = MAX_RUN_MEMORY_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                resource.setrlimit(resource.RLIMIT_CPU, (MAX_RUN_CPU_SEC, MAX_RUN_CPU_SEC + 1))
                resource.setrlimit(resource.RLIMIT_NPROC, (MAX_RUN_NPROC, MAX_RUN_NPROC))
            except Exception:
                pass
        preexec = _limit_resources

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=creationflags,
        preexec_fn=preexec,
    )

    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout_sec)
        truncated = (len(out) > MAX_OUTPUT_BYTES) or (len(err) > MAX_OUTPUT_BYTES)
        return proc.returncode, out[:MAX_OUTPUT_BYTES], err[:MAX_OUTPUT_BYTES], truncated
    except asyncio.TimeoutError:
        try:
            if IS_WINDOWS:
                killp = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await killp.communicate()
            else:
                proc.kill()
        except Exception:
            pass
        return -999, b"", b"TIMEOUT", False

async def compile_cpp(code: str, workdir: str) -> Tuple[bool, str]:
    src = os.path.join(workdir, "main.cpp")
    exe = exe_path(workdir)

    with open(src, "w", encoding="utf-8") as f:
        f.write(code + "\n")

    cmd = [GPP, "-std=c++17", src, "-O2", "-pipe", "-o", exe]

    logging.info("JUDGE: compiler=%s", GPP)
    logging.info("JUDGE: compiling: %s", " ".join(cmd))

    rc, out, err, _ = await run_subprocess(cmd, stdin_data=None, timeout_sec=COMPILE_TIMEOUT_SEC, cwd=workdir)

    if rc == 0 and os.path.exists(exe):
        return True, ""

    if rc == -999:
        return False, f"{ERR_COMPILE_TIMEOUT}: Compilation timed out."

    msg = (out.decode("utf-8", errors="replace") + "\n" + err.decode("utf-8", errors="replace")).strip()
    return False, (f"{ERR_COMPILE_FAIL}: " + msg) if msg else f"{ERR_COMPILE_FAIL}: Compilation failed (no output). Check that GPP points to g++."

async def run_one_test(workdir: str, t: Dict[str, Any]) -> Tuple[bool, str, Optional[dict]]:
    exe = exe_path(workdir)
    inp = t["inp"].encode("utf-8")
    expected = normalize_output(t["out"])

    rc, out, err, truncated = await run_subprocess([exe], stdin_data=inp, timeout_sec=RUN_TIMEOUT_SEC, cwd=workdir)

    if rc == -999:
        return False, f"{ERR_RUN_TIMEOUT}: Time Limit Exceeded", {"test": t, "kind": "timeout"}

    if rc != 0:
        msg = err.decode("utf-8", errors="replace").strip()
        if msg == "TIMEOUT":
            msg = "Time Limit Exceeded"
        if not msg:
            msg = "Runtime Error"
        return False, f"{ERR_RUNTIME}: Runtime Error (exit {rc})\n{msg}", {"test": t, "kind": "runtime"}

    if truncated:
        return False, f"{ERR_OUTPUT_LIMIT}: Output Limit Exceeded", {"test": t, "kind": "output_limit"}

    got = normalize_output(out.decode("utf-8", errors="replace"))
    if got != expected:
        return False, f"{ERR_WRONG_ANSWER}: Wrong Answer", {"test": t, "expected": expected, "got": got, "kind": "wa"}

    return True, "OK", None

# =========================
# DAILY LOOP
# =========================
@tasks.loop(time=POST_TIME)
async def post_daily_problem():
    if DAILY_CHANNEL_ID == 0:
        logging.warning("DAILY_CHANNEL_ID not set.")
        return

    channel = client.get_channel(DAILY_CHANNEL_ID)
    if channel is None or not isinstance(channel, discord.abc.Messageable):
        logging.warning("Daily channel not found. Check DAILY_CHANNEL_ID and permissions.")
        return

    state = load_state()
    date_str = today_str_ph()

    if state.get("last_posted_date") == date_str:
        logging.info("Already posted today. Skipping.")
        return

    day_index = int(state.get("day_index", 0))
    problem = generate_problem(day_index, date_str)

    await channel.send("⚙️ **DAILY MP DROP:** Solve it in C++ and submit with `!submit`.", embed=build_embed(problem))

    pb = state.get("problems_by_date", {})
    pb[date_str] = problem
    state["problems_by_date"] = pb
    state["day_index"] = day_index + 1
    state["last_posted_date"] = date_str
    save_state(state)

    logging.info("Posted MP for %s (day_index=%s).", date_str, day_index)

@post_daily_problem.before_loop
async def before_post_daily():
    await client.wait_until_ready()
    logging.info("CS1 Judge armed.")

# =========================
# STARTUP CHECKS
# =========================
def _die(msg: str) -> None:
    raise RuntimeError(f"{ERR_CONFIG}: {msg}")

def validate_config() -> None:
    if not TOKEN:
        _die("DISCORD_TOKEN env var missing. Set DISCORD_TOKEN before running.")
    if DAILY_CHANNEL_ID == 0:
        _die("DAILY_CHANNEL_ID env var missing. Set DAILY_CHANNEL_ID before running.")
    if COMPILE_TIMEOUT_SEC <= 0 or RUN_TIMEOUT_SEC <= 0:
        _die("Timeout values must be positive integers.")
    if MAX_CODE_BYTES <= 0:
        _die("MAX_CODE_BYTES must be > 0.")
    if not (shutil.which(GPP) or os.path.exists(GPP)):
        logging.warning("%s: GPP binary not found on startup: %s", ERR_CONFIG, GPP)

# =========================
# HELP TEXT
# =========================
HELP_TEXT = (
    "**📌 CS1 Daily MP Bot — Commands**\n\n"
    "**Student**\n"
    "• `!help` — show this menu\n"
    "• `!ping` — bot check\n"
    "• `!today` — show today’s problem\n"
    "• `!submit` — submit your C++ solution\n"
    "• `!rules` — submission format\n"
    "• `!format` — same as rules\n"
    "• `!explain` — topic explanation\n"
    "• `!approach` — steps for today’s MP\n"
    "• `!hint`, `!hint2`, `!hint3` — progressive hints (daily limit)\n"
    "• `!dryrun` — show sample I/O\n"
    "• `!constraints` — show constraints\n"
    "• `!leaderboard` — weekly leaderboard\n\n"
    "**Admin**\n"
    "• `!status` — bot status + metrics\n"
    "• `!postnow` — post today’s MP now\n"
    "• `!reset_today` — reset today’s MP\n"
    "• `!regen_today` — regenerate today’s MP\n"
    "• `!repost_date YYYY-MM-DD` — repost stored MP\n\n"
    "**Dev (Admin only)**\n"
    "• `!dev help`\n"
    "• `!dev list`\n"
    "• `!dev random [family]`\n"
    "• `!dev pick <family> <kind>`\n"
    "• `!dev setup`\n"
)

# =========================
# SLASH COMMANDS — STUDENT
# =========================
@tree.command(name="help", description="Show all bot commands")
async def help_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(HELP_TEXT, ephemeral=True)

@tree.command(name="ping", description="Bot check")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong", ephemeral=True)

@tree.command(name="today", description="Show today's problem")
async def today(interaction: discord.Interaction):
    st = load_state()
    date_str = today_str_ph()
    p = st.get("problems_by_date", {}).get(date_str)
    if not p:
        await interaction.response.send_message("❌ No problem stored for today yet. Ask admin to `/postnow` or wait for schedule.", ephemeral=True)
        return
    await interaction.response.send_message(embed=build_embed(p))

@tree.command(name="rules", description="Show how to submit")
async def rules(interaction: discord.Interaction):
    await interaction.response.send_message(
        "**How to Submit (C++ only)**\n"
        "Use `!submit` and either:\n"
        "1) Paste your full code in the `code` field (you may include a ```cpp``` block), OR\n"
        "2) Attach a `.cpp` file.\n\n"
        "**No prompts** like `Enter n:` — output must match exactly.",
        ephemeral=True
    )

@tree.command(name="format", description="Alias for /rules")
async def format_cmd(interaction: discord.Interaction):
    await rules(interaction)

@tree.command(name="explain", description="Explain today's topic")
async def explain(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    fam = str(p.get("family", "")).lower()
    if fam.startswith("arrays"):
        msg = "🧠 **Arrays**: store input values in an array/vector and process using indexing like `a[i]`."
    elif fam == "functions":
        msg = "🧠 **Functions**: define at least one user-defined function (besides `main`) and call it from `main()`."
    elif fam == "recursion":
        msg = "🧠 **Recursion**: create a function that calls itself with a smaller input, with a clear base case."
    elif fam == "strings":
        msg = "🧠 **Strings**: use `string` / `getline` and handle spaces vs tokens carefully."
    elif fam == "patterns":
        msg = "🧠 **Patterns**: use loops to print line-by-line; outer loop = rows, inner loop = columns."
    elif fam == "stl_intro":
        msg = "🧠 **STL**: use `vector`, `sort`, `set`, `map`, etc. as required."
    else:
        msg = f"🧠 Topic: **{fam}**. Follow the input/output format and apply the required concept."
    await interaction.response.send_message(msg, ephemeral=True)

@tree.command(name="approach", description="Suggested steps for today's MP")
async def approach(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    fam = str(p.get("family", "")).lower()
    steps = ["1) Read input exactly as specified."]
    if fam == "arrays_basic":
        steps += ["2) Store the n values in `vector<long long> a(n)`.",
                  "3) Loop through `a[i]` to compute the required result.",
                  "4) Print the result only (no prompts)."]
    elif fam == "arrays_nested":
        steps += ["2) Store the data (often 2D / multiple values).",
                  "3) Use nested loops (`for` inside `for`) to compute.",
                  "4) Print result only."]
    elif fam == "functions":
        steps += ["2) Write a helper function that solves the core task.",
                  "3) Call it from `main()` and print the return value."]
    elif fam == "recursion":
        steps += ["2) Identify base case.", "3) Write recursive step reducing the problem size.",
                  "4) Call the recursive function and print the result."]
    else:
        steps += ["2) Use the required topic tools (see `/explain`).", "3) Match output format exactly."]
    await interaction.response.send_message("🧩 **Approach (today's problem)**\n" + "\n".join(steps), ephemeral=True)

@tree.command(name="hint", description="Get hint 1 (limited per day)")
async def hint(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    st = load_state()
    ok, used = consume_hint(st, interaction.user.id, today_str_ph())
    if not ok:
        await interaction.response.send_message(f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).", ephemeral=True)
        return
    save_state(st)
    await interaction.response.send_message(f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {tutor_hints(p)[0]}\n➡️ Next: implement input parsing, then test with sample.", ephemeral=True)

@tree.command(name="hint2", description="Get hint 2 (limited per day)")
async def hint2(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    st = load_state()
    ok, used = consume_hint(st, interaction.user.id, today_str_ph())
    if not ok:
        await interaction.response.send_message(f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).", ephemeral=True)
        return
    save_state(st)
    hints = tutor_hints(p)
    await interaction.response.send_message(f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {hints[1] if len(hints) > 1 else hints[0]}", ephemeral=True)

@tree.command(name="hint3", description="Get hint 3 (limited per day)")
async def hint3(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    st = load_state()
    ok, used = consume_hint(st, interaction.user.id, today_str_ph())
    if not ok:
        await interaction.response.send_message(f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).", ephemeral=True)
        return
    save_state(st)
    hints = tutor_hints(p)
    msg = hints[2] if len(hints) > 2 else hints[-1]
    if TUTOR_FULL_CODE:
        msg += "\n\n✅ *(Tutor mode)* `TUTOR_FULL_CODE=true` is enabled."
    await interaction.response.send_message(f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {msg}", ephemeral=True)

@tree.command(name="dryrun", description="Show sample input/output again")
async def dryrun(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    await interaction.response.send_message(
        "🧪 **Sample Dry Run**\n"
        f"**Sample Input**\n```text\n{p.get('sample_in','')}```"
        f"**Sample Output**\n```text\n{p.get('sample_out','')}```",
        ephemeral=True
    )

@tree.command(name="constraints", description="Show constraints for today's MP")
async def constraints(interaction: discord.Interaction):
    p = get_today_problem_from_state()
    if not p:
        await interaction.response.send_message("❌ No stored problem for today yet. Ask admin to `/postnow`.", ephemeral=True)
        return
    await interaction.response.send_message(f"📌 **Constraints**\n{p.get('constraints','-')}", ephemeral=True)

@tree.command(name="leaderboard", description="Show weekly leaderboard")
async def leaderboard(interaction: discord.Interaction):
    st = load_state()
    rows = list(st.get("scores", {}).values())
    if not rows:
        await interaction.response.send_message("No leaderboard data yet.", ephemeral=True)
        return
    rows.sort(key=lambda r: (int(r.get("weekly_accepts", 0)), int(r.get("accepted", 0))), reverse=True)
    lines = ["🏆 **Weekly Leaderboard**"]
    for i, r in enumerate(rows[:10], start=1):
        lines.append(f"{i}. **{r.get('name','user')}** — weekly `{r.get('weekly_accepts',0)}` | total `{r.get('accepted',0)}` | streak `{r.get('streak',0)}`")
    await interaction.response.send_message("\n".join(lines))

# =========================
# SLASH COMMAND — SUBMIT
# =========================
@tree.command(name="submit", description="Submit your C++ solution (paste code or attach .cpp)")
@app_commands.describe(
    code="Paste your full C++ code here (optional if you attach a file)",
    file="Attach a .cpp/.cc/.cxx/.txt file (optional if you paste code)"
)
async def submit(
    interaction: discord.Interaction,
    code: Optional[str] = None,
    file: Optional[discord.Attachment] = None,
):
    if SUBMIT_LOCK.locked():
        # acknowledge quickly
        await interaction.response.send_message("⏳ Another submission is being judged right now. Please wait a moment.", ephemeral=True)
        return

    async with SUBMIT_LOCK:
        # channel restriction
        if SUBMIT_CHANNEL_ID and interaction.channel and interaction.channel.id != SUBMIT_CHANNEL_ID:
            if SUBMIT_CHANNEL_ID != DAILY_CHANNEL_ID:
                await interaction.response.send_message(f"❌ Submit only in <#{SUBMIT_CHANNEL_ID}>.", ephemeral=True)
                return

        st = load_state()
        uid = interaction.user.id
        rem = cooldown_remaining_sec(st, uid)
        if rem > 0:
            await interaction.response.send_message(f"⏳ Cooldown: wait `{rem}s` before submitting again.", ephemeral=True)
            return

        date_str = today_str_ph()
        problem = st.get("problems_by_date", {}).get(date_str)
        if not problem:
            await interaction.response.send_message("❌ No active problem for today. Ask admin to `/postnow`.", ephemeral=True)
            return

        src, parse_err = await get_code_from_slash_inputs(code, file)
        if parse_err:
            await interaction.response.send_message(f"❌ {parse_err}", ephemeral=True)
            return
        if not src:
            await interaction.response.send_message(f"❌ {ERR_NO_CODE}: No code payload found.", ephemeral=True)
            return

        if len(src.encode("utf-8", errors="ignore")) > MAX_CODE_BYTES:
            await interaction.response.send_message(f"❌ {ERR_CODE_TOO_LARGE}: Code too large. Limit is {MAX_CODE_BYTES} bytes.", ephemeral=True)
            return

        if re.search(r'cout\s*<<\s*".*(enter|input|please)', src, flags=re.IGNORECASE):
            # non-blocking warning
            try:
                await interaction.response.send_message("⚠️ Heads up: prompts like `Enter n:` often cause Wrong Answer. Output should be answer only.", ephemeral=True)
            except Exception:
                pass

        if ENFORCE_SKILLS:
            ok_skill, skill_msg, confidence = enforce_skill(problem, src)
            if not ok_skill and confidence >= SKILL_HARD_FAIL_CONFIDENCE:
                await interaction.response.send_message(
                    skill_msg + f"\n(Confidence: {confidence:.2f}. Teacher: set `ENFORCE_SKILLS=false` to disable.)",
                    ephemeral=True
                )
                return
            if not ok_skill and confidence >= SKILL_WARN_CONFIDENCE:
                # warning only
                if not interaction.response.is_done():
                    await interaction.response.send_message(f"⚠️ Skill-check warning (confidence {confidence:.2f}): {skill_msg}", ephemeral=True)

        tests = problem.get("tests", [])
        # if we already responded with a warning above, follow-up; else initial response
        if interaction.response.is_done():
            await interaction.followup.send("🧪 Compiling...", ephemeral=True)
        else:
            await interaction.response.send_message("🧪 Compiling...", ephemeral=True)

        JUDGE_METRICS["submissions"] += 1

        with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
            ok, cerr = await compile_cpp(src, workdir)
            if not ok:
                cerr = cerr.strip()
                record_compile_error(st, uid, cerr)
                JUDGE_METRICS["compile_errors"] += 1
                set_cooldown(st, uid, COOLDOWN_AFTER_FAIL_SEC)
                score_submission(st, uid, str(interaction.user), accepted=False, date_str=date_str)
                save_state(st)

                short = cerr[:1800] + ("\n... (truncated)" if len(cerr) > 1800 else "")
                await interaction.followup.send(f"❌ {ERR_COMPILE_FAIL}: Compilation Error\n```text\n{short}\n```", ephemeral=True)
                return

            await interaction.followup.send(f"✅ Compiled. Running tests (0/{len(tests)})...", ephemeral=True)

            for i, t in enumerate(tests, start=1):
                passed, verdict, details = await run_one_test(workdir, t)
                if not passed:
                    kind = (details or {}).get("kind")
                    if kind == "wa":
                        JUDGE_METRICS["wrong_answers"] += 1
                    elif kind == "runtime":
                        JUDGE_METRICS["runtime_errors"] += 1
                    elif kind == "timeout":
                        JUDGE_METRICS["timeouts"] += 1
                    elif kind == "output_limit":
                        JUDGE_METRICS["output_limit_exceeded"] += 1

                    set_cooldown(st, uid, COOLDOWN_AFTER_FAIL_SEC)
                    score_submission(st, uid, str(interaction.user), accepted=False, date_str=date_str)
                    save_state(st)

                    await interaction.followup.send(f"❌ {verdict} — failed test #{i}", ephemeral=True)
                    tinp = details["test"]["inp"] if details and "test" in details else ""

                    if details and (details.get("kind") == "wa"):
                        exp = details["expected"]
                        got = details["got"]
                        line_no, e_line, g_line = first_mismatch_line(exp, got)
                        msg_out = (
                            f"**Input**\n```text\n{clamp_block(tinp, 900)}```"
                            f"**Expected**\n```text\n{clamp_block(exp, 900)}```"
                            f"**Got**\n```text\n{clamp_block(got, 900)}```"
                        )
                        if line_no:
                            msg_out += f"\n🔎 First mismatch at **line {line_no}**:\n- expected: `{e_line}`\n- got: `{g_line}`"
                        await interaction.followup.send(msg_out, ephemeral=True)
                    else:
                        if tinp:
                            await interaction.followup.send(f"**Input**\n```text\n{clamp_block(tinp, 1200)}```", ephemeral=True)
                    return

            JUDGE_METRICS["accepted"] += 1
            set_cooldown(st, uid, COOLDOWN_AFTER_ACCEPT_SEC)
            score_submission(st, uid, str(interaction.user), accepted=True, date_str=date_str)
            save_state(st)

            await interaction.followup.send(f"✅ Accepted — {len(tests)}/{len(tests)} tests passed.\nProblem: **{problem['title']}** (Day {problem['day']})", ephemeral=True)

# =========================
# SLASH COMMANDS — ADMIN
# =========================
def admin_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            return False
        return is_admin_member(interaction.user)
    return app_commands.check(predicate)

@tree.command(name="status", description="Admin: show bot status/metrics")
@admin_only()
async def status(interaction: discord.Interaction):
    st = load_state()
    date_str = today_str_ph()
    stored = date_str in st.get("problems_by_date", {})
    di = int(st.get("day_index", 0))
    up = int(time.monotonic() - BOT_START_MONO)
    nxt = next_post_time_ph()

    await interaction.response.send_message(
        "**Bot Status**\n"
        f"- Uptime: `{up}s`\n"
        f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
        f"- Day index: `{di}`\n"
        f"- Today stored: `{stored}` ({date_str})\n"
        f"- Next scheduled post (PH): `{nxt.isoformat()}`\n"
        f"- Queue busy: `{SUBMIT_LOCK.locked()}`\n"
        f"- DAILY_CHANNEL_ID: `{DAILY_CHANNEL_ID}`\n"
        f"- SUBMIT_CHANNEL_ID: `{SUBMIT_CHANNEL_ID}`\n"
        f"- GPP: `{GPP}` exists=`{bool(shutil.which(GPP) or os.path.exists(GPP))}`\n"
        f"- Metrics: `{JUDGE_METRICS}`\n"
        f"- Updates build: `{BOT_UPDATES_VERSION}`\n",
        ephemeral=True
    )

@tree.command(name="postnow", description="Admin: post today's problem now")
@admin_only()
async def postnow(interaction: discord.Interaction):
    if DAILY_CHANNEL_ID == 0:
        await interaction.response.send_message("❌ DAILY_CHANNEL_ID not set.", ephemeral=True)
        return
    ch = client.get_channel(DAILY_CHANNEL_ID)
    if ch is None or not isinstance(ch, discord.abc.Messageable):
        await interaction.response.send_message("❌ Daily channel not found. Check DAILY_CHANNEL_ID.", ephemeral=True)
        return

    st = load_state()
    date_str = today_str_ph()
    existing = st.get("problems_by_date", {}).get(date_str)
    if existing:
        await ch.send("⚙️ **DAILY MP DROP (repost):**", embed=build_embed(existing))
        await interaction.response.send_message("✅ Reposted the already-stored problem for today.", ephemeral=True)
        return

    di = int(st.get("day_index", 0))
    p = generate_problem(di, date_str)

    await ch.send("⚙️ **DAILY MP DROP (manual):** Solve it in C++ and submit with `!submit`.", embed=build_embed(p))

    pb = st.get("problems_by_date", {})
    pb[date_str] = p
    st["problems_by_date"] = pb
    st["day_index"] = di + 1
    st["last_posted_date"] = date_str
    save_state(st)

    await interaction.response.send_message(f"✅ Posted today’s problem to <#{DAILY_CHANNEL_ID}> (day_index was {di}).", ephemeral=True)

@tree.command(name="reset_today", description="Admin: reset today's stored problem")
@admin_only()
async def reset_today(interaction: discord.Interaction):
    st = load_state()
    date_str = today_str_ph()
    pb = st.get("problems_by_date", {})

    if date_str not in pb:
        await interaction.response.send_message("✅ Nothing to reset for today (no stored problem).", ephemeral=True)
        return

    pb.pop(date_str, None)
    st["problems_by_date"] = pb
    st["last_posted_date"] = None
    save_state(st)
    await interaction.response.send_message("✅ Reset done. Use `/postnow` to post a new problem for today.", ephemeral=True)

@tree.command(name="regen_today", description="Admin: regenerate today's problem (new seed)")
@admin_only()
async def regen_today(interaction: discord.Interaction):
    st = load_state()
    date_str = today_str_ph()
    di = int(st.get("day_index", 0))
    p = generate_problem(di, date_str)
    st.setdefault("problems_by_date", {})[date_str] = p
    st["day_index"] = di + 1
    st["last_posted_date"] = date_str
    append_audit(st, {"action": "regen_today", "by": str(interaction.user), "day_index": di, "seed": p.get("seed")})
    save_state(st)

    ch = client.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
    if ch and isinstance(ch, discord.abc.Messageable):
        await ch.send("⚙️ **DAILY MP DROP (regenerated):**", embed=build_embed(p))

    await interaction.response.send_message(f"✅ Regenerated today's problem (seed={p.get('seed')}).", ephemeral=True)

@tree.command(name="repost_date", description="Admin: repost stored problem for a date (YYYY-MM-DD)")
@admin_only()
@app_commands.describe(date="Date like 2026-02-13")
async def repost_date(interaction: discord.Interaction, date: str):
    st = load_state()
    p = st.get("problems_by_date", {}).get(date)
    if not p:
        await interaction.response.send_message("❌ No stored problem for that date.", ephemeral=True)
        return
    ch = client.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
    if ch is None or not isinstance(ch, discord.abc.Messageable):
        await interaction.response.send_message("❌ Daily channel not found.", ephemeral=True)
        return
    await ch.send(f"⚙️ **DAILY MP DROP (backfill {date}):**", embed=build_embed(p))
    append_audit(st, {"action": "repost_date", "by": str(interaction.user), "date": date})
    save_state(st)
    await interaction.response.send_message("✅ Reposted.", ephemeral=True)

# =========================
# SLASH COMMANDS — DEV GROUP (ADMIN)
# =========================
dev = app_commands.Group(name="dev", description="Admin: dev tools")

@dev.command(name="help", description="Show dev commands")
@admin_only()
async def dev_help(interaction: discord.Interaction):
    await interaction.response.send_message(
        "**DEV COMMANDS:**\n"
        "`/dev list` → list all families/kinds\n"
        "`/dev random [family]` → pick random problem\n"
        "`/dev pick <family> <kind>` → pick specific problem\n"
        "`/dev setup` → show runtime config\n",
        ephemeral=True
    )

@dev.command(name="setup", description="Show runtime config")
@admin_only()
async def dev_setup(interaction: discord.Interaction):
    ch = client.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
    await interaction.response.send_message(
        f"✅ Bot online.\n"
        f"- Daily channel ID: `{DAILY_CHANNEL_ID}` (name: `{getattr(ch, 'name', None)}`)\n"
        f"- Submit channel ID: `{SUBMIT_CHANNEL_ID}` (0 means any channel)\n"
        f"- Post time: `{POST_TIME}` (PH time)\n"
        f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
        f"- Windows mode: `{IS_WINDOWS}`\n"
        f"- GPP: `{GPP}`\n"
        f"- ADMIN_ROLES: `{ADMIN_ROLES}`\n",
        ephemeral=True
    )

@dev.command(name="list", description="List all families and kinds")
@admin_only()
async def dev_list(interaction: discord.Interaction):
    msg = "**FAMILIES AND KINDS:**\n"
    total_count = 0
    for f, kinds in family_kinds.items():
        msg += f"- **{f}**: {', '.join(f'`{k}`' for k in kinds)}\n"
        total_count += len(kinds)
    msg += f"\n**Total:** {total_count}"
    await interaction.response.send_message(msg, ephemeral=True)

@dev.command(name="random", description="Pick a random problem (optionally choose a family)")
@admin_only()
@app_commands.describe(family="Optional family like arrays_basic")
async def dev_random(interaction: discord.Interaction, family: Optional[str] = None):
    st = load_state()
    date_str = today_str_ph()

    if family is None:
        family = random.choice(list(family_kinds.keys()))
    else:
        family = family.lower().strip()
        if family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await interaction.response.send_message(f"❌ Invalid family '{family}'. Available:\n{families_list}", ephemeral=True)
            return

    kind = random.choice(family_kinds[family])
    day_index = 0
    seed = stable_seed_for_day(day_index, date_str)
    rng = random.Random(seed)

    try:
        if family == "arrays_basic":
            p = gen_arrays_basic(rng, kind=kind)
        elif family == "arrays_nested":
            p = gen_arrays_nested(rng, kind=kind)
        elif family == "bool_checks":
            p = gen_bool_checks(rng, kind=kind)
        elif family == "functions":
            p = gen_functions(rng, kind=kind)
        elif family == "patterns":
            p = gen_patterns(rng, kind=kind)
        elif family == "strings":
            p = gen_strings(rng, kind=kind)
        elif family == "math_logic":
            p = gen_math_logic(rng, kind=kind)
        elif family == "recursion":
            p = gen_recursion(rng, kind=kind)
        elif family == "stl_intro":
            p = gen_stl_intro(rng, kind=kind)
        else:
            await interaction.response.send_message("❌ Unexpected error: family not recognized.", ephemeral=True)
            return
    except ValueError as e:
        await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        return

    p["day"] = date_str
    p["seed"] = seed
    p["day_index"] = day_index

    st.setdefault("problems_by_date", {})[date_str] = p
    st["last_posted_date"] = date_str
    st["day_index"] = day_index + 1
    save_state(st)

    await interaction.response.send_message(f"⚙️ **DEV PICK RANDOM:** {family} • {kind}", embed=build_embed(p))

@dev.command(name="pick", description="Pick a specific family+kind")
@admin_only()
@app_commands.describe(family="Family (e.g. arrays_basic)", kind="Kind (from /dev list)")
async def dev_pick(interaction: discord.Interaction, family: str, kind: str):
    st = load_state()
    date_str = today_str_ph()
    family = family.lower().strip()
    kind = kind.strip()

    if family not in family_kinds:
        families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
        await interaction.response.send_message(f"❌ Invalid family. Available:\n{families_list}", ephemeral=True)
        return

    if kind not in family_kinds[family]:
        kinds_list = ", ".join(f"`{k}`" for k in family_kinds[family])
        await interaction.response.send_message(f"❌ Invalid kind for `{family}`. Available:\n{kinds_list}", ephemeral=True)
        return

    day_index = 0
    seed = stable_seed_for_day(day_index, date_str)
    rng = random.Random(seed)

    try:
        if family == "arrays_basic":
            p = gen_arrays_basic(rng, kind=kind)
        elif family == "arrays_nested":
            p = gen_arrays_nested(rng, kind=kind)
        elif family == "bool_checks":
            p = gen_bool_checks(rng, kind=kind)
        elif family == "functions":
            p = gen_functions(rng, kind=kind)
        elif family == "patterns":
            p = gen_patterns(rng, kind=kind)
        elif family == "strings":
            p = gen_strings(rng, kind=kind)
        elif family == "math_logic":
            p = gen_math_logic(rng, kind=kind)
        elif family == "recursion":
            p = gen_recursion(rng, kind=kind)
        elif family == "stl_intro":
            p = gen_stl_intro(rng, kind=kind)
        else:
            await interaction.response.send_message("❌ Unexpected error: family not recognized.", ephemeral=True)
            return
    except ValueError as e:
        await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        return

    p["day"] = date_str
    p["seed"] = seed
    p["day_index"] = day_index

    st.setdefault("problems_by_date", {})[date_str] = p
    st["last_posted_date"] = date_str
    st["day_index"] = day_index + 1
    save_state(st)

    await interaction.response.send_message(f"⚙️ **DEV PICK:** {family} • {kind}", embed=build_embed(p))

tree.add_command(dev)



async def handle_prefix_submit(message: discord.Message, raw_payload: str) -> None:
    if SUBMIT_LOCK.locked():
        await message.channel.send("⏳ Another submission is being judged right now. Please wait a moment.")
        return

    async with SUBMIT_LOCK:
        if SUBMIT_CHANNEL_ID and message.channel and message.channel.id != SUBMIT_CHANNEL_ID:
            if SUBMIT_CHANNEL_ID != DAILY_CHANNEL_ID:
                await message.channel.send(f"❌ Submit only in <#{SUBMIT_CHANNEL_ID}>.")
                return

        st = load_state()
        uid = message.author.id
        rem = cooldown_remaining_sec(st, uid)
        if rem > 0:
            await message.channel.send(f"⏳ Cooldown: wait `{rem}s` before submitting again.")
            return

        date_str = today_str_ph()
        problem = st.get("problems_by_date", {}).get(date_str)
        if not problem:
            await message.channel.send("❌ No active problem for today. Ask admin to `!postnow`.")
            return

        src, parse_err = await get_code_from_message(message, raw_payload)
        if parse_err:
            await message.channel.send(f"❌ {parse_err}")
            return
        if not src:
            await message.channel.send(f"❌ {ERR_NO_CODE}: No code payload found.")
            return

        if len(src.encode("utf-8", errors="ignore")) > MAX_CODE_BYTES:
            await message.channel.send(f"❌ {ERR_CODE_TOO_LARGE}: Code too large. Limit is {MAX_CODE_BYTES} bytes.")
            return

        if ENFORCE_SKILLS:
            ok_skill, skill_msg, confidence = enforce_skill(problem, src)
            if not ok_skill and confidence >= SKILL_HARD_FAIL_CONFIDENCE:
                await message.channel.send(skill_msg + f"\n(Confidence: {confidence:.2f}. Teacher: set `ENFORCE_SKILLS=false` to disable.)")
                return
            if not ok_skill and confidence >= SKILL_WARN_CONFIDENCE:
                await message.channel.send(f"⚠️ Skill-check warning (confidence {confidence:.2f}): {skill_msg}")

        tests = problem.get("tests", [])
        await message.channel.send("🧪 Compiling...")
        JUDGE_METRICS["submissions"] += 1

        with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
            ok, cerr = await compile_cpp(src, workdir)
            if not ok:
                cerr = cerr.strip()
                record_compile_error(st, uid, cerr)
                JUDGE_METRICS["compile_errors"] += 1
                set_cooldown(st, uid, COOLDOWN_AFTER_FAIL_SEC)
                score_submission(st, uid, str(message.author), accepted=False, date_str=date_str)
                save_state(st)
                short = cerr[:1800] + ("\n... (truncated)" if len(cerr) > 1800 else "")
                await message.channel.send(f"❌ {ERR_COMPILE_FAIL}: Compilation Error\n```text\n{short}\n```")
                return

            await message.channel.send(f"✅ Compiled. Running tests (0/{len(tests)})...")
            for i, t in enumerate(tests, start=1):
                passed, verdict, details = await run_one_test(workdir, t)
                if not passed:
                    kind = (details or {}).get("kind")
                    if kind == "wa":
                        JUDGE_METRICS["wrong_answers"] += 1
                    elif kind == "runtime":
                        JUDGE_METRICS["runtime_errors"] += 1
                    elif kind == "timeout":
                        JUDGE_METRICS["timeouts"] += 1
                    elif kind == "output_limit":
                        JUDGE_METRICS["output_limit_exceeded"] += 1

                    set_cooldown(st, uid, COOLDOWN_AFTER_FAIL_SEC)
                    score_submission(st, uid, str(message.author), accepted=False, date_str=date_str)
                    save_state(st)

                    await message.channel.send(f"❌ {verdict} — failed test #{i}")
                    tinp = details["test"]["inp"] if details and "test" in details else ""
                    if details and details.get("kind") == "wa":
                        exp = details["expected"]
                        got = details["got"]
                        line_no, e_line, g_line = first_mismatch_line(exp, got)
                        msg_out = (
                            f"**Input**\n```text\n{clamp_block(tinp, 900)}```"
                            f"**Expected**\n```text\n{clamp_block(exp, 900)}```"
                            f"**Got**\n```text\n{clamp_block(got, 900)}```"
                        )
                        if line_no:
                            msg_out += f"\n🔎 First mismatch at **line {line_no}**:\n- expected: `{e_line}`\n- got: `{g_line}`"
                        await message.channel.send(msg_out)
                    elif tinp:
                        await message.channel.send(f"**Input**\n```text\n{clamp_block(tinp, 1200)}```")
                    return

            JUDGE_METRICS["accepted"] += 1
            set_cooldown(st, uid, COOLDOWN_AFTER_ACCEPT_SEC)
            score_submission(st, uid, str(message.author), accepted=True, date_str=date_str)
            save_state(st)
            await message.channel.send(f"✅ Accepted — {len(tests)}/{len(tests)} tests passed.\nProblem: **{problem['title']}** (Day {problem['day']})")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    content = (message.content or "").strip()
    if not content.startswith(COMMAND_PREFIX):
        return

    cmdline = content[len(COMMAND_PREFIX):].strip()
    if not cmdline:
        return
    cmd, _, rest = cmdline.partition(" ")
    cmd = cmd.lower()

    if cmd == "help":
        await message.channel.send(HELP_TEXT)
    elif cmd == "ping":
        await message.channel.send("pong")
    elif cmd == "today":
        st = load_state()
        p = st.get("problems_by_date", {}).get(today_str_ph())
        if not p:
            await message.channel.send("❌ No problem stored for today yet. Ask admin to `!postnow` or wait for schedule.")
            return
        await message.channel.send(embed=build_embed(p))
    elif cmd in {"rules", "format"}:
        await message.channel.send(
            "**How to Submit (C++ only)**\n"
            "Use `!submit` and either:\n"
            "1) Paste your full code after the command (you may include a ```cpp``` block), OR\n"
            "2) Attach a `.cpp` file.\n\n"
            "**No prompts** like `Enter n:` — output must match exactly."
        )
    elif cmd == "submit":
        await handle_prefix_submit(message, rest)

# =========================
# EVENT: READY + COMMAND SYNC
# =========================
@client.event
async def on_ready():
    logging.info("Logged in as %s (id: %s)", client.user, client.user.id)

    logging.info("Config: DAILY_CHANNEL_ID=%s SUBMIT_CHANNEL_ID=%s ENFORCE_SKILLS=%s ADMIN_ROLES=%s",
                 DAILY_CHANNEL_ID, SUBMIT_CHANNEL_ID, ENFORCE_SKILLS, ADMIN_ROLES)
    logging.info("Config: COMPILE_TIMEOUT=%ss RUN_TIMEOUT=%ss MAX_CODE_BYTES=%s COOLDOWN=%ss",
                 COMPILE_TIMEOUT_SEC, RUN_TIMEOUT_SEC, MAX_CODE_BYTES, SUBMIT_COOLDOWN_SEC)

    # Prefix-only mode: clear slash commands so only !commands are recognized.
    try:
        tree.clear_commands(guild=None)
        await tree.sync()
        logging.info("Cleared slash commands; prefix-only mode enabled.")
    except Exception as e:
        logging.exception("Failed to clear slash commands: %s", e)

    if not post_daily_problem.is_running():
        post_daily_problem.start()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    validate_config()
    client.run(TOKEN)