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
from typing import List, Optional, Tuple, Dict, Any, Callable, Awaitable

import discord
from discord import app_commands
from discord.ext import tasks, commands
from discord.ext.commands import CheckFailure, CommandNotFound

# ========= Your generators =========
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

ENABLE_SLASH = os.getenv("ENABLE_SLASH", "true").strip().lower() in ("1", "true", "yes", "y", "on")
ENABLE_PREFIX = os.getenv("ENABLE_PREFIX", "true").strip().lower() in ("1", "true", "yes", "y", "on")
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!").strip() or "!"

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

# Error codes
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

SKILL_HARD_FAIL_CONFIDENCE = float(os.getenv("SKILL_HARD_FAIL_CONFIDENCE", "0.75"))
SKILL_WARN_CONFIDENCE = float(os.getenv("SKILL_WARN_CONFIDENCE", "0.45"))
HINTS_PER_DAY_LIMIT = int(os.getenv("HINTS_PER_DAY_LIMIT", "5"))

BOT_UPDATES_VERSION = "2026-02-command-system-rewrite"

# =========================
# DISCORD BOT SETUP
# =========================
intents = discord.Intents.default()
# Prefix commands need message_content to see messages
intents.message_content = True

# IMPORTANT: callable prefix prevents weird None behavior and makes logging/debug easier
def _prefix_callable(_bot: commands.Bot, _message: discord.Message):
    return COMMAND_PREFIX if ENABLE_PREFIX else "NO_PREFIX__"

bot = commands.Bot(
    command_prefix=_prefix_callable,
    intents=intents,
    help_command=None,
)

# =========================
# PERMISSIONS / ADMIN
# =========================
def is_admin_member(member: discord.Member) -> bool:
    try:
        if member.guild_permissions.administrator:
            return True
        names = {r.name for r in member.roles}
        return any(ar in names for ar in ADMIN_ROLES)
    except Exception:
        return False

def prefix_admin_only():
    async def predicate(ctx: commands.Context) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        return is_admin_member(ctx.author)
    return commands.check(predicate)

def slash_admin_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            return False
        return is_admin_member(interaction.user)
    return app_commands.check(predicate)

# =========================
# TIME HELPERS
# =========================
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

    how = []
    if ENABLE_PREFIX:
        how.append(f"Use `{COMMAND_PREFIX}submit` and paste your full C++ code (or attach a .cpp file).")
    if ENABLE_SLASH:
        how.append("Use `/submit` and paste your full C++ code (or attach a .cpp file).")
    how.append("No prompts like `Enter n:` — output must match exactly.")
    embed.add_field(name="How to Submit (C++ only)", value=" ".join(how), inline=False)

    embed.set_footer(text=f"Day: {problem['day']} • Seed: {problem['seed']}")
    return embed

# =========================
# CODE EXTRACTION
# =========================
CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_cpp_blocks(content: str) -> List[str]:
    return [m.group(1).strip() for m in CODE_BLOCK_RE.finditer(content or "") if m.group(1).strip()]

async def get_code_from_inputs(code_text: Optional[str], attachment: Optional[discord.Attachment]) -> Tuple[Optional[str], Optional[str]]:
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

def tutor_hints(problem: dict) -> List[str]:
    fam = str(problem.get("family", "")).lower()
    by_family: Dict[str, List[str]] = {
        "arrays_basic": [
            "Read **n**, then read **n numbers**. Store them in an array/vector so you can process them using `a[i]`.",
            "Use `vector<long long> a(n); for (int i=0;i<n;i++) cin>>a[i];` then compute the required value in a loop.",
            "Double-check you print **only** what the problem asks (no extra spaces/prompts)."
        ],
        "arrays_nested": [
            "This one usually needs **nested loops** (a loop inside a loop) + an array/matrix.",
            "If it's 2D: store inputs, then use `for (i) for (j)` to compute the required value.",
            "Outer loop = rows, inner loop = columns. Avoid `map`/`unordered_map` if disallowed."
        ],
        "bool_checks": [
            "Compute the condition using comparisons (`<`, `>`, `==`, etc.).",
            "Store it in `bool ok = (condition);` then print exactly what’s required (`YES/NO` or `1/0`).",
            "Be careful with boundary cases (equal, negative, etc.)."
        ],
        "functions": [
            "Your instructor wants a **user-defined function** besides `main()`.",
            "Write a helper function that does the core work, then call it from `main()`.",
            "Keep I/O in `main()`, logic inside the function if possible."
        ],
        "patterns": [
            "Patterns are about **loops + printing**. Print line by line.",
            "Figure out how many rows; outer loop controls rows; inner loop prints characters.",
            "Print `\\n` after each row; don’t print extra lines."
        ],
        "strings": [
            "Use `string` / `getline` as required. Read input carefully (line vs token).",
            "If processing characters: loop through the string and apply conditions.",
            "Remember: `getline(cin, s)` reads spaces; `cin >> s` does not."
        ],
        "math_logic": [
            "Write the math/formula first, then implement carefully. Watch integer division.",
            "Test with small numbers and edge cases (0, 1, negatives if allowed).",
            "Use `long long` if values can get large."
        ],
        "recursion": [
            "This requires a **recursive function** (it calls itself).",
            "Define a base case that stops recursion, then reduce the input each call.",
            "Avoid infinite recursion: make sure it progresses toward the base case."
        ],
        "stl_intro": [
            "Use STL containers/algorithms (`vector`, `sort`, `set`, `map`) as required.",
            "For ordering tasks, `vector + sort` is usually enough.",
            "Make sure output formatting matches exactly."
        ],
    }
    return by_family.get(fam, ["Read the problem carefully.", "Follow the I/O format exactly.", "Test using the sample I/O."])

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

    channel = bot.get_channel(DAILY_CHANNEL_ID)
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

    await channel.send("⚙️ **DAILY MP DROP:** Solve it in C++ and submit.", embed=build_embed(problem))

    pb = state.get("problems_by_date", {})
    pb[date_str] = problem
    state["problems_by_date"] = pb
    state["day_index"] = day_index + 1
    state["last_posted_date"] = date_str
    save_state(state)

    logging.info("Posted MP for %s (day_index=%s).", date_str, day_index)

@post_daily_problem.before_loop
async def before_post_daily():
    await bot.wait_until_ready()
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
def help_text() -> str:
    parts = ["**📌 CS1 Daily MP Bot — Commands**\n"]
    if ENABLE_PREFIX:
        parts += [
            f"**Prefix (Student)**\n"
            f"• `{COMMAND_PREFIX}help` • `{COMMAND_PREFIX}ping` • `{COMMAND_PREFIX}today` • `{COMMAND_PREFIX}submit` • `{COMMAND_PREFIX}rules`\n"
            f"• `{COMMAND_PREFIX}explain` • `{COMMAND_PREFIX}approach` • `{COMMAND_PREFIX}hint` • `{COMMAND_PREFIX}hint2` • `{COMMAND_PREFIX}hint3`\n"
            f"• `{COMMAND_PREFIX}dryrun` • `{COMMAND_PREFIX}constraints` • `{COMMAND_PREFIX}leaderboard`\n"
            f"• `{COMMAND_PREFIX}whoami` • `{COMMAND_PREFIX}whotest`\n\n"
            f"**Prefix (Admin)**\n"
            f"• `{COMMAND_PREFIX}status` • `{COMMAND_PREFIX}postnow` • `{COMMAND_PREFIX}reset_today` • `{COMMAND_PREFIX}regen_today` • `{COMMAND_PREFIX}repost_date YYYY-MM-DD`\n"
            f"• `{COMMAND_PREFIX}dev help|list|random [family]|pick <family> <kind>|setup`\n"
        ]
    if ENABLE_SLASH:
        parts += [
            "\n**Slash (Student)**\n"
            "• `/help` • `/ping` • `/today` • `/submit` • `/rules` • `/format` • `/explain` • `/approach`\n"
            "• `/hint` • `/hint2` • `/hint3` • `/dryrun` • `/constraints` • `/leaderboard`\n\n"
            "**Slash (Admin)**\n"
            "• `/status` • `/postnow` • `/reset_today` • `/regen_today` • `/repost_date` • `/dev ...`\n"
        ]
    parts.append(f"\nBuild: `{BOT_UPDATES_VERSION}`")
    return "".join(parts)

# =========================
# SHARED “SEND” HELPERS
# =========================
async def send_ephemeral(interaction: discord.Interaction, content: str = "", *, embed: Optional[discord.Embed] = None) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(content, embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(content, embed=embed, ephemeral=True)

async def send_public(ctx: commands.Context, content: str = "", *, embed: Optional[discord.Embed] = None) -> None:
    await ctx.send(content, embed=embed)

# =========================
# SHARED SUBMISSION PIPELINE
# =========================
async def judge_submission(
    *,
    user_id: int,
    user_display: str,
    channel_id: int,
    code_text: Optional[str],
    attachment: Optional[discord.Attachment],
    progress_cb: Callable[[str], Awaitable[None]],
    result_cb: Callable[[str], Awaitable[None]],
) -> None:
    # channel restriction
    if SUBMIT_CHANNEL_ID and channel_id != SUBMIT_CHANNEL_ID:
        if SUBMIT_CHANNEL_ID != DAILY_CHANNEL_ID:
            await result_cb(f"❌ Submit only in <#{SUBMIT_CHANNEL_ID}>.")
            return

    if SUBMIT_LOCK.locked():
        await result_cb("⏳ Another submission is being judged right now. Please try again in a moment.")
        return

    async with SUBMIT_LOCK:
        st = load_state()

        rem = cooldown_remaining_sec(st, user_id)
        if rem > 0:
            await result_cb(f"⏳ Cooldown: wait `{rem}s` before submitting again.")
            return

        date_str = today_str_ph()
        problem = st.get("problems_by_date", {}).get(date_str)
        if not problem:
            await result_cb("❌ No active problem for today. Ask admin to post today’s MP.")
            return

        src, parse_err = await get_code_from_inputs(code_text, attachment)
        if parse_err:
            await result_cb(f"❌ {parse_err}")
            return
        if not src:
            await result_cb(f"❌ {ERR_NO_CODE}: No code payload found.")
            return

        if len(src.encode("utf-8", errors="ignore")) > MAX_CODE_BYTES:
            await result_cb(f"❌ {ERR_CODE_TOO_LARGE}: Code too large. Limit is {MAX_CODE_BYTES} bytes.")
            return

        if re.search(r'cout\s*<<\s*".*(enter|input|please)', src, flags=re.IGNORECASE):
            await progress_cb("⚠️ Heads up: prompts like `Enter n:` often cause Wrong Answer. Output should be answer only.")

        if ENFORCE_SKILLS:
            ok_skill, skill_msg, confidence = enforce_skill(problem, src)
            if not ok_skill and confidence >= SKILL_HARD_FAIL_CONFIDENCE:
                await result_cb(skill_msg + f"\n(Confidence: {confidence:.2f}. Teacher: set `ENFORCE_SKILLS=false` to disable.)")
                return
            if not ok_skill and confidence >= SKILL_WARN_CONFIDENCE:
                await progress_cb(f"⚠️ Skill-check warning (confidence {confidence:.2f}): {skill_msg}")

        tests = problem.get("tests", [])
        await progress_cb("🧪 Compiling...")
        JUDGE_METRICS["submissions"] += 1

        with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
            ok, cerr = await compile_cpp(src, workdir)
            if not ok:
                cerr = cerr.strip()
                record_compile_error(st, user_id, cerr)
                JUDGE_METRICS["compile_errors"] += 1
                set_cooldown(st, user_id, COOLDOWN_AFTER_FAIL_SEC)
                score_submission(st, user_id, user_display, accepted=False, date_str=date_str)
                save_state(st)

                short = cerr[:1800] + ("\n... (truncated)" if len(cerr) > 1800 else "")
                await result_cb(f"❌ {ERR_COMPILE_FAIL}: Compilation Error\n```text\n{short}\n```")
                return

            await progress_cb(f"✅ Compiled. Running tests (0/{len(tests)})...")

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

                    set_cooldown(st, user_id, COOLDOWN_AFTER_FAIL_SEC)
                    score_submission(st, user_id, user_display, accepted=False, date_str=date_str)
                    save_state(st)

                    msg = f"❌ {verdict} — failed test #{i}\n"
                    tinp = details["test"]["inp"] if details and "test" in details else ""
                    if details and (details.get("kind") == "wa"):
                        exp = details["expected"]
                        got = details["got"]
                        line_no, e_line, g_line = first_mismatch_line(exp, got)
                        msg += (
                            f"**Input**\n```text\n{clamp_block(tinp, 900)}```"
                            f"**Expected**\n```text\n{clamp_block(exp, 900)}```"
                            f"**Got**\n```text\n{clamp_block(got, 900)}```"
                        )
                        if line_no:
                            msg += f"\n🔎 First mismatch at **line {line_no}**:\n- expected: `{e_line}`\n- got: `{g_line}`"
                    else:
                        if tinp:
                            msg += f"**Input**\n```text\n{clamp_block(tinp, 1200)}```"
                    await result_cb(msg)
                    return

            JUDGE_METRICS["accepted"] += 1
            set_cooldown(st, user_id, COOLDOWN_AFTER_ACCEPT_SEC)
            score_submission(st, user_id, user_display, accepted=True, date_str=date_str)
            save_state(st)

            await result_cb(f"✅ Accepted — {len(tests)}/{len(tests)} tests passed.\nProblem: **{problem['title']}** (Day {problem['day']})")

# =========================
# COMMAND SYSTEM FIXES
# =========================
@bot.event
async def on_message(message: discord.Message):
    """
    ✅ If you define on_message, you MUST call bot.process_commands
    otherwise ALL prefix commands stop working.
    """
    if message.author.bot:
        return

    content = (message.content or "").strip()
    if ENABLE_PREFIX and content == f"{COMMAND_PREFIX}whotest":
        try:
            await message.channel.send("✅ I can see messages here (prefix alive).")
        except discord.Forbidden:
            logging.warning("Forbidden: cannot send in channel=%s", getattr(message.channel, "id", None))

    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx: commands.Context, error: Exception):
    if isinstance(error, CommandNotFound):
        return

    if isinstance(error, CheckFailure):
        roles = [r.name for r in ctx.author.roles] if isinstance(ctx.author, discord.Member) else []
        await ctx.send(
            "❌ You don't have permission to use that command.\n"
            f"Expected ADMIN_ROLES: `{ADMIN_ROLES}`\n"
            f"Your roles: `{roles}`"
        )
        return

    logging.exception("Command error: %s", error)
    await ctx.send(f"❌ Command error: `{type(error).__name__}`")

# =========================
# SLASH COMMANDS
# =========================
if ENABLE_SLASH:
    tree = bot.tree

    @tree.command(name="help", description="Show all bot commands")
    async def slash_help(interaction: discord.Interaction):
        await send_ephemeral(interaction, help_text())

    @tree.command(name="ping", description="Bot check")
    async def slash_ping(interaction: discord.Interaction):
        await send_ephemeral(interaction, "pong")

    @tree.command(name="today", description="Show today's problem")
    async def slash_today(interaction: discord.Interaction):
        st = load_state()
        date_str = today_str_ph()
        p = st.get("problems_by_date", {}).get(date_str)
        if not p:
            await send_ephemeral(interaction, "❌ No problem stored for today yet. Ask admin to `/postnow` or wait for schedule.")
            return
        await interaction.response.send_message(embed=build_embed(p))

    @tree.command(name="rules", description="Show how to submit")
    async def slash_rules(interaction: discord.Interaction):
        await send_ephemeral(
            interaction,
            "**How to Submit (C++ only)**\n"
            "Use `/submit` and either:\n"
            "1) Paste your full code in the `code` field (you may include a ```cpp``` block), OR\n"
            "2) Attach a `.cpp` file.\n\n"
            "**No prompts** like `Enter n:` — output must match exactly."
        )

    @tree.command(name="format", description="Alias for /rules")
    async def slash_format(interaction: discord.Interaction):
        await slash_rules(interaction)

    @tree.command(name="explain", description="Explain today's topic")
    async def slash_explain(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        fam = str(p.get("family", "")).lower()
        mapping = {
            "arrays_basic": "🧠 **Arrays**: store input values in an array/vector and process using indexing like `a[i]`.",
            "arrays_nested": "🧠 **Arrays (nested)**: store data and use nested loops (`for` inside `for`).",
            "functions": "🧠 **Functions**: define at least one user-defined function (besides `main`) and call it from `main()`.",
            "recursion": "🧠 **Recursion**: create a function that calls itself with a smaller input, with a clear base case.",
            "strings": "🧠 **Strings**: use `string` / `getline` and handle spaces vs tokens carefully.",
            "patterns": "🧠 **Patterns**: use loops to print line-by-line; outer loop = rows, inner loop = columns.",
            "stl_intro": "🧠 **STL**: use `vector`, `sort`, `set`, `map`, etc. as required.",
        }
        await send_ephemeral(interaction, mapping.get(fam, f"🧠 Topic: **{fam}**. Follow the input/output format and apply the required concept."))

    @tree.command(name="approach", description="Suggested steps for today's MP")
    async def slash_approach(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        fam = str(p.get("family", "")).lower()
        steps = ["1) Read input exactly as specified."]
        if fam == "arrays_basic":
            steps += ["2) Store the n values in `vector<long long> a(n)`.","3) Loop through `a[i]` to compute the required result.","4) Print the result only (no prompts)."]
        elif fam == "arrays_nested":
            steps += ["2) Store the data (often 2D / multiple values).","3) Use nested loops (`for` inside `for`) to compute.","4) Print result only."]
        elif fam == "functions":
            steps += ["2) Write a helper function that solves the core task.","3) Call it from `main()` and print the return value."]
        elif fam == "recursion":
            steps += ["2) Identify base case.","3) Write recursive step reducing the problem size.","4) Call the recursive function and print the result."]
        else:
            steps += ["2) Use the required topic tools (see `/explain`).", "3) Match output format exactly."]
        await send_ephemeral(interaction, "🧩 **Approach (today's problem)**\n" + "\n".join(steps))

    @tree.command(name="hint", description="Get hint 1 (limited per day)")
    async def slash_hint(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, interaction.user.id, today_str_ph())
        if not ok:
            await send_ephemeral(interaction, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        await send_ephemeral(interaction, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {tutor_hints(p)[0]}")

    @tree.command(name="hint2", description="Get hint 2 (limited per day)")
    async def slash_hint2(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, interaction.user.id, today_str_ph())
        if not ok:
            await send_ephemeral(interaction, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        hints = tutor_hints(p)
        await send_ephemeral(interaction, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {hints[1] if len(hints) > 1 else hints[0]}")

    @tree.command(name="hint3", description="Get hint 3 (limited per day)")
    async def slash_hint3(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, interaction.user.id, today_str_ph())
        if not ok:
            await send_ephemeral(interaction, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        hints = tutor_hints(p)
        msg = hints[2] if len(hints) > 2 else hints[-1]
        if TUTOR_FULL_CODE:
            msg += "\n\n✅ *(Tutor mode)* `TUTOR_FULL_CODE=true` is enabled."
        await send_ephemeral(interaction, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {msg}")

    @tree.command(name="dryrun", description="Show sample input/output again")
    async def slash_dryrun(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        await send_ephemeral(
            interaction,
            "🧪 **Sample Dry Run**\n"
            f"**Sample Input**\n```text\n{p.get('sample_in','')}```"
            f"**Sample Output**\n```text\n{p.get('sample_out','')}```"
        )

    @tree.command(name="constraints", description="Show constraints for today's MP")
    async def slash_constraints(interaction: discord.Interaction):
        p = get_today_problem_from_state()
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for today yet. Ask admin to `/postnow`.")
            return
        await send_ephemeral(interaction, f"📌 **Constraints**\n{p.get('constraints','-')}")

    @tree.command(name="leaderboard", description="Show weekly leaderboard")
    async def slash_leaderboard(interaction: discord.Interaction):
        st = load_state()
        rows = list(st.get("scores", {}).values())
        if not rows:
            await send_ephemeral(interaction, "No leaderboard data yet.")
            return
        rows.sort(key=lambda r: (int(r.get("weekly_accepts", 0)), int(r.get("accepted", 0))), reverse=True)
        lines = ["🏆 **Weekly Leaderboard**"]
        for i, r in enumerate(rows[:10], start=1):
            lines.append(f"{i}. **{r.get('name','user')}** — weekly `{r.get('weekly_accepts',0)}` | total `{r.get('accepted',0)}` | streak `{r.get('streak',0)}`")
        await interaction.response.send_message("\n".join(lines))

    @tree.command(name="submit", description="Submit your C++ solution (paste code or attach .cpp)")
    @app_commands.describe(code="Paste your full C++ code here (optional if you attach a file)",
                           file="Attach a .cpp/.cc/.cxx/.txt file (optional if you paste code)")
    async def slash_submit(interaction: discord.Interaction, code: Optional[str] = None, file: Optional[discord.Attachment] = None):
        async def progress(msg: str):
            await send_ephemeral(interaction, msg)

        async def result(msg: str):
            await send_ephemeral(interaction, msg)

        await send_ephemeral(interaction, "✅ Submission received. Starting judge...")
        channel_id = interaction.channel.id if interaction.channel else 0

        await judge_submission(
            user_id=interaction.user.id,
            user_display=str(interaction.user),
            channel_id=channel_id,
            code_text=code,
            attachment=file,
            progress_cb=progress,
            result_cb=result,
        )

    @tree.command(name="status", description="Admin: show bot status/metrics")
    @slash_admin_only()
    async def slash_status(interaction: discord.Interaction):
        st = load_state()
        date_str = today_str_ph()
        stored = date_str in st.get("problems_by_date", {})
        di = int(st.get("day_index", 0))
        up = int(time.monotonic() - BOT_START_MONO)
        nxt = next_post_time_ph()

        await send_ephemeral(
            interaction,
            "**Bot Status**\n"
            f"- Uptime: `{up}s`\n"
            f"- ENABLE_PREFIX: `{ENABLE_PREFIX}` | ENABLE_SLASH: `{ENABLE_SLASH}`\n"
            f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
            f"- Day index: `{di}`\n"
            f"- Today stored: `{stored}` ({date_str})\n"
            f"- Next scheduled post (PH): `{nxt.isoformat()}`\n"
            f"- Queue busy: `{SUBMIT_LOCK.locked()}`\n"
            f"- DAILY_CHANNEL_ID: `{DAILY_CHANNEL_ID}`\n"
            f"- SUBMIT_CHANNEL_ID: `{SUBMIT_CHANNEL_ID}`\n"
            f"- GPP: `{GPP}` exists=`{bool(shutil.which(GPP) or os.path.exists(GPP))}`\n"
            f"- Metrics: `{JUDGE_METRICS}`\n"
            f"- Build: `{BOT_UPDATES_VERSION}`\n"
        )

    @tree.command(name="postnow", description="Admin: post today's problem now")
    @slash_admin_only()
    async def slash_postnow(interaction: discord.Interaction):
        if DAILY_CHANNEL_ID == 0:
            await send_ephemeral(interaction, "❌ DAILY_CHANNEL_ID not set.")
            return
        ch = bot.get_channel(DAILY_CHANNEL_ID)
        if ch is None or not isinstance(ch, discord.abc.Messageable):
            await send_ephemeral(interaction, "❌ Daily channel not found. Check DAILY_CHANNEL_ID.")
            return

        st = load_state()
        date_str = today_str_ph()
        existing = st.get("problems_by_date", {}).get(date_str)
        if existing:
            await ch.send("⚙️ **DAILY MP DROP (repost):**", embed=build_embed(existing))
            await send_ephemeral(interaction, "✅ Reposted the already-stored problem for today.")
            return

        di = int(st.get("day_index", 0))
        p = generate_problem(di, date_str)
        await ch.send("⚙️ **DAILY MP DROP (manual):** Solve it in C++ and submit.", embed=build_embed(p))

        st.setdefault("problems_by_date", {})[date_str] = p
        st["day_index"] = di + 1
        st["last_posted_date"] = date_str
        save_state(st)

        await send_ephemeral(interaction, f"✅ Posted today’s problem to <#{DAILY_CHANNEL_ID}> (day_index was {di}).")

    @tree.command(name="reset_today", description="Admin: reset today's stored problem")
    @slash_admin_only()
    async def slash_reset_today(interaction: discord.Interaction):
        st = load_state()
        date_str = today_str_ph()
        pb = st.get("problems_by_date", {})

        if date_str not in pb:
            await send_ephemeral(interaction, "✅ Nothing to reset for today (no stored problem).")
            return

        pb.pop(date_str, None)
        st["problems_by_date"] = pb
        st["last_posted_date"] = None
        save_state(st)
        await send_ephemeral(interaction, "✅ Reset done. Use `/postnow` to post a new problem for today.")

    @tree.command(name="regen_today", description="Admin: regenerate today's problem (new seed)")
    @slash_admin_only()
    async def slash_regen_today(interaction: discord.Interaction):
        st = load_state()
        date_str = today_str_ph()
        di = int(st.get("day_index", 0))
        p = generate_problem(di, date_str)
        st.setdefault("problems_by_date", {})[date_str] = p
        st["day_index"] = di + 1
        st["last_posted_date"] = date_str
        append_audit(st, {"action": "regen_today", "by": str(interaction.user), "day_index": di, "seed": p.get("seed")})
        save_state(st)

        ch = bot.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
        if ch and isinstance(ch, discord.abc.Messageable):
            await ch.send("⚙️ **DAILY MP DROP (regenerated):**", embed=build_embed(p))

        await send_ephemeral(interaction, f"✅ Regenerated today's problem (seed={p.get('seed')}).")

    @tree.command(name="repost_date", description="Admin: repost stored problem for a date (YYYY-MM-DD)")
    @slash_admin_only()
    @app_commands.describe(date="Date like 2026-02-13")
    async def slash_repost_date(interaction: discord.Interaction, date: str):
        st = load_state()
        p = st.get("problems_by_date", {}).get(date)
        if not p:
            await send_ephemeral(interaction, "❌ No stored problem for that date.")
            return
        ch = bot.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
        if ch is None or not isinstance(ch, discord.abc.Messageable):
            await send_ephemeral(interaction, "❌ Daily channel not found.")
            return
        await ch.send(f"⚙️ **DAILY MP DROP (backfill {date}):**", embed=build_embed(p))
        append_audit(st, {"action": "repost_date", "by": str(interaction.user), "date": date})
        save_state(st)
        await send_ephemeral(interaction, "✅ Reposted.")

    dev = app_commands.Group(name="dev", description="Admin: dev tools")

    @dev.command(name="help", description="Show dev commands")
    @slash_admin_only()
    async def dev_help(interaction: discord.Interaction):
        await send_ephemeral(
            interaction,
            "**DEV COMMANDS:**\n"
            "`/dev list` → list all families/kinds\n"
            "`/dev random [family]` → pick random problem\n"
            "`/dev pick <family> <kind>` → pick specific problem\n"
            "`/dev setup` → show runtime config\n"
        )

    @dev.command(name="setup", description="Show runtime config")
    @slash_admin_only()
    async def dev_setup(interaction: discord.Interaction):
        ch = bot.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
        await send_ephemeral(
            interaction,
            f"✅ Bot online.\n"
            f"- Daily channel ID: `{DAILY_CHANNEL_ID}` (name: `{getattr(ch, 'name', None)}`)\n"
            f"- Submit channel ID: `{SUBMIT_CHANNEL_ID}` (0 means any channel)\n"
            f"- Post time: `{POST_TIME}` (PH time)\n"
            f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
            f"- Windows mode: `{IS_WINDOWS}`\n"
            f"- GPP: `{GPP}`\n"
            f"- ADMIN_ROLES: `{ADMIN_ROLES}`\n"
        )

    @dev.command(name="list", description="List all families and kinds")
    @slash_admin_only()
    async def dev_list(interaction: discord.Interaction):
        msg = "**FAMILIES AND KINDS:**\n"
        total_count = 0
        for f, kinds in family_kinds.items():
            msg += f"- **{f}**: {', '.join(f'`{k}`' for k in kinds)}\n"
            total_count += len(kinds)
        msg += f"\n**Total:** {total_count}"
        await send_ephemeral(interaction, msg)

    @dev.command(name="random", description="Pick a random problem (optionally choose a family)")
    @slash_admin_only()
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
                await send_ephemeral(interaction, f"❌ Invalid family '{family}'. Available:\n{families_list}")
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
                await send_ephemeral(interaction, "❌ Unexpected error: family not recognized.")
                return
        except ValueError as e:
            await send_ephemeral(interaction, f"❌ {e}")
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
    @slash_admin_only()
    @app_commands.describe(family="Family (e.g. arrays_basic)", kind="Kind (from /dev list)")
    async def dev_pick(interaction: discord.Interaction, family: str, kind: str):
        st = load_state()
        date_str = today_str_ph()
        family = family.lower().strip()
        kind = kind.strip()

        if family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await send_ephemeral(interaction, f"❌ Invalid family. Available:\n{families_list}")
            return

        if kind not in family_kinds[family]:
            kinds_list = ", ".join(f"`{k}`" for k in family_kinds[family])
            await send_ephemeral(interaction, f"❌ Invalid kind for `{family}`. Available:\n{kinds_list}")
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
                await send_ephemeral(interaction, "❌ Unexpected error: family not recognized.")
                return
        except ValueError as e:
            await send_ephemeral(interaction, f"❌ {e}")
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

# =========================
# PREFIX COMMANDS
# =========================
if ENABLE_PREFIX:
    @bot.command(name="help")
    async def p_help(ctx: commands.Context):
        await send_public(ctx, help_text())

    @bot.command(name="ping")
    async def p_ping(ctx: commands.Context):
        await send_public(ctx, "pong")

    @bot.command(name="whoami")
    async def p_whoami(ctx: commands.Context):
        if not isinstance(ctx.author, discord.Member):
            await ctx.send("DM context (not a server).")
            return
        await ctx.send(
            "✅ Prefix system is working.\n"
            f"- User: `{ctx.author}`\n"
            f"- Admin perm: `{ctx.author.guild_permissions.administrator}`\n"
            f"- Roles: `{[r.name for r in ctx.author.roles]}`\n"
            f"- ADMIN_ROLES expected: `{ADMIN_ROLES}`\n"
            f"- ENABLE_PREFIX: `{ENABLE_PREFIX}` Prefix: `{COMMAND_PREFIX}`\n"
            f"- message_content intent (code): `{bot.intents.message_content}`"
        )

    @bot.command(name="today")
    async def p_today(ctx: commands.Context):
        st = load_state()
        p = st.get("problems_by_date", {}).get(today_str_ph())
        if not p:
            await send_public(ctx, f"❌ No problem stored for today yet. Ask admin to `{COMMAND_PREFIX}postnow` or wait for schedule.")
            return
        await send_public(ctx, embed=build_embed(p))

    @bot.command(name="rules")
    async def p_rules(ctx: commands.Context):
        await send_public(
            ctx,
            "**How to Submit (C++ only)**\n"
            f"Use `{COMMAND_PREFIX}submit` and either:\n"
            "1) Paste your full code after the command (you may include a ```cpp``` block), OR\n"
            "2) Attach a `.cpp` file.\n\n"
            "**No prompts** like `Enter n:` — output must match exactly."
        )

    @bot.command(name="format")
    async def p_format(ctx: commands.Context):
        await p_rules(ctx)

    @bot.command(name="explain")
    async def p_explain(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        fam = str(p.get("family", "")).lower()
        mapping = {
            "arrays_basic": "🧠 **Arrays**: store input values in an array/vector and process using indexing like `a[i]`.",
            "arrays_nested": "🧠 **Arrays (nested)**: store data and use nested loops (`for` inside `for`).",
            "functions": "🧠 **Functions**: define at least one user-defined function (besides `main`) and call it from `main()`.",
            "recursion": "🧠 **Recursion**: create a function that calls itself with a smaller input, with a clear base case.",
            "strings": "🧠 **Strings**: use `string` / `getline` and handle spaces vs tokens carefully.",
            "patterns": "🧠 **Patterns**: use loops to print line-by-line; outer loop = rows, inner loop = columns.",
            "stl_intro": "🧠 **STL**: use `vector`, `sort`, `set`, `map`, etc. as required.",
        }
        await send_public(ctx, mapping.get(fam, f"🧠 Topic: **{fam}**. Follow the input/output format and apply the required concept."))

    @bot.command(name="approach")
    async def p_approach(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        fam = str(p.get("family", "")).lower()
        steps = ["1) Read input exactly as specified."]
        if fam == "arrays_basic":
            steps += ["2) Store the n values in `vector<long long> a(n)`.","3) Loop through `a[i]` to compute the required result.","4) Print the result only (no prompts)."]
        elif fam == "arrays_nested":
            steps += ["2) Store the data (often 2D / multiple values).","3) Use nested loops (`for` inside `for`) to compute.","4) Print result only."]
        elif fam == "functions":
            steps += ["2) Write a helper function that solves the core task.","3) Call it from `main()` and print the return value."]
        elif fam == "recursion":
            steps += ["2) Identify base case.","3) Write recursive step reducing the problem size.","4) Call the recursive function and print the result."]
        else:
            steps += ["2) Use the required topic tools (see `!explain`).", "3) Match output format exactly."]
        await send_public(ctx, "🧩 **Approach (today's problem)**\n" + "\n".join(steps))

    @bot.command(name="hint")
    async def p_hint(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, ctx.author.id, today_str_ph())
        if not ok:
            await send_public(ctx, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        await send_public(ctx, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {tutor_hints(p)[0]}")

    @bot.command(name="hint2")
    async def p_hint2(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, ctx.author.id, today_str_ph())
        if not ok:
            await send_public(ctx, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        hints = tutor_hints(p)
        await send_public(ctx, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {hints[1] if len(hints) > 1 else hints[0]}")

    @bot.command(name="hint3")
    async def p_hint3(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        st = load_state()
        ok, used = consume_hint(st, ctx.author.id, today_str_ph())
        if not ok:
            await send_public(ctx, f"❌ Hint limit reached for today ({HINTS_PER_DAY_LIMIT}).")
            return
        save_state(st)
        hints = tutor_hints(p)
        msg = hints[2] if len(hints) > 2 else hints[-1]
        if TUTOR_FULL_CODE:
            msg += "\n\n✅ *(Tutor mode)* `TUTOR_FULL_CODE=true` is enabled."
        await send_public(ctx, f"💡 ({used}/{HINTS_PER_DAY_LIMIT}) {msg}")

    @bot.command(name="dryrun")
    async def p_dryrun(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        await send_public(
            ctx,
            "🧪 **Sample Dry Run**\n"
            f"**Sample Input**\n```text\n{p.get('sample_in','')}```"
            f"**Sample Output**\n```text\n{p.get('sample_out','')}```"
        )

    @bot.command(name="constraints")
    async def p_constraints(ctx: commands.Context):
        p = get_today_problem_from_state()
        if not p:
            await send_public(ctx, f"❌ No stored problem for today yet. Ask admin to `{COMMAND_PREFIX}postnow`.")
            return
        await send_public(ctx, f"📌 **Constraints**\n{p.get('constraints','-')}")

    @bot.command(name="leaderboard")
    async def p_leaderboard(ctx: commands.Context):
        st = load_state()
        rows = list(st.get("scores", {}).values())
        if not rows:
            await send_public(ctx, "No leaderboard data yet.")
            return
        rows.sort(key=lambda r: (int(r.get("weekly_accepts", 0)), int(r.get("accepted", 0))), reverse=True)
        lines = ["🏆 **Weekly Leaderboard**"]
        for i, r in enumerate(rows[:10], start=1):
            lines.append(f"{i}. **{r.get('name','user')}** — weekly `{r.get('weekly_accepts',0)}` | total `{r.get('accepted',0)}` | streak `{r.get('streak',0)}`")
        await send_public(ctx, "\n".join(lines))

    @bot.command(name="submit")
    async def p_submit(ctx: commands.Context, *, payload: str = ""):
        attachment = ctx.message.attachments[0] if ctx.message.attachments else None

        async def progress(msg: str):
            await send_public(ctx, msg)

        async def result(msg: str):
            await send_public(ctx, msg)

        await progress("✅ Submission received. Starting judge...")
        await judge_submission(
            user_id=ctx.author.id,
            user_display=str(ctx.author),
            channel_id=ctx.channel.id,
            code_text=payload.strip() if payload.strip() else None,
            attachment=attachment,
            progress_cb=progress,
            result_cb=result,
        )

    # ---- PREFIX: ADMIN ----
    @bot.command(name="status")
    @prefix_admin_only()
    async def p_status(ctx: commands.Context):
        st = load_state()
        date_str = today_str_ph()
        stored = date_str in st.get("problems_by_date", {})
        di = int(st.get("day_index", 0))
        up = int(time.monotonic() - BOT_START_MONO)
        nxt = next_post_time_ph()
        await send_public(
            ctx,
            "**Bot Status**\n"
            f"- Uptime: `{up}s`\n"
            f"- ENABLE_PREFIX: `{ENABLE_PREFIX}` | ENABLE_SLASH: `{ENABLE_SLASH}`\n"
            f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
            f"- Day index: `{di}`\n"
            f"- Today stored: `{stored}` ({date_str})\n"
            f"- Next scheduled post (PH): `{nxt.isoformat()}`\n"
            f"- Queue busy: `{SUBMIT_LOCK.locked()}`\n"
            f"- DAILY_CHANNEL_ID: `{DAILY_CHANNEL_ID}`\n"
            f"- SUBMIT_CHANNEL_ID: `{SUBMIT_CHANNEL_ID}`\n"
            f"- GPP: `{GPP}` exists=`{bool(shutil.which(GPP) or os.path.exists(GPP))}`\n"
            f"- Metrics: `{JUDGE_METRICS}`\n"
            f"- Build: `{BOT_UPDATES_VERSION}`\n"
        )

    @bot.command(name="postnow")
    @prefix_admin_only()
    async def p_postnow(ctx: commands.Context):
        if DAILY_CHANNEL_ID == 0:
            await send_public(ctx, "❌ DAILY_CHANNEL_ID is 0 / missing. Set env DAILY_CHANNEL_ID.")
            return

        ch = bot.get_channel(DAILY_CHANNEL_ID)
        if ch is None:
            await send_public(ctx, f"❌ Can't find channel for DAILY_CHANNEL_ID={DAILY_CHANNEL_ID}. Check the ID.")
            return
        if not isinstance(ch, discord.abc.Messageable):
            await send_public(ctx, "❌ Daily channel is not messageable.")
            return

        try:
            st = load_state()
            date_str = today_str_ph()
            existing = st.get("problems_by_date", {}).get(date_str)

            if existing:
                await ch.send("⚙️ **DAILY MP DROP (repost):**", embed=build_embed(existing))
                await send_public(ctx, "✅ Reposted the already-stored problem for today.")
                return

            di = int(st.get("day_index", 0))
            p = generate_problem(di, date_str)

            await ch.send("⚙️ **DAILY MP DROP (manual):** Solve it in C++ and submit.", embed=build_embed(p))

            st.setdefault("problems_by_date", {})[date_str] = p
            st["day_index"] = di + 1
            st["last_posted_date"] = date_str
            save_state(st)

            await send_public(ctx, f"✅ Posted today’s problem to <#{DAILY_CHANNEL_ID}> (day_index was {di}).")
        except discord.Forbidden:
            await send_public(ctx, "❌ Bot lacks permission in daily channel (Send Messages + Embed Links).")
        except Exception as e:
            logging.exception("postnow error: %s", e)
            await send_public(ctx, f"❌ postnow crashed: `{type(e).__name__}` (check logs)")

# =========================
# READY: SYNC + START TASKS
# =========================
@bot.event
async def on_ready():
    logging.info("Logged in as %s (id: %s)", bot.user, bot.user.id)
    logging.info("Config: DAILY_CHANNEL_ID=%s SUBMIT_CHANNEL_ID=%s ENABLE_PREFIX=%s ENABLE_SLASH=%s ENFORCE_SKILLS=%s ADMIN_ROLES=%s",
                 DAILY_CHANNEL_ID, SUBMIT_CHANNEL_ID, ENABLE_PREFIX, ENABLE_SLASH, ENFORCE_SKILLS, ADMIN_ROLES)
    logging.info("Config: COMPILE_TIMEOUT=%ss RUN_TIMEOUT=%ss MAX_CODE_BYTES=%s COOLDOWN=%ss",
                 COMPILE_TIMEOUT_SEC, RUN_TIMEOUT_SEC, MAX_CODE_BYTES, SUBMIT_COOLDOWN_SEC)
    logging.info("Command System: prefix=%r message_content_intent(code)=%s", COMMAND_PREFIX, bot.intents.message_content)

    if ENABLE_SLASH:
        try:
            await bot.tree.sync()
            logging.info("Synced slash commands.")
        except Exception as e:
            logging.exception("Slash sync failed: %s", e)

    if not post_daily_problem.is_running():
        post_daily_problem.start()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    validate_config()
    bot.run(TOKEN)
