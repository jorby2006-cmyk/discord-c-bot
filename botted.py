# botted.py — CS1 Daily MP + C++ Judge (Windows/MSYS2-friendly)
#
# QoL + enforcement edition:
# - Admin commands: !postnow, !reset_today, !status
# - Student help: !rules / !format
# - Anti-spam: per-user submit cooldown + code size limit
# - Better WA feedback: first mismatch line + truncation
# - Skill enforcement (heuristics) per problem family (toggle via ENFORCE_SKILLS)
# - Attachment support: .cpp/.cc/.cxx and .txt containing C++ code blocks
#
# NOTE: This is still a heuristic checker. It enforces “use arrays / nested loops / recursion”
# by scanning source text, not by AST parsing.

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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

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

import discord
from discord.ext import tasks, commands

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

# Judge limits (override via env vars if you want)
COMPILE_TIMEOUT_SEC = int(os.getenv("COMPILE_TIMEOUT_SEC", "12"))
RUN_TIMEOUT_SEC = int(os.getenv("RUN_TIMEOUT_SEC", "2"))
MAX_OUTPUT_BYTES = int(os.getenv("MAX_OUTPUT_BYTES", "64000"))

# Code size limit (to avoid abuse)
MAX_CODE_BYTES = int(os.getenv("MAX_CODE_BYTES", "100000"))  # ~100KB

# Cooldown per user to avoid spam
SUBMIT_COOLDOWN_SEC = int(os.getenv("SUBMIT_COOLDOWN_SEC", "15"))

# Toggle skill enforcement quickly if you need to
ENFORCE_SKILLS = os.getenv("ENFORCE_SKILLS", "true").strip().lower() in ("1", "true", "yes", "y", "on")

# Admin roles list (comma-separated). If empty, fall back to "Root Admin".
ADMIN_ROLES = [r.strip() for r in os.getenv("ADMIN_ROLES", "Root Admin").split(",") if r.strip()]

# g++ command.
# On Windows (MSYS2), set env var GPP to something like:
#   C:\msys64\ucrt64\bin\g++.exe
GPP = os.getenv("GPP", "g++")

IS_WINDOWS = (os.name == "nt")

# Only allow one submission to compile/run at a time (no overlaps)
SUBMIT_LOCK = asyncio.Lock()

# In-memory QoL state (resets on restart)
BOT_START_MONO = time.monotonic()
USER_LAST_SUBMIT: Dict[int, float] = {}
USER_LAST_COMPILE_ERR: Dict[int, str] = {}

# =========================
# DISCORD BOT SETUP
# =========================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def is_admin_member(member: discord.Member) -> bool:
    # True if member has ANY role name in ADMIN_ROLES, or is guild admin.
    try:
        if member.guild_permissions.administrator:
            return True
        names = {r.name for r in member.roles}
        return any(ar in names for ar in ADMIN_ROLES)
    except Exception:
        return False

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
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"day_index": 0, "last_posted_date": None, "problems_by_date": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"day_index": 0, "last_posted_date": None, "problems_by_date": {}}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

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
    families = ["arrays_basic", "arrays_nested", "bool_checks", "functions", "patterns", "strings", "math_logic", "recursion", "stl_intro"]
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
    else:  # stl_intro
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
        value="Use `!submit` then paste your full C++ code in a ```cpp``` block (or attach a .cpp file). "
              "No prompts like `Enter n:` — output must match exactly.",
        inline=False,
    )
    embed.set_footer(text=f"Day: {problem['day']} • Seed: {problem['seed']}")
    return embed

# =========================
# JUDGE HELPERS
# =========================
CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_cpp_from_message(content: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(content)
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None

async def read_attachment_code(message: discord.Message) -> Optional[str]:
    # Accept .cpp/.cc/.cxx and also .txt (common student mistake).
    if not message.attachments:
        return None
    for att in message.attachments:
        fn = att.filename.lower()
        if fn.endswith((".cpp", ".cc", ".cxx", ".txt")):
            data = await att.read()
            text = data.decode("utf-8", errors="replace").strip()
            # If they uploaded a .txt that contains a ```cpp``` block, prefer extracting it.
            code = extract_cpp_from_message(text) or text
            return code.strip() if code else None
    return None

def exe_path(workdir: str) -> str:
    return os.path.join(workdir, "main.exe" if IS_WINDOWS else "main.out")

# -------------------------
# Better diff utilities
# -------------------------
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
        r"(?:static\s+|inline\s+|constexpr\s+|friend\s+)?*"
        r"(?:[\w:\<\>\,\&\*\s]+?)\s+"
        r"([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{"
    )
    names = [m.group(1) for m in fn_pat.finditer(code)]
    return any(n != "main" for n in names)

def _has_array_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)

    # container types
    has_vector = bool(re.search(r"\b(?:std::)?vector\s*<", code))
    has_array = bool(re.search(r"\b(?:std::)?array\s*<", code))
    # C-style array declarations like: int a[100];  long long b[n];
    has_c_array_decl = bool(re.search(r"\b(?:bool|char|short|int|long|long\s+long|float|double|string|std::string)\s+\w+\s*\[\s*\w*\s*\]\s*;", code))

    # index access: a[i], a[i+1], etc.
    has_index = bool(re.search(r"\[[^\]]+\]", code))

    return (has_vector or has_array or has_c_array_decl) and has_index

def _has_nested_loops(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    pat = re.compile(r"(for|while)\s*\([^\)]*\)\s*\{[\s\S]{0,600}?(for|while)\s*\(", re.MULTILINE)
    if pat.search(code):
        return True
    # fallback: at least 2 loops present (weak but helpful)
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
    # patterns: star or nested loops or repeated prints
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

def enforce_skill(problem: dict, code: str) -> Tuple[bool, str]:
    fam = str(problem.get("family", "")).lower()

    if fam == "arrays_basic":
        if not _has_array_usage(code):
            return False, "❌ **ARRAYS (basic)**: I didn’t detect array/vector + indexing like `a[i]`."
    elif fam == "arrays_nested":
        if not _has_array_usage(code):
            return False, "❌ **ARRAYS (nested)**: I didn’t detect array/vector + indexing like `a[i]`."
        if not _has_nested_loops(code):
            return False, "❌ **ARRAYS (nested)**: I didn’t detect **nested loops** (e.g., `for` inside `for`)."
        if re.search(r"\bmap\b|\bunordered_map\b", _strip_cpp_comments_and_strings(code)):
            return False, "❌ **ARRAYS (nested)**: `map`/`unordered_map` not allowed. Use loops as required."
    elif fam == "bool_checks":
        if not _has_bool_logic(code):
            return False, "❌ **BOOL CHECKS**: I didn’t detect boolean logic (`bool`, comparisons, true/false)."
    elif fam == "functions":
        if not _has_user_defined_function(code):
            return False, "❌ **FUNCTIONS**: Define at least one user-defined function (besides `main`)."
    elif fam == "patterns":
        if not _has_pattern_printing(code):
            return False, "❌ **PATTERNS**: Use loops to print the pattern (typically `cout` inside loops)."
    elif fam == "strings":
        if not _has_string_usage(code):
            return False, "❌ **STRINGS**: Use `string`/`std::string` or `getline`."
    elif fam == "math_logic":
        if not _has_math_logic_ops(code):
            return False, "❌ **MATH/LOGIC**: I didn’t detect typical math ops (`%`, arithmetic, etc.)."
    elif fam == "recursion":
        if not _has_recursion(code):
            return False, "❌ **RECURSION**: I didn’t detect a recursive function (a function calling itself)."
    elif fam == "stl_intro":
        if not _has_stl_usage(code):
            return False, "❌ **STL INTRO**: Use STL (e.g., `vector`, `set`, `map`, or `sort`)."

    return True, ""

# =========================
# SUBPROCESS + JUDGE
# =========================
async def run_subprocess(
    cmd: List[str],
    stdin_data: Optional[bytes],
    timeout_sec: int,
    cwd: Optional[str] = None
) -> Tuple[int, bytes, bytes]:
    creationflags = 0
    if IS_WINDOWS:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=creationflags,
    )

    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout_sec)
        return proc.returncode, out[:MAX_OUTPUT_BYTES], err[:MAX_OUTPUT_BYTES]
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
        return -999, b"", b"TIMEOUT"

async def compile_cpp(code: str, workdir: str) -> Tuple[bool, str]:
    src = os.path.join(workdir, "main.cpp")
    exe = exe_path(workdir)

    with open(src, "w", encoding="utf-8") as f:
        f.write(code + "\n")

    cmd = [GPP, "-std=c++17", src, "-O2", "-pipe", "-o", exe]

    logging.info("JUDGE: compiler=%s", GPP)
    logging.info("JUDGE: compiling: %s", " ".join(cmd))

    rc, out, err = await run_subprocess(cmd, stdin_data=None, timeout_sec=COMPILE_TIMEOUT_SEC, cwd=workdir)

    if rc == 0 and os.path.exists(exe):
        return True, ""

    if rc == -999:
        return False, "Compilation timed out."

    msg = (out.decode("utf-8", errors="replace") + "\n" + err.decode("utf-8", errors="replace")).strip()
    return False, msg if msg else "Compilation failed (no output). Check that GPP points to g++."

async def run_one_test(workdir: str, t: Dict[str, Any]) -> Tuple[bool, str, Optional[dict]]:
    exe = exe_path(workdir)
    inp = t["inp"].encode("utf-8")
    expected = normalize_output(t["out"])

    rc, out, err = await run_subprocess([exe], stdin_data=inp, timeout_sec=RUN_TIMEOUT_SEC, cwd=workdir)

    if rc == -999:
        return False, "Time Limit Exceeded", {"test": t}

    if rc != 0:
        msg = err.decode("utf-8", errors="replace").strip()
        if msg == "TIMEOUT":
            msg = "Time Limit Exceeded"
        if not msg:
            msg = "Runtime Error"
        return False, f"Runtime Error (exit {rc})\n{msg}", {"test": t}

    got = normalize_output(out.decode("utf-8", errors="replace"))
    if got != expected:
        return False, "Wrong Answer", {"test": t, "expected": expected, "got": got}

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
    if channel is None:
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
    await bot.wait_until_ready()
    logging.info("CS1 Judge armed.")

@bot.event
async def on_ready():
    logging.info("Logged in as %s (id: %s)", bot.user, bot.user.id)

    # Startup config log (no token leak)
    logging.info("Config: DAILY_CHANNEL_ID=%s SUBMIT_CHANNEL_ID=%s ENFORCE_SKILLS=%s ADMIN_ROLES=%s",
                 DAILY_CHANNEL_ID, SUBMIT_CHANNEL_ID, ENFORCE_SKILLS, ADMIN_ROLES)
    logging.info("Config: COMPILE_TIMEOUT=%ss RUN_TIMEOUT=%ss MAX_CODE_BYTES=%s COOLDOWN=%ss",
                 COMPILE_TIMEOUT_SEC, RUN_TIMEOUT_SEC, MAX_CODE_BYTES, SUBMIT_COOLDOWN_SEC)

    if not post_daily_problem.is_running():
        post_daily_problem.start()

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    await ctx.send(f"❌ {type(error).__name__}: {error}")

# =========================
# STUDENT COMMANDS
# =========================
@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send("pong")

@bot.command(name="today")
async def today(ctx: commands.Context):
    state = load_state()
    date_str = today_str_ph()
    p = state.get("problems_by_date", {}).get(date_str)
    if not p:
        await ctx.send("❌ No problem stored for today yet. Ask admin to `!postnow` or wait for schedule.")
        return
    await ctx.send(embed=build_embed(p))

@bot.command(name="rules")
async def rules(ctx: commands.Context):
    await ctx.send(
        "**How to Submit (C++ only)**\n"
        "1) Type `!submit` and paste your full code inside a cpp block:\n"
        "```cpp\n"
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        "int main(){\n"
        "  ios::sync_with_stdio(false);\n"
        "  cin.tie(nullptr);\n"
        "  // solve...\n"
        "  return 0;\n"
        "}\n"
        "```\n"
        "2) No extra prompts like `Enter n:`.\n"
        "3) Output must match exactly.\n"
    )

@bot.command(name="format")
async def format_cmd(ctx: commands.Context):
    await rules(ctx)

# =========================
# ADMIN COMMANDS
# =========================
@bot.command(name="status")
async def status(ctx: commands.Context):
    if not isinstance(ctx.author, discord.Member) or not is_admin_member(ctx.author):
        await ctx.send("❌ Admin only.")
        return

    state = load_state()
    date_str = today_str_ph()
    stored = date_str in state.get("problems_by_date", {})
    di = int(state.get("day_index", 0))
    up = int(time.monotonic() - BOT_START_MONO)
    nxt = next_post_time_ph()

    await ctx.send(
        "**Bot Status**\n"
        f"- Uptime: `{up}s`\n"
        f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
        f"- Day index: `{di}`\n"
        f"- Today stored: `{stored}` ({date_str})\n"
        f"- Next scheduled post (PH): `{nxt.isoformat()}`\n"
        f"- DAILY_CHANNEL_ID: `{DAILY_CHANNEL_ID}`\n"
        f"- SUBMIT_CHANNEL_ID: `{SUBMIT_CHANNEL_ID}`\n"
        f"- GPP: `{GPP}`\n"
    )

@bot.command(name="postnow")
async def postnow(ctx: commands.Context):
    if not isinstance(ctx.author, discord.Member) or not is_admin_member(ctx.author):
        await ctx.send("❌ Admin only.")
        return

    if DAILY_CHANNEL_ID == 0:
        await ctx.send("❌ DAILY_CHANNEL_ID not set.")
        return

    channel = bot.get_channel(DAILY_CHANNEL_ID)
    if channel is None:
        await ctx.send("❌ Daily channel not found. Check DAILY_CHANNEL_ID.")
        return

    state = load_state()
    date_str = today_str_ph()

    existing = state.get("problems_by_date", {}).get(date_str)
    if existing:
        await channel.send("⚙️ **DAILY MP DROP (repost):**", embed=build_embed(existing))
        await ctx.send("✅ Reposted the already-stored problem for today.")
        return

    day_index = int(state.get("day_index", 0))
    problem = generate_problem(day_index, date_str)

    await channel.send("⚙️ **DAILY MP DROP (manual):** Solve it in C++ and submit with `!submit`.", embed=build_embed(problem))

    pb = state.get("problems_by_date", {})
    pb[date_str] = problem
    state["problems_by_date"] = pb
    state["day_index"] = day_index + 1
    state["last_posted_date"] = date_str
    save_state(state)

    await ctx.send(f"✅ Posted today’s problem to <#{DAILY_CHANNEL_ID}> (day_index was {day_index}).")

@bot.command(name="reset_today")
async def reset_today(ctx: commands.Context):
    if not isinstance(ctx.author, discord.Member) or not is_admin_member(ctx.author):
        await ctx.send("❌ Admin only.")
        return

    state = load_state()
    date_str = today_str_ph()
    pb = state.get("problems_by_date", {})

    if date_str not in pb:
        await ctx.send("✅ Nothing to reset for today (no stored problem).")
        return

    # Remove today's problem
    pb.pop(date_str, None)
    state["problems_by_date"] = pb
    state["last_posted_date"] = None  # allow schedule to post again

    # Do NOT decrement day_index automatically (can cause repeats). Keep simple.
    save_state(state)
    await ctx.send("✅ Reset done. Use `!postnow` to post a new problem for today.")

# =========================
# SUBMIT COMMAND
# =========================
@bot.command(name="submit")
async def submit(ctx: commands.Context):
    # If another submission is running, give quick feedback before waiting.
    if SUBMIT_LOCK.locked():
        await ctx.send("⏳ Another submission is being judged right now. Your turn is next—please wait.")
    async with SUBMIT_LOCK:
        # Channel restriction
        if SUBMIT_CHANNEL_ID and ctx.channel.id != SUBMIT_CHANNEL_ID:
            if SUBMIT_CHANNEL_ID != DAILY_CHANNEL_ID:
                await ctx.send(f"❌ Submit only in <#{SUBMIT_CHANNEL_ID}>.")
                return

        # Cooldown
        uid = ctx.author.id
        now = time.monotonic()
        last = USER_LAST_SUBMIT.get(uid, 0.0)
        if (now - last) < SUBMIT_COOLDOWN_SEC:
            wait = int(SUBMIT_COOLDOWN_SEC - (now - last))
            await ctx.send(f"⏳ Cooldown: wait `{wait}s` before submitting again.")
            return
        USER_LAST_SUBMIT[uid] = now

        state = load_state()
        date_str = today_str_ph()
        problem = state.get("problems_by_date", {}).get(date_str)
        if not problem:
            await ctx.send("❌ No active problem for today. Ask admin to `!postnow`.")
            return

        msg: discord.Message = ctx.message

        code = await read_attachment_code(msg)
        if not code:
            code = extract_cpp_from_message(msg.content)

        if not code:
            await ctx.send("❌ I didn't find C++ code. Paste it inside a ```cpp``` block or attach a .cpp file.")
            return

        # Code size limit
        if len(code.encode("utf-8", errors="ignore")) > MAX_CODE_BYTES:
            await ctx.send(f"❌ Code too large. Limit is {MAX_CODE_BYTES} bytes.")
            return

        # Prompt warnings (common)
        if re.search(r'cout\s*<<\s*".*(enter|input|please)', code, flags=re.IGNORECASE):
            await ctx.send("⚠️ Heads up: prompts like `Enter n:` often cause Wrong Answer. Output should be answer only.")

        # Skill enforcement
        if ENFORCE_SKILLS:
            ok_skill, skill_msg = enforce_skill(problem, code)
            if not ok_skill:
                await ctx.send(skill_msg + "\n(Mark: set `ENFORCE_SKILLS=false` to disable enforcement.)")
                return

        tests = problem.get("tests", [])
        status_msg = await ctx.send("🧪 Compiling...")

        with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
            ok, cerr = await compile_cpp(code, workdir)
            if not ok:
                cerr = cerr.strip()
                USER_LAST_COMPILE_ERR[uid] = cerr
                short = cerr
                if len(short) > 1800:
                    short = short[:1800] + "\n... (truncated)"
                await status_msg.edit(content="❌ Compilation Error")
                await ctx.send(f"```text\n{short}\n```")
                return

            await status_msg.edit(content=f"✅ Compiled. Running tests (0/{len(tests)})...")

            for i, t in enumerate(tests, start=1):
                await status_msg.edit(content=f"🏃 Running tests ({i}/{len(tests)})...")
                passed, verdict, details = await run_one_test(workdir, t)
                if not passed:
                    await status_msg.edit(content=f"❌ {verdict} — failed test #{i}")

                    tinp = details["test"]["inp"] if details and "test" in details else ""

                    if verdict == "Wrong Answer" and details:
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
                        await ctx.send(msg_out)
                    else:
                        if tinp:
                            await ctx.send(f"**Input**\n```text\n{clamp_block(tinp, 1200)}```")
                    return

            await status_msg.edit(content=f"✅ Accepted — {len(tests)}/{len(tests)} tests passed.")
            await ctx.send(f"Problem: **{problem['title']}** (Day {problem['day']})")

# =========================
# DEV COMMANDS (Teacher tools)
# =========================
@bot.command(name="dev")
async def dev(ctx: commands.Context, action: str, family: Optional[str] = None, kind: Optional[str] = None):
    # Admin-only dev command bundle.
    if not isinstance(ctx.author, discord.Member) or not is_admin_member(ctx.author):
        await ctx.send("❌ Admin only.")
        return

    state = load_state()
    date_str = today_str_ph()
    action = action.lower().strip()

    if action == "help":
        await ctx.send(
            "**DEV COMMANDS:**\n"
            "`!dev list` → list all families/kinds\n"
            "`!dev random [family]` → pick random problem\n"
            "`!dev pick <family> <kind>` → pick specific problem\n"
            "`!dev submit` → judge your code in this channel (admin testing)\n"
            "`!dev postnow` → same as `!postnow`\n"
            "`!dev reset_today` → same as `!reset_today`\n"
            "`!dev setup` → show runtime config\n"
        )
        return

    if action == "setup":
        ch = bot.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
        await ctx.send(
            f"✅ Bot online.\n"
            f"- Daily channel ID: `{DAILY_CHANNEL_ID}` (name: `{getattr(ch, 'name', None)}`)\n"
            f"- Submit channel ID: `{SUBMIT_CHANNEL_ID}` (0 means any channel)\n"
            f"- Post time: `{POST_TIME}` (PH time)\n"
            f"- ENFORCE_SKILLS: `{ENFORCE_SKILLS}`\n"
            f"- Windows mode: `{IS_WINDOWS}`\n"
            f"- GPP: `{GPP}`\n"
            f"- ADMIN_ROLES: `{ADMIN_ROLES}`"
        )
        return

    if action == "postnow":
        await postnow(ctx)
        return

    if action == "reset_today":
        await reset_today(ctx)
        return

    if action == "list":
        msg = "**FAMILIES AND KINDS:**\n"
        total_count = 0
        for f, kinds in family_kinds.items():
            msg += f"- **{f}**: {', '.join(f'`{k}`' for k in kinds)}\n"
            total_count += len(kinds)
        msg += f"\n**Total:** {total_count}"
        await ctx.send(msg)
        return

    if action == "random":
        if family is None:
            family = random.choice(list(family_kinds.keys()))
        elif family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await ctx.send(f"❌ Invalid family '{family}'. Available families:\n{families_list}")
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
                await ctx.send("❌ Unexpected error: family not recognized.")
                return
        except ValueError as e:
            await ctx.send(f"❌ {e}")
            return

        p["day"] = date_str
        p["seed"] = seed
        p["day_index"] = day_index

        pb = state.get("problems_by_date", {})
        pb[date_str] = p
        state["problems_by_date"] = pb
        state["last_posted_date"] = date_str
        state["day_index"] = day_index + 1
        save_state(state)

        await ctx.send(f"⚙️ **DEV PICK RANDOM:** {family} • {kind}", embed=build_embed(p))
        return

    if action == "pick":
        family = (family or "").lower().strip()
        kind = (kind or "").strip()

        if not family or family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await ctx.send(f"❌ Invalid or missing family. Available families:\n{families_list}")
            return

        if not kind or kind not in family_kinds[family]:
            kinds_list = ", ".join(f"`{k}`" for k in family_kinds[family])
            await ctx.send(f"❌ Invalid or missing kind for family `{family}`. Available kinds:\n{kinds_list}")
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
                await ctx.send("❌ Unexpected error: family not recognized.")
                return
        except ValueError as e:
            await ctx.send(f"❌ {e}")
            return

        p["day"] = date_str
        p["seed"] = seed
        p["day_index"] = day_index

        pb = state.get("problems_by_date", {})
        pb[date_str] = p
        state["problems_by_date"] = pb
        state["last_posted_date"] = date_str
        state["day_index"] = day_index + 1
        save_state(state)

        await ctx.send(f"⚙️ **DEV PICK:** {family} • {kind}", embed=build_embed(p))
        return

    if action == "submit":
        # Admin testing: judge code provided in the same message.
        if ctx.channel.id != DAILY_CHANNEL_ID:
            await ctx.send(f"❌ `!dev submit` only works in <#{DAILY_CHANNEL_ID}>.")
            return

        msg: discord.Message = ctx.message
        code = await read_attachment_code(msg) or extract_cpp_from_message(msg.content)

        if not code:
            await ctx.send("❌ No C++ code found in message or attachment.")
            return

        problem = state.get("problems_by_date", {}).get(date_str)
        if not problem:
            await ctx.send("❌ No active problem for today. Pick one first with `!dev pick`.")
            return

        if ENFORCE_SKILLS:
            ok_skill, skill_msg = enforce_skill(problem, code)
            if not ok_skill:
                await ctx.send("[DEV] " + skill_msg)
                return

        async with SUBMIT_LOCK:
            status_msg = await ctx.send("🧪 [DEV] Compiling...")

            with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
                ok, cerr = await compile_cpp(code, workdir)
                if not ok:
                    cerr = cerr.strip()
                    if len(cerr) > 1800:
                        cerr = cerr[:1800] + "\n... (truncated)"
                    await status_msg.edit(content="❌ Compilation Error")
                    await ctx.send(f"```text\n{cerr}\n```")
                    return

                await status_msg.edit(content=f"✅ Compiled. Running tests (0/{len(problem.get('tests',[]))})...")

                for i, t in enumerate(problem.get("tests", []), start=1):
                    await status_msg.edit(content=f"🏃 Running tests ({i}/{len(problem.get('tests',[]))})...")
                    passed, verdict, details = await run_one_test(workdir, t)
                    if not passed:
                        await status_msg.edit(content=f"❌ {verdict} — failed test #{i}")
                        tinp = details["test"]["inp"] if details and "test" in details else ""
                        if verdict == "Wrong Answer" and details:
                            exp = details["expected"]
                            got = details["got"]
                            line_no, e_line, g_line = first_mismatch_line(exp, got)
                            out_msg = (
                                f"**Input**\n```text\n{clamp_block(tinp, 900)}```"
                                f"**Expected**\n```text\n{clamp_block(exp, 900)}```"
                                f"**Got**\n```text\n{clamp_block(got, 900)}```"
                            )
                            if line_no:
                                out_msg += f"\n🔎 First mismatch at **line {line_no}**:\n- expected: `{e_line}`\n- got: `{g_line}`"
                            await ctx.send(out_msg)
                        else:
                            if tinp:
                                await ctx.send(f"**Input**\n```text\n{clamp_block(tinp, 1200)}```")
                        return

                await status_msg.edit(content="✅ [DEV] Accepted — all tests passed.")
                await ctx.send(f"[DEV] Problem: **{problem['title']}** (Day {problem['day']})")
        return

    await ctx.send("❌ Invalid `!dev` action. Use `help`, `list`, `random`, `pick`, `submit`, `postnow`, `reset_today`, `setup`.")

# =========================
# STARTUP CHECKS
# =========================
def _die(msg: str) -> None:
    raise RuntimeError(msg)

if not TOKEN:
    _die("DISCORD_TOKEN env var missing. Set DISCORD_TOKEN before running.")
if DAILY_CHANNEL_ID == 0:
    _die("DAILY_CHANNEL_ID env var missing. Set DAILY_CHANNEL_ID before running.")

bot.run(TOKEN)

