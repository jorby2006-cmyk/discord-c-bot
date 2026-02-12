# botted.py — CS1 Daily MP + C++ Judge (Windows/MSYS2-friendly)
#
# Repo-safe + practical:
# - No token is hardcoded here. It only reads DISCORD_TOKEN + channel IDs from env vars.
# - If python-dotenv is installed, it’ll load a local .env for convenience (just don’t commit .env).
#
# What it does:
# - Posts one “Daily Machine Problem” every day at 9:00 AM (Philippines time).
# - Students submit C++ via !submit; the bot compiles + runs hidden tests.
#
# A few quality-of-life details:
# - Prevents overlapping compiles/runs (submission lock).
# - Better compile error output (shows real compiler output).
# - Windows-friendly timeout killing (taskkill /T /F for process trees).
# - Doesn’t spam chat on unknown commands.

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

# g++ command.
# On Windows (MSYS2), set env var GPP to something like:
#   C:\msys64\ucrt64\bin\g++.exe
GPP = os.getenv("GPP", "g++")

IS_WINDOWS = (os.name == "nt")

# Only allow one submission to compile/run at a time (no overlaps)
SUBMIT_LOCK = asyncio.Lock()

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
    # Keeps track of day_index + the exact problem used per date.
    # If the file is missing/corrupt, fall back safely instead of crashing.
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
    # Same date + same day_index => same generated problem.
    # This makes restarts “safe” (today’s MP won’t change if the bot restarts).
    h = hashlib.sha256(f"{day_index}|{date_str}|CS1JUDGE".encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def pick_family(day_index: int) -> str:
    # Rotates problem families so the class gets variety.
    families = ["arrays_basic", "arrays_nested", "bool_checks", "functions", "patterns", "strings", "math_logic", "recursion", "stl_intro"]
    return families[day_index % len(families)]

def normalize_output(s: str) -> str:
    # Normalize line endings + trim trailing whitespace/blank lines for fair judging.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")

def total_elements(family_kinds: dict) -> int:
    return sum(len(elements) for elements in family_kinds.values())


def generate_problem(day_index: int, date_str: str) -> dict:
    # Central generator: picks the family, builds the problem, and stamps metadata.
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
    else: # stl_intro
        p = gen_stl_intro(rng)

    p["day"] = date_str
    p["seed"] = seed
    p["day_index"] = day_index
    return p

# =========================
# EMBED BUILDER
# =========================
def build_embed(problem: dict) -> discord.Embed:
    # Formats the MP nicely for Discord.
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
# JUDGE HELPERS (Windows-safe)
# =========================
CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*([\s\S]*?)```", re.IGNORECASE)

def extract_cpp_from_message(content: str) -> Optional[str]:
    # Pulls the first ```cpp ... ``` block from a message.
    m = CODE_BLOCK_RE.search(content)
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None

async def read_attachment_cpp(message: discord.Message) -> Optional[str]:
    # If they attach a .cpp file, use that.
    if not message.attachments:
        return None
    for att in message.attachments:
        if att.filename.lower().endswith((".cpp", ".cc", ".cxx")):
            data = await att.read()
            return data.decode("utf-8", errors="replace").strip()
    return None

def exe_path(workdir: str) -> str:
    return os.path.join(workdir, "main.exe" if IS_WINDOWS else "main.out")


# =========================
# SKILL ENFORCEMENT (heuristics)
# =========================

def _strip_cpp_comments_and_strings(code: str) -> str:
    # Remove //... and /*...*/ and also strip string/char literals.
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # Replace string literals with empty quotes to preserve structure a bit
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
    # Either C-style arrays or vector usage, plus at least one index access.
    has_container = bool(re.search(r"\bvector\s*<|\bstd::vector\s*<|\barray\s*<|\bstd::array\s*<|\b\w+\s*\w+\s*\[\s*\d*\s*\]", code))
    has_index = bool(re.search(r"\[[^\]]+\]", code))
    return has_container and has_index

def _has_nested_loops(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    # Very simple nesting heuristic: if a loop keyword appears inside braces after another loop.
    # This catches typical CS1 nested-for solutions.
    # Look for "for/while" then later within some block another "for/while".
    pat = re.compile(r"(for|while)\s*\([^\)]*\)\s*\{[\s\S]{0,500}?(for|while)\s*\(", re.MULTILINE)
    if pat.search(code):
        return True
    # fallback: at least 2 loops present
    return len(re.findall(r"\bfor\b|\bwhile\b", code)) >= 2

def _has_bool_logic(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\bbool\b|\btrue\b|\bfalse\b|==|!=|<=|>=|<|>", code))

def _has_string_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\bstring\b|\bstd::string\b|getline\s*\(", code))

def _has_pattern_printing(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    has_loop = bool(re.search(r"\bfor\b|\bwhile\b", code))
    # Patterns usually print '*' or digits in loops
    has_star = ("*" in code) or bool(re.search(r"\*\s*\w|\w\s*\*", code))
    uses_cout = "cout" in code
    return has_loop and uses_cout and has_star

def _has_math_logic_ops(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    # Look for typical ops used in these problems: %, sqrt, prime loops, digit ops
    return bool(re.search(r"%|\bsqrt\b|\bprime\b|\bdigit\b|/|\*|\+", code))

def _has_recursion(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    # Find function definitions (excluding main), then check for self-call anywhere in code.
    fn_pat = re.compile(
        r"(?mx)^\s*(?:[\w:\<\>\,\&\*\s]+?)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{"
    )
    names = [m.group(1) for m in fn_pat.finditer(code)]
    names = [n for n in names if n != "main"]
    for n in names:
        # require at least two appearances: definition + call
        if len(re.findall(rf"\b{re.escape(n)}\s*\(", code)) >= 2:
            return True
    return False

def _has_stl_usage(code: str) -> bool:
    code = _strip_cpp_comments_and_strings(code)
    return bool(re.search(r"\bvector\s*<|\bset\s*<|\bmap\s*<|\bunordered_map\s*<|\bsort\s*\(|\bstd::sort\b", code))

def enforce_skill(problem: dict, code: str) -> Tuple[bool, str]:
    """Return (ok, message). Heuristic enforcement per family."""
    fam = str(problem.get("family", "")).lower()
    if fam == "stl_intro" or fam == "stl_intro".lower() or fam == "stl_intro".replace(" ", "_"):
        fam = "stl_intro"

    if fam == "arrays_basic":
        if not _has_array_usage(code):
            return False, "❌ This is an **ARRAYS (basic)** problem. Your code must use an array/vector with indexing (`a[i]`)."
    elif fam == "arrays_nested":
        if not _has_array_usage(code):
            return False, "❌ This is an **ARRAYS (nested loops)** problem. Your code must use an array/vector with indexing (`a[i]`)."
        if not _has_nested_loops(code):
            return False, "❌ This is an **ARRAYS (nested loops)** problem. Your code must use **nested loops** (e.g., `for` inside `for`)."
        # Optional restriction: discourage map usage
        if re.search(r"\bmap\b|\bunordered_map\b", _strip_cpp_comments_and_strings(code)):
            return False, "❌ This **nested loops** problem disallows `map`/`unordered_map`. Use loops (as required)."
    elif fam == "bool_checks":
        if not _has_bool_logic(code):
            return False, "❌ This is a **BOOL CHECKS** problem. Your code must use boolean logic (`bool`, comparisons, true/false)."
    elif fam == "functions":
        if not _has_user_defined_function(code):
            return False, "❌ This is a **FUNCTIONS** problem. You must define at least one user-defined function (besides `main`)."
    elif fam == "patterns":
        if not _has_pattern_printing(code):
            return False, "❌ This is a **PATTERNS** problem. Your solution should use loops to print the pattern (typically `cout` in loops)."
    elif fam == "strings":
        if not _has_string_usage(code):
            return False, "❌ This is a **STRINGS** problem. Your code must use `string`/`std::string` or `getline`."
    elif fam == "math_logic":
        if not _has_math_logic_ops(code):
            return False, "❌ This is a **MATH/LOGIC** problem. Your code should use appropriate math operations (e.g., `%`, loops, etc.)."
    elif fam == "recursion":
        if not _has_recursion(code):
            return False, "❌ This is a **RECURSION** problem. Your solution must include a recursive function (a function that calls itself)."
    elif fam == "stl_intro" or fam == "stl_intro".lower() or fam == "stl_intro".replace(" ", "_") or fam == "stl_intro":
        if not _has_stl_usage(code):
            return False, "❌ This is an **STL INTRO** problem. Your code must use STL (e.g., `vector`, `set`, `map`, or `sort`)."

    return True, ""

async def run_subprocess(
    cmd: List[str],
    stdin_data: Optional[bytes],
    timeout_sec: int,
    cwd: Optional[str] = None
) -> Tuple[int, bytes, bytes]:
    # Runs a subprocess with a timeout and returns (exit_code, stdout, stderr).
    # On Windows, create a new process group so we can kill the tree on timeout.
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
                # Kill the whole process tree (common fix for stuck student code)
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
    # Writes code to main.cpp and compiles it.
    src = os.path.join(workdir, "main.cpp")
    exe = exe_path(workdir)

    with open(src, "w", encoding="utf-8") as f:
        f.write(code + "\n")

    cmd = [GPP, "-std=c++17", src, "-o", exe]

    logging.info("JUDGE: compiler=%s", GPP)
    logging.info("JUDGE: compiling: %s", " ".join(cmd))

    rc, out, err = await run_subprocess(cmd, stdin_data=None, timeout_sec=COMPILE_TIMEOUT_SEC, cwd=workdir)

    if rc == 0 and os.path.exists(exe):
        return True, ""

    if rc == -999:
        return False, "Compilation timed out."

    msg = (out.decode("utf-8", errors="replace") + "\n" + err.decode("utf-8", errors="replace")).strip()
    return False, msg if msg else "Compilation failed (no output). Check that GPP points to g++.exe."

async def run_one_test(workdir: str, t: Dict[str, Any]) -> Tuple[bool, str, Optional[dict]]:
    # Runs one hidden test and returns pass/fail + details for feedback.
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
    # Scheduled daily post (PH time).
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
    # Wait until the bot is connected before starting the scheduler.
    await bot.wait_until_ready()
    logging.info("CS1 Judge armed.")

@bot.event
async def on_ready():
    logging.info("Logged in as %s (id: %s)", bot.user, bot.user.id)
    if not post_daily_problem.is_running():
        post_daily_problem.start()

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    # Don’t spam chat on typos/unknown commands.
    if isinstance(error, commands.CommandNotFound):
        return
    await ctx.send(f"❌ {type(error).__name__}: {error}")

# =========================
# COMMANDS
# =========================
@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send("pong")
    
@bot.command(name="today")
async def today(ctx: commands.Context):
    # Re-sends today’s stored MP (so students can pull it again).
    state = load_state()
    date_str = today_str_ph()
    p = state.get("problems_by_date", {}).get(date_str)
    if not p:
        await ctx.send("❌ No problem stored for today yet. Ask admin to `!postnow` or wait for schedule.")
        return
    await ctx.send(embed=build_embed(p))

@bot.command(name="submit")
async def submit(ctx: commands.Context):
    # Main judge command: compile + run hidden tests.
    async with SUBMIT_LOCK:
        # Fix: allow if both IDs are the same
        if SUBMIT_CHANNEL_ID and ctx.channel.id != SUBMIT_CHANNEL_ID:
            if SUBMIT_CHANNEL_ID != DAILY_CHANNEL_ID:
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

        if re.search(r'cout\s*<<\s*".*enter', code, flags=re.IGNORECASE):
            await ctx.send("⚠️ Heads up: prompts like `Enter n:` usually cause Wrong Answer. Output should be answer only.")


# ✅ ENFORCE REQUIRED SKILL CATEGORY (heuristics)
ok_skill, skill_msg = enforce_skill(problem, code)
if not ok_skill:
    await ctx.send(skill_msg)
    return

        tests = problem.get("tests", [])
        status_msg = await ctx.send("🧪 Compiling...")

        with tempfile.TemporaryDirectory(prefix="cs1judge_") as workdir:
            ok, cerr = await compile_cpp(code, workdir)
            if not ok:
                cerr = cerr.strip()
                if len(cerr) > 1800:
                    cerr = cerr[:1800] + "\n... (truncated)"
                await status_msg.edit(content="❌ Compilation Error")
                await ctx.send(f"```text\n{cerr}\n```")
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
                        await ctx.send(
                            f"**Input**\n```text\n{tinp}```"
                            f"**Expected**\n```text\n{exp}```"
                            f"**Got**\n```text\n{got}```"
                        )
                    else:
                        if tinp:
                            await ctx.send(f"**Input**\n```text\n{tinp}```")
                    return

            await status_msg.edit(content=f"✅ Accepted — {len(tests)}/{len(tests)} tests passed.")
            await ctx.send(f"Problem: **{problem['title']}** (Day {problem['day']})")

@bot.command(name="chelp")
async def help(ctx: commands.Context):
    msg = """**COMMANDS:**
```
!submit
`\u200B`\u200B`\u200Bcpp
statements...
`\u200B`\u200B`\u200B
```→ Submit your code
`!today` → Resends today's machine problem
`!help` → Shows this guide
"""
    await ctx.send(msg)


# =========================
# DEV COMMANDS
# =========================
@bot.command(name="dev")
@commands.has_role("Root Admin")  # replace with actual role
async def dev(ctx: commands.Context, action: str, family: Optional[str] = None, kind: Optional[str] = None):
    """
    Combined dev command:
    - !dev pick <family> <kind> → picks today's problem manually
    - !dev pick_random [family] → picks random problem (optionally from a specific family)
    - !dev show_variants → lists all families + kinds
    - !dev submit → submits code for debug/testing (debug only)
    - !dev help → shows all dev commands
    """

    state = load_state()
    date_str = today_str_ph()

    action = action.lower()

    # --------------------
    # !dev setup
    # --------------------

    if action == "setup":
        # Quick sanity-check command for admins.
        ch = bot.get_channel(DAILY_CHANNEL_ID) if DAILY_CHANNEL_ID else None
        await ctx.send(
            f"✅ Bot online.\n"
            f"- Daily channel ID: `{DAILY_CHANNEL_ID}` (name: `{getattr(ch, 'name', None)}`)\n"
            f"- Submit channel ID: `{SUBMIT_CHANNEL_ID}` (0 means any channel)\n"
            f"- Post time: `{POST_TIME}` (PH time)\n"
            f"- Windows mode: `{IS_WINDOWS}`\n"
            f"- GPP: `{GPP}`"
        )

    # --------------------
    # !dev help
    # --------------------
    elif action == "help":
        msg = (
            "**DEV COMMANDS:**\n"
            "`!dev pick <family> <kind>` → Manually pick a problem\n"
            "`!dev random` → Picks random family & kind\n"
            "`!dev random <family>` → Picks random kind from specific family\n"
            "`!dev list` → Lists all machine problems\n"
            "`!dev submit` → Submits code for debug/testing\n"
            "`!dev setup` → Shows this bot's status"
            "`!dev help` → Shows this guide"
        )
        await ctx.send(msg)
        return

    # --------------------
    # !dev list
    # --------------------
    elif action == "list":
        msg = "**FAMILIES AND KINDS:**\n"
        total_count = 0
        for f, kinds in family_kinds.items():
            msg += f"- **{f}**: {', '.join(f'`{k}`' for k in kinds)}\n"
            total_count += len(kinds)  # add the number of elements in this family

        msg += f"\n**Total:** {total_count}"
        await ctx.send(msg)
        return

    # --------------------
    # !dev random [family]
    # --------------------
    elif action == "random":
        # Random family if not specified
        if family is None:
            family = random.choice(list(family_kinds.keys()))
        elif family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await ctx.send(f"❌ Invalid family '{family}'. Available families:\n{families_list}")
            return

        kind = random.choice(family_kinds[family])
        # Now reuse your pick logic
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

    # --------------------
    # !dev pick <family> <kind>
    # --------------------
    elif action == "pick":
        # Normalize input
        family = (family or "").lower().strip()
        kind = (kind or "").strip()

        # Case 1: Family missing or invalid
        if not family or family not in family_kinds:
            families_list = ", ".join(f"`{f}`" for f in family_kinds.keys())
            await ctx.send(f"❌ Invalid or missing family. Available families:\n{families_list}")
            return

        # Case 2: Kind missing or invalid
        if not kind or kind not in family_kinds[family]:
            kinds_list = ", ".join(f"`{k}`" for k in family_kinds[family])
            await ctx.send(f"❌ Invalid or missing kind for family `{family}`. Available kinds:\n{kinds_list}")
            return

        # Now we have valid family and kind, generate the problem
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

        # Stamp metadata and save state
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


    # --------------------
    # !dev submit
    # --------------------
    # Keep your existing submit code here (unchanged)
    elif action == "submit":
        # Only allowed in DAILY_CHANNEL_ID
        if ctx.channel.id != DAILY_CHANNEL_ID:
            await ctx.send(f"❌ `!dev submit` only works in <#{DAILY_CHANNEL_ID}>.")
            return

        # Only allowed for Root Admin role
        if "Root Admin" not in [r.name for r in ctx.author.roles]:
            await ctx.send("❌ You are not allowed to use `!dev submit`.")
            return

        msg: discord.Message = ctx.message
        code = await read_attachment_cpp(msg) or extract_cpp_from_message(msg.content)

        if not code:
            await ctx.send("❌ No C++ code found in message or attachment.")
            return

        problem = state.get("problems_by_date", {}).get(date_str)
        if not problem:
            await ctx.send("❌ No active problem for today. Pick one first with `!dev pick`.")
            return

        # ✅ ENFORCE REQUIRED SKILL CATEGORY (heuristics)
        ok_skill, skill_msg = enforce_skill(problem, code)
        if not ok_skill:
            await ctx.send('[DEV] ' + skill_msg)
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
                            await ctx.send(
                                f"**Input**\n```text\n{tinp}```"
                                f"**Expected**\n```text\n{exp}```"
                                f"**Got**\n```text\n{got}```"
                            )
                        else:
                            if tinp:
                                await ctx.send(f"**Input**\n```text\n{tinp}```")
                        return

                await status_msg.edit(content=f"✅ [DEV] Accepted — all tests passed.")
                await ctx.send(f"[DEV] Problem: **{problem['title']}** (Day {problem['day']})")

    else:
        await ctx.send("❌ Invalid `!dev` action. Use `pick`, `random`, `list`, `submit`, `setup`, or `help`.")


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
