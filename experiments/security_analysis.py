"""
Security Multi-Pass Experiment
================================
Compares vulnerability detection strategies with the same compute budget:
  A: 3x generic passes
  B: 3x specialized prompt (no contrast)
  C: 3x contrastive same-pair
  D: 3x specialized contrastive passes (different vulnerability classes)

10 known vulnerabilities scored by keyword matching. No API calls.

Run with:
    python experiments/security_analysis.py --model Qwen/Qwen2.5-3B-Instruct

Expected results:
    A: 8.3/10, B: 6.7/10, C: 5.3/10, D: 9.0/10
"""

import sys
import argparse
import json

import numpy as np
import pandas as pd

if not sys.stdout.isatty():
    import functools

    print = functools.partial(print, flush=True)  # noqa: A001

from prism.token_level import load_model, contrastive_generate

CODE = '''import threading
import hashlib
import time
import sqlite3
from functools import wraps

db_conn = sqlite3.connect("app.db", check_same_thread=False)
_cache = {}
_cache_lock = threading.Lock()
_user_sessions = {}

def rate_limit(max_calls=10, period=60):
    calls = []
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if now - c < period]
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_user(user_id):
    if user_id in _cache:
        return _cache[user_id]
    with _cache_lock:
        if user_id in _cache:
            return _cache[user_id]
        cursor = db_conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = " + str(user_id))
        user = cursor.fetchone()
        _cache[user_id] = user
        return user

def hash_password(password, salt=None):
    if salt is None:
        salt = str(time.time())
    return hashlib.md5(salt + password).hexdigest()

def create_session(user_id):
    token = hashlib.md5(str(time.time()) + str(user_id)).hexdigest()
    _user_sessions[token] = {
        "user_id": user_id,
        "created": time.time(),
        "expires": time.time() + 3600
    }
    return token

def validate_session(token):
    session = _user_sessions.get(token)
    if session and session["expires"] > time.time():
        return session["user_id"]
    return None

def get_user_data(token, requested_user_id):
    current_user_id = validate_session(token)
    if not current_user_id:
        return None
    user = get_user(requested_user_id)
    return user

@rate_limit(max_calls=10, period=60)
def login(username, password):
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = \\'" + username + "\\'"
    )
    user = cursor.fetchone()
    if user and hash_password(password, user[2]) == user[3]:
        return create_session(user[0])
    return None'''

PROMPT = f"Find all security vulnerabilities in this code:\n\n```python\n{CODE}\n```"

VULNS = {
    1: {"desc": "SQL injection login()", "terms": ["sql inject", "username", "login", "concatenat", "string format"]},
    2: {"desc": "SQL injection get_user()", "terms": ["sql inject", "user_id", "get_user", "concatenat", "str(user_id)"]},
    3: {"desc": "MD5 password hashing", "terms": ["md5", "weak hash", "broken hash", "cryptograph"]},
    4: {"desc": "Predictable session token", "terms": ["predictable", "session token", "time()", "guessable", "weak token"]},
    5: {"desc": "Predictable salt", "terms": ["predictable salt", "time()", "random salt", "weak salt", "salt is"]},
    6: {"desc": "Rate limit not per-user", "terms": ["per-user", "shared", "rate limit", "all users", "global calls", "single list"]},
    7: {"desc": "Rate limit not thread-safe", "terms": ["thread", "rate limit", "lock", "calls[:]", "concurrent rate", "race"]},
    8: {"desc": "IDOR get_user_data()", "terms": ["idor", "authorization", "access control", "requested_user", "current_user", "permission", "any user"]},
    9: {"desc": "Global sqlite connection", "terms": ["check_same_thread", "connection pool", "global connection", "sqlite thread", "shared connection"]},
    10: {"desc": "Lockless cache read", "terms": ["lockless", "cpython", "gil", "cache read", "before lock", "unprotected read", "outside the lock", "without lock", "first check"]},
}

GENERIC_SYS = "You are a helpful assistant."

SPECIALIZED_CREATIVE = (
    "You are a security researcher. Assume this code contains exploitable "
    "vulnerabilities. Find the specific gap between intended and actual "
    "behavior. Name the exact function, the specific input that triggers "
    "the vulnerability, and the outcome an attacker achieves."
)

SPECIALIZED_CONSERVATIVE = (
    "You are a developer explaining this code to a new team member. "
    "Explain how each security control works and what it protects."
)

D_PASSES = [
    {
        "creative": (
            "Find all thread-safety violations and race conditions. "
            "Consider concurrent execution: what happens when two threads execute "
            "simultaneously? Look for shared state accessed without locks, "
            "check-then-act patterns, and operations that appear atomic but aren't."
        ),
        "conservative": (
            "Explain the synchronization design. Where are locks used, what shared "
            "state do they protect, and why is the design safe for concurrent use?"
        ),
    },
    {
        "creative": (
            "Find all injection vulnerabilities and cryptographic weaknesses. "
            "Look for unsanitized input reaching SQL queries, weak or broken "
            "hash functions, predictable values used for tokens or salts."
        ),
        "conservative": (
            "Explain the input validation, authentication, and cryptographic design. "
            "How does the code safely handle user input and protect credentials?"
        ),
    },
    {
        "creative": (
            "Find authorization flaws, broken access control, and business logic "
            "errors. Look for missing permission checks and places where the code "
            "assumes something about the caller that an attacker can violate."
        ),
        "conservative": (
            "Explain the authorization model. Who is allowed to access what "
            "resources, and how does the code enforce those boundaries?"
        ),
    },
]

N_PIPELINE_REPS = 3


def score_output(text):
    text_lower = text.lower()
    return {vid: 1 if any(t in text_lower for t in v["terms"]) else 0 for vid, v in VULNS.items()}


def union_scores(score_list):
    union = {i: 0 for i in range(1, 11)}
    for s in score_list:
        for vid, found in s.items():
            if found:
                union[vid] = 1
    return union


def run_experiment(model, tokenizer):
    all_results = {"A": [], "B": [], "C": [], "D": []}
    all_raw = []
    total_gens = N_PIPELINE_REPS * 12
    done = 0

    for rep in range(1, N_PIPELINE_REPS + 1):
        print(f"\n{'=' * 60}")
        print(f"Pipeline repetition {rep}/{N_PIPELINE_REPS}")

        # A: 3x generic
        a_scores = []
        for i in range(3):
            done += 1
            print(f"  A pass {i + 1} ({done}/{total_gens})", end=" ")
            r = contrastive_generate(model, tokenizer, problem=PROMPT, alpha=0.0,
                                     max_new_tokens=600, creative_system=GENERIC_SYS,
                                     conservative_system=GENERIC_SYS)
            s = score_output(r.text)
            a_scores.append(s)
            print(f"found={sum(s.values())}")
            all_raw.append(dict(condition="A", rep=rep, pass_num=i + 1, text=r.text,
                                vulns_found=sum(s.values())))
        all_results["A"].append(union_scores(a_scores))

        # B: 3x specialized, no contrastive
        b_scores = []
        for i in range(3):
            done += 1
            print(f"  B pass {i + 1} ({done}/{total_gens})", end=" ")
            r = contrastive_generate(model, tokenizer, problem=PROMPT, alpha=0.0,
                                     max_new_tokens=600, creative_system=SPECIALIZED_CREATIVE,
                                     conservative_system=SPECIALIZED_CONSERVATIVE)
            s = score_output(r.text)
            b_scores.append(s)
            print(f"found={sum(s.values())}")
            all_raw.append(dict(condition="B", rep=rep, pass_num=i + 1, text=r.text,
                                vulns_found=sum(s.values())))
        all_results["B"].append(union_scores(b_scores))

        # C: 3x contrastive, same pair
        c_scores = []
        for i in range(3):
            done += 1
            print(f"  C pass {i + 1} ({done}/{total_gens})", end=" ")
            r = contrastive_generate(model, tokenizer, problem=PROMPT, alpha=1.0,
                                     max_new_tokens=600, creative_system=SPECIALIZED_CREATIVE,
                                     conservative_system=SPECIALIZED_CONSERVATIVE)
            s = score_output(r.text)
            c_scores.append(s)
            print(f"found={sum(s.values())}")
            all_raw.append(dict(condition="C", rep=rep, pass_num=i + 1, text=r.text,
                                vulns_found=sum(s.values())))
        all_results["C"].append(union_scores(c_scores))

        # D: 3x specialized contrastive passes
        d_scores = []
        for i, dp in enumerate(D_PASSES):
            done += 1
            print(f"  D pass {i + 1} ({done}/{total_gens})", end=" ")
            r = contrastive_generate(model, tokenizer, problem=PROMPT, alpha=1.0,
                                     max_new_tokens=600, creative_system=dp["creative"],
                                     conservative_system=dp["conservative"])
            s = score_output(r.text)
            d_scores.append(s)
            print(f"found={sum(s.values())}")
            all_raw.append(dict(condition="D", rep=rep, pass_num=i + 1, text=r.text,
                                vulns_found=sum(s.values())))
        all_results["D"].append(union_scores(d_scores))

    analyze(all_results)


def analyze(all_results):
    from scipy import stats

    print(f"\n{'=' * 60}")
    print("RESULTS: Vulnerabilities found /10 (union of 3 passes)")
    print(f"{'=' * 60}")

    descs = {"A": "3x generic", "B": "3x specialized (no contrastive)",
             "C": "3x contrastive same-pair", "D": "3x specialized contrastive"}

    totals = {}
    for cond in "ABCD":
        vals = [sum(r.values()) for r in all_results[cond]]
        totals[cond] = vals
        print(f"  {descs[cond]:<40s} {np.mean(vals):>5.1f}  {vals}")

    _, p_db = stats.mannwhitneyu(totals["D"], totals["B"], alternative="greater")
    _, p_da = stats.mannwhitneyu(totals["D"], totals["A"], alternative="greater")
    print(f"\n  D vs B: p={p_db:.4f}, D vs A: p={p_da:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    run_experiment(model, tokenizer)
