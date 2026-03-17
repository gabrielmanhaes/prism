"""
Security Vulnerability Detection — Multi-pass architecture.

Uses three specialized contrastive passes targeting different
vulnerability classes, then unions the results.
"""

from prism import TokenLevelPRISM, PRISMConfig, get_mode

prism = TokenLevelPRISM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    config=PRISMConfig(max_new_tokens=600),
)

code = '''
def get_user(user_id):
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + str(user_id))
    return cursor.fetchone()

def hash_password(password, salt=None):
    if salt is None:
        salt = str(time.time())
    return hashlib.md5(salt + password).hexdigest()

def get_user_data(token, requested_user_id):
    current_user_id = validate_session(token)
    if not current_user_id:
        return None
    return get_user(requested_user_id)
'''

problem = f"Find all security vulnerabilities in this code:\n\n```python\n{code}\n```"

# Three specialized passes
security_modes = ["security_injection", "security_authorization", "security_threading"]

all_findings = []
for mode_name in security_modes:
    mode = get_mode(mode_name)
    print(f"\n{'=' * 60}")
    print(f"Pass: {mode.name} (alpha={mode.optimal_alpha})")
    print("=" * 60)

    result = prism.generate(
        problem=problem,
        creative_system=mode.creative_system,
        conservative_system=mode.conservative_system,
        alpha=mode.optimal_alpha,
    )
    print(result.text[:500])
    all_findings.append(result.text)
