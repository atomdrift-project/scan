#!/usr/bin/env python3
"""
Extract 10K features from samples using cleave.

This script:
1. Runs cleave on each sample file
2. Extracts ~10,000 features from the JSON output
3. Saves features to a compressed numpy file

Usage:
    python extract_features.py --malicious ../samples --benign ./benign_samples --output features.npz
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Known capability IDs for one-hot encoding (will be populated from training data)
KNOWN_CAPABILITIES: list[str] = []

# Known import symbols for one-hot encoding
KNOWN_IMPORTS: list[str] = []

# Feature name registry
FEATURE_NAMES: list[str] = []

# Known file types for one-hot encoding and byte baseline normalization
KNOWN_FILE_TYPES = [
    "python_script",
    "javascript",
    "shell_script",
    "elf",
    "macho",
    "pe",
    "go",
    "java_class",
    "unknown",
]

# Byte distribution baselines per file type (computed from training data)
# Format: {file_type: np.ndarray of shape (256,) with expected frequencies}
BYTE_BASELINES: dict[str, np.ndarray] = {}


def expand_capability_hierarchy(cap_id: str) -> set[str]:
    """Expand a capability ID to include parent categories.

    For example, 'comms/socket/connect-xyz' expands to:
      - 'comms/socket/connect-xyz' (original)
      - 'comms/socket' (1 level up)
      - 'comms' (2 levels up)

    This allows the model to learn both specific and general patterns.
    """
    if not cap_id:
        return set()

    result = {cap_id}
    parts = cap_id.split("/")

    # Add parent levels (up to 2 levels)
    if len(parts) >= 2:
        result.add("/".join(parts[:-1]))  # 1 level up
    if len(parts) >= 3:
        result.add("/".join(parts[:-2]))  # 2 levels up

    return result


# Global cache directory for cleave reports
CACHE_DIR: Optional[Path] = None

# cleave source directory (needed to find traits/)
CLEAVE_DIR: Optional[Path] = None


def get_cache_path(file_path: Path) -> Optional[Path]:
    """Get cache path for a file's cleave report."""
    if CACHE_DIR is None:
        return None
    # Use hash of absolute path as cache key
    import hashlib
    path_hash = hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{path_hash}.json"


def run_cleave(file_path: Path, verbose: bool = False) -> tuple[Optional[dict], int, str]:
    """Run cleave and return (report, exit_code, error_msg). Uses cache if available."""
    # Check cache first
    cache_path = get_cache_path(file_path)
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("error"):
                return None, cached.get("exit_code", -1), cached["error"]
            return cached.get("report"), 0, ""
        except Exception:
            pass  # Cache miss, run cleave

    # cleave needs to run from its source dir to find traits/
    cwd = CLEAVE_DIR
    if cwd:
        cmd = ["./target/release/cleave", "--json", str(file_path.resolve())]
    else:
        cmd = ["cleave", "--json", str(file_path)]
        cwd = None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per file
            cwd=cwd,
        )
        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[0] if result.stderr else "unknown error"
            # Cache the error
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({"error": err, "exit_code": result.returncode}, f)
            return None, result.returncode, err
        # Parse JSON array output directly (--json outputs clean JSON to stdout)
        if not result.stdout.strip():
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({"error": "empty output"}, f)
            return None, 0, "empty output"
        reports = json.loads(result.stdout)
        if not reports:
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({"error": "no reports"}, f)
            return None, 0, "no reports"
        # Cache successful result
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"report": reports[0]}, f)
        return reports[0], 0, ""
    except subprocess.TimeoutExpired:
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"error": "timeout", "exit_code": -1}, f)
        return None, -1, "timeout"
    except json.JSONDecodeError as e:
        return None, 0, f"json: {e}"
    except Exception as e:
        return None, -2, str(e)


def get_file_type(report: dict) -> str:
    """Extract normalized file type from cleave report."""
    raw_type = report.get("target", {}).get("type", "unknown")
    # Normalize to known types
    if raw_type in KNOWN_FILE_TYPES:
        return raw_type
    # Map variations
    type_map = {
        "python": "python_script",
        "js": "javascript",
        "bash": "shell_script",
        "sh": "shell_script",
        "zsh": "shell_script",
        "macho64": "macho",
        "elf64": "elf",
        "exe": "pe",
        "dll": "pe",
    }
    return type_map.get(raw_type, "unknown")


def collect_vocabulary(samples: list[Path], label: str, verbose: bool = False) -> tuple[set, set, dict, dict]:
    """First pass: collect all capability IDs, imports, and byte distributions from samples."""
    capabilities = set()
    imports = set()
    stats = {"ok": 0, "fail": 0, "errors": {}}

    # Byte distribution accumulators per file type
    byte_accum = {ft: np.zeros(256, dtype=np.float64) for ft in KNOWN_FILE_TYPES}
    byte_counts = {ft: 0 for ft in KNOWN_FILE_TYPES}

    print(f"Collecting vocabulary from {label} samples...")
    for file_path in tqdm(samples, desc=f"Scanning {label}", disable=verbose):
        report, exit_code, err = run_cleave(file_path, verbose)
        if report is None:
            stats["fail"] += 1
            err_key = err[:50] if err else "unknown"
            stats["errors"][err_key] = stats["errors"].get(err_key, 0) + 1
            if verbose:
                print(f"  FAIL: {file_path.name} (exit={exit_code}, {err})")
            continue

        stats["ok"] += 1
        n_findings = len(report.get("findings", []))
        n_imports = len(report.get("imports", []))
        if verbose:
            print(f"  OK: {file_path.name} findings={n_findings} imports={n_imports}")

        # Collect capability IDs (with hierarchical expansion)
        for finding in report.get("findings", []):
            cap_id = finding.get("id", "")
            capabilities.update(expand_capability_hierarchy(cap_id))

        # Collect import symbols
        for imp in report.get("imports", []):
            imports.add(imp.get("symbol", ""))

        # Accumulate byte distribution for this file type
        file_type = get_file_type(report)
        try:
            with open(file_path, "rb") as f:
                data = f.read(65536)
            if len(data) > 0:
                hist = np.zeros(256, dtype=np.float64)
                for b in data:
                    hist[b] += 1
                hist /= len(data)  # Normalize to frequencies
                byte_accum[file_type] += hist
                byte_counts[file_type] += 1
        except Exception:
            pass

    # Compute average byte distributions per file type
    byte_baselines = {}
    for ft in KNOWN_FILE_TYPES:
        if byte_counts[ft] > 0:
            byte_baselines[ft] = byte_accum[ft] / byte_counts[ft]
        else:
            # Use uniform distribution as fallback
            byte_baselines[ft] = np.ones(256, dtype=np.float64) / 256

    return capabilities, imports, stats, byte_baselines


def extract_features(report: dict, file_path: Path) -> np.ndarray:
    """Convert cleave JSON to feature vector."""
    features = []

    # === 1. Capability presence (~500 features) ===
    # Expand each finding's capability ID to include parent categories
    present_caps = set()
    for f in report.get("findings", []):
        present_caps.update(expand_capability_hierarchy(f.get("id", "")))
    for cap_id in KNOWN_CAPABILITIES:
        features.append(1.0 if cap_id in present_caps else 0.0)

    # === 2. Criticality counts (7 features) ===
    criticality_counts = Counter(
        f.get("crit", f.get("criticality", "inert")) for f in report.get("findings", [])
    )
    features.append(float(criticality_counts.get("hostile", 0)))
    features.append(float(criticality_counts.get("suspicious", 0)))
    features.append(float(criticality_counts.get("notable", 0)))
    features.append(float(criticality_counts.get("inert", 0)))
    features.append(float(len(report.get("findings", []))))  # total findings

    # Ratios
    total = max(len(report.get("findings", [])), 1)
    features.append(criticality_counts.get("hostile", 0) / total)
    features.append(criticality_counts.get("suspicious", 0) / total)

    # === 3. Import one-hot encoding (~2000 features) ===
    present_imports = {i.get("symbol", "") for i in report.get("imports", [])}
    for imp in KNOWN_IMPORTS:
        features.append(1.0 if imp in present_imports else 0.0)

    # Import count
    features.append(float(len(report.get("imports", []))))

    # === 4. String features (~20 features) ===
    strings = report.get("strings", [])
    features.append(float(len(strings)))  # total string count

    if strings:
        lengths = [len(s.get("value", "")) for s in strings]
        features.append(float(max(lengths)))  # max length
        features.append(float(sum(lengths) / len(lengths)))  # avg length
    else:
        features.append(0.0)
        features.append(0.0)

    # Trait-based string counts
    traits = report.get("traits", [])
    url_count = sum(1 for t in traits if t.get("kind") == "url")
    ip_count = sum(1 for t in traits if t.get("kind") == "ip")
    domain_count = sum(1 for t in traits if t.get("kind") == "domain")
    path_count = sum(1 for t in traits if t.get("kind") == "path")
    base64_count = sum(1 for t in traits if t.get("kind") == "base64")

    features.append(float(url_count))
    features.append(float(ip_count))
    features.append(float(domain_count))
    features.append(float(path_count))
    features.append(float(base64_count))

    # === 5. Text/code metrics (~20 features) ===
    metrics = report.get("metrics", {})
    text = metrics.get("text", {})

    features.append(float(text.get("char_entropy", 0)))
    features.append(float(text.get("avg_line_length", 0)))
    features.append(float(text.get("max_line_length", 0)))
    features.append(float(text.get("non_ascii_ratio", 0)))
    features.append(float(text.get("hex_escape_count", 0)))
    features.append(float(text.get("unicode_escape_count", 0)))
    features.append(float(text.get("total_lines", 0)))
    features.append(float(text.get("whitespace_ratio", 0)))
    features.append(float(text.get("digit_ratio", 0)))
    features.append(float(text.get("repeated_char_sequences", 0)))
    features.append(float(text.get("trailing_whitespace_lines", 0)))

    # === 6. Identifier metrics (obfuscation detection) ===
    idents = metrics.get("identifiers", {})
    features.append(float(idents.get("total", 0)))
    features.append(float(idents.get("unique", 0)))
    features.append(float(idents.get("reuse_ratio", 0)))
    features.append(float(idents.get("avg_length", 0)))
    features.append(float(idents.get("single_char_count", 0)))
    features.append(float(idents.get("single_char_ratio", 0)))  # HIGH = obfuscation
    features.append(float(idents.get("avg_entropy", 0)))
    features.append(float(idents.get("high_entropy_count", 0)))
    features.append(float(idents.get("high_entropy_ratio", 0)))  # HIGH = obfuscation
    features.append(float(idents.get("all_uppercase_ratio", 0)))
    features.append(float(idents.get("underscore_prefix_count", 0)))
    features.append(float(idents.get("double_underscore_count", 0)))
    features.append(float(idents.get("sequential_names", 0)))  # x1, x2, x3 pattern
    features.append(float(idents.get("has_digit_ratio", 0)))
    features.append(float(idents.get("numeric_suffix_count", 0)))

    # === 7. String metrics from cleave ===
    str_metrics = metrics.get("strings", {})
    features.append(float(str_metrics.get("total", 0)))
    features.append(float(str_metrics.get("total_bytes", 0)))
    features.append(float(str_metrics.get("avg_length", 0)))
    features.append(float(str_metrics.get("max_length", 0)))
    features.append(float(str_metrics.get("avg_entropy", 0)))
    features.append(float(str_metrics.get("entropy_stddev", 0)))
    features.append(float(str_metrics.get("shell_command_strings", 0)))  # VERY suspicious
    features.append(float(str_metrics.get("base64_candidates", 0)))  # Encoded payloads
    features.append(float(str_metrics.get("path_count", 0)))

    # === 8. Comment metrics (malware has few comments) ===
    comments = metrics.get("comments", {})
    features.append(float(comments.get("total", 0)))
    features.append(float(comments.get("lines", 0)))
    features.append(float(comments.get("to_code_ratio", 0)))  # LOW = suspicious

    # === 9. Function metrics ===
    funcs = metrics.get("functions", {})
    features.append(float(funcs.get("total", 0)))
    features.append(float(funcs.get("anonymous", 0)))  # Obfuscation indicator
    features.append(float(funcs.get("async_count", 0)))  # Network operations
    features.append(float(funcs.get("avg_length_lines", 0)))
    features.append(float(funcs.get("max_length_lines", 0)))
    features.append(float(funcs.get("one_liners", 0)))  # Packed code
    features.append(float(funcs.get("avg_params", 0)))
    features.append(float(funcs.get("no_params_count", 0)))
    features.append(float(funcs.get("single_char_params", 0)))  # Obfuscation
    features.append(float(funcs.get("max_nesting_depth", 0)))
    features.append(float(funcs.get("avg_nesting_depth", 0)))
    features.append(float(funcs.get("nested_functions", 0)))  # Closures/obfuscation
    features.append(float(funcs.get("density_per_100_lines", 0)))
    features.append(float(funcs.get("code_in_functions_ratio", 0)))

    # === 10. Language-specific metrics ===
    # Python
    py_metrics = metrics.get("python", {})
    features.append(float(py_metrics.get("try_except_count", 0)))  # Error suppression

    # JavaScript
    js_metrics = metrics.get("javascript", {})
    features.append(float(js_metrics.get("arrow_function_count", 0)))

    # === 11. Legacy source code metrics (may overlap) ===
    scm = report.get("source_code_metrics", {})
    features.append(float(scm.get("function_count", 0)))
    features.append(float(scm.get("dynamic_execution_count", 0)))
    features.append(float(scm.get("obfuscation_indicators", 0)))
    features.append(float(scm.get("high_entropy_strings", 0)))
    features.append(float(scm.get("total_lines", 0)))
    features.append(float(scm.get("code_lines", 0)))
    features.append(float(scm.get("comment_lines", 0)))
    features.append(float(scm.get("avg_function_size", 0)))
    features.append(float(scm.get("max_function_size", 0)))
    features.append(float(scm.get("string_count", 0)))
    features.append(float(scm.get("avg_string_entropy", 0)))

    # === 12. Suspicious function name detection ===
    func_names = [f.get("name", "").lower() for f in report.get("functions", [])]
    suspicious_keywords = [
        "inject", "shell", "payload", "execute", "exec", "download", "upload",
        "encrypt", "decrypt", "exfil", "backdoor", "keylog", "hook", "dump",
        "steal", "capture", "reverse", "bind", "connect", "socket", "spawn",
        "persist", "hide", "obfuscate", "encode", "decode", "base64", "eval",
        "credential", "password", "token", "cookie", "browser", "chrome", "firefox"
    ]
    suspicious_func_count = sum(
        1 for name in func_names
        for kw in suspicious_keywords
        if kw in name
    )
    features.append(float(suspicious_func_count))

    # === 13. Supply-chain attack indicators (from raw file content) ===
    try:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
        content_lower = content.lower()
    except Exception:
        content = ""
        content_lower = ""

    # Credential/secret file paths (SSH, cloud, package managers)
    sensitive_paths = [
        ".ssh/", "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
        ".aws/credentials", ".aws/config", ".azure/", ".config/gcloud",
        ".npmrc", ".pypirc", ".gem/credentials", ".docker/config",
        ".kube/config", ".terraform", ".vault-token",
        ".gnupg/", ".pgp", "private_key", "privkey",
    ]
    sensitive_path_count = sum(1 for p in sensitive_paths if p in content_lower)
    features.append(float(sensitive_path_count))

    # Browser credential paths
    browser_paths = [
        "chrome", "chromium", "firefox", "mozilla", "safari", "edge", "opera", "brave",
        "local state", "login data", "cookies", "web data",
        "application support/google", "appdata/local/google",
        "appdata/roaming/mozilla", ".config/google-chrome",
    ]
    browser_path_count = sum(1 for p in browser_paths if p in content_lower)
    features.append(float(browser_path_count))

    # Exfiltration endpoints (webhooks, paste services)
    exfil_patterns = [
        "discord.com/api/webhook", "discord.gg", "discordapp.com",
        "telegram.org", "api.telegram", "t.me/",
        "pastebin.com", "hastebin", "ghostbin", "paste.ee", "dpaste",
        "ngrok.io", "serveo.net", "localhost.run",
        "transfer.sh", "file.io", "0x0.st",
    ]
    exfil_count = sum(1 for p in exfil_patterns if p in content_lower)
    features.append(float(exfil_count))

    # Hardcoded IPs (non-private, potential C2)
    import re
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ips = re.findall(ip_pattern, content)
    # Filter out private/localhost IPs
    public_ips = [ip for ip in ips if not (
        ip.startswith("127.") or ip.startswith("10.") or
        ip.startswith("192.168.") or ip.startswith("172.16.") or
        ip.startswith("0.") or ip == "255.255.255.255"
    )]
    features.append(float(len(public_ips)))
    features.append(float(len(set(public_ips))))  # unique public IPs

    # Environment variable access (secret stealing)
    env_patterns = [
        "os.environ", "os.getenv", "process.env", "env[",
        "getenv(", "environ.get", "environment.",
    ]
    env_access_count = sum(1 for p in env_patterns if p in content_lower)
    features.append(float(env_access_count))

    # Specific sensitive env vars
    sensitive_env_vars = [
        "api_key", "apikey", "api-key", "secret", "token", "password", "passwd",
        "aws_access", "aws_secret", "github_token", "npm_token", "pypi_token",
        "discord_token", "slack_token", "telegram_token",
        "private_key", "encryption_key", "signing_key",
        "database_url", "db_password", "redis_url", "mongodb_uri",
    ]
    sensitive_env_count = sum(1 for v in sensitive_env_vars if v in content_lower)
    features.append(float(sensitive_env_count))

    # Crypto wallet patterns (ransomware, cryptominers)
    crypto_patterns = [
        r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin
        r'\b0x[a-fA-F0-9]{40}\b',  # Ethereum
        r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b',  # Monero
    ]
    crypto_count = sum(len(re.findall(p, content)) for p in crypto_patterns)
    features.append(float(min(crypto_count, 10)))  # cap at 10

    # Install hook indicators (setup.py, package.json scripts)
    install_hooks = [
        "cmdclass", "install_requires", "setup(", "setuptools",
        "postinstall", "preinstall", "prepare", "build:",
        "__import__", "exec(", "eval(", "compile(",
        "subprocess.run", "subprocess.popen", "subprocess.call",
        "os.system(", "os.popen(",
    ]
    install_hook_count = sum(1 for h in install_hooks if h in content_lower)
    features.append(float(install_hook_count))

    # Data exfiltration methods
    exfil_methods = [
        "requests.post", "requests.put", "urllib.request",
        "http.client", "httplib", "aiohttp",
        "socket.connect", "socket.send",
        "ftplib", "smtplib", "paramiko",
        "base64.b64encode", "codecs.encode", "binascii",
    ]
    exfil_method_count = sum(1 for m in exfil_methods if m in content_lower)
    features.append(float(exfil_method_count))

    # Obfuscation indicators in content
    obfuscation_patterns = [
        "\\x", "\\u00", "chr(", "ord(",
        "lambda", "map(", "reduce(",
        "getattr(", "setattr(", "__dict__",
        "globals()", "locals()", "vars(",
        "importlib", "__import__", "exec(",
    ]
    obfuscation_count = sum(content_lower.count(p) for p in obfuscation_patterns)
    features.append(float(min(obfuscation_count, 100)))  # cap

    # Persistence mechanisms
    persistence_patterns = [
        "crontab", "cron.d", "/etc/rc", "systemd", "launchd", "plist",
        "registry", "hkey_", "run\\", "runonce",
        "startup", "autostart", ".bashrc", ".zshrc", ".profile",
        "scheduled task", "at job",
    ]
    persistence_count = sum(1 for p in persistence_patterns if p in content_lower)
    features.append(float(persistence_count))

    # Anti-analysis / evasion
    evasion_patterns = [
        "virtualbox", "vmware", "qemu", "sandbox", "malware",
        "wireshark", "fiddler", "burp", "debugger", "ida",
        "isdebuggerpresent", "ptrace", "strace",
        "time.sleep", "sleep(", "timeout",
    ]
    evasion_count = sum(1 for p in evasion_patterns if p in content_lower)
    features.append(float(evasion_count))

    # Network data collection (supply-chain exfil)
    data_collection = [
        "gethostname", "getfqdn", "platform.node", "socket.gethostname",
        "platform.system", "platform.release", "platform.machine",
        "getpass.getuser", "os.getlogin", "pwd.getpwuid",
        "uuid.getnode", "get_mac", "netifaces",
        "psutil", "wmi", "subprocess.check_output",
    ]
    data_collection_count = sum(1 for p in data_collection if p in content_lower)
    features.append(float(data_collection_count))

    # Known malicious package patterns (typosquatting indicators)
    # Long random-looking package names, version checks, etc.
    suspicious_imports = [
        "colorama", "requests", "urllib3",  # commonly typosquatted
        "__builtins__", "__code__", "__globals__",
        "marshal.loads", "types.FunctionType", "types.CodeType",
    ]
    suspicious_import_count = sum(1 for p in suspicious_imports if p in content_lower)
    features.append(float(suspicious_import_count))

    # Steganography / hidden data patterns
    stego_patterns = [
        "from_bytes", "to_bytes", "struct.unpack", "struct.pack",
        "zlib.decompress", "gzip.decompress", "lzma.decompress", "bz2.decompress",
        "PIL", "Image.open", "png", "jpg", "jpeg", "bmp",
    ]
    stego_count = sum(1 for p in stego_patterns if p in content_lower)
    features.append(float(stego_count))

    # Code injection patterns
    injection_patterns = [
        "ctypes", "windll", "cdll", "CFUNCTYPE",
        "mmap", "PROT_EXEC", "PAGE_EXECUTE",
        "VirtualAlloc", "WriteProcessMemory", "CreateRemoteThread",
        "dlopen", "dlsym", "LD_PRELOAD",
    ]
    injection_count = sum(1 for p in injection_patterns if p in content_lower)
    features.append(float(injection_count))

    # Keylogging / input capture
    keylog_patterns = [
        "pynput", "keyboard", "GetAsyncKeyState", "SetWindowsHookEx",
        "pyautogui", "pyscreenshot", "mss", "screenshot",
        "clipboard", "pyperclip", "win32clipboard",
    ]
    keylog_count = sum(1 for p in keylog_patterns if p in content_lower)
    features.append(float(keylog_count))

    # Suspicious URL patterns in strings
    suspicious_urls = content_lower.count("http://") + content_lower.count("https://")
    raw_ip_urls = len(re.findall(r'https?://\d+\.\d+\.\d+\.\d+', content_lower))
    features.append(float(suspicious_urls))
    features.append(float(raw_ip_urls))  # URLs with raw IPs are suspicious

    # Base64 encoded content (potential hidden payloads)
    b64_pattern = r'[A-Za-z0-9+/]{40,}={0,2}'
    b64_strings = re.findall(b64_pattern, content)
    long_b64_count = sum(1 for s in b64_strings if len(s) > 100)
    features.append(float(len(b64_strings)))
    features.append(float(long_b64_count))

    # === 7. Binary properties (~10 features) ===
    bp = report.get("binary_properties", {})
    security = bp.get("security", {})

    features.append(1.0 if security.get("canary", False) else 0.0)
    features.append(1.0 if security.get("nx", False) else 0.0)
    features.append(1.0 if security.get("pic", False) else 0.0)
    features.append(1.0 if security.get("stripped", False) else 0.0)
    features.append(1.0 if security.get("uses_crypto", False) else 0.0)

    # === 8. Structural features (~10 features) ===
    features.append(float(len(report.get("structure", []))))
    features.append(float(len(report.get("sections", []))))
    features.append(float(len(report.get("functions", []))))
    features.append(float(len(report.get("syscalls", []))))
    features.append(float(len(report.get("paths", []))))
    features.append(float(len(report.get("env_vars", []))))

    # File size (log-normalized)
    size = report.get("target", {}).get("size_bytes", 0)
    features.append(np.log1p(size))

    # === 9. File type one-hot encoding (len(KNOWN_FILE_TYPES) features) ===
    file_type = get_file_type(report)
    for ft in KNOWN_FILE_TYPES:
        features.append(1.0 if ft == file_type else 0.0)

    # === 10. Byte histogram anomaly scores (256 features) ===
    # Compute deviation from expected distribution for this file type
    try:
        with open(file_path, "rb") as f:
            data = f.read(65536)  # First 64KB
        byte_hist = np.zeros(256, dtype=np.float64)
        for b in data:
            byte_hist[b] += 1
        if len(data) > 0:
            byte_hist /= len(data)

        # Get baseline for this file type
        baseline = BYTE_BASELINES.get(file_type)
        if baseline is not None:
            # Compute signed deviation (positive = more than expected)
            # Use log-ratio for better scale: log(observed / expected)
            # Add small epsilon to avoid log(0)
            eps = 1e-6
            anomaly = np.log((byte_hist + eps) / (baseline + eps))
            features.extend(anomaly.tolist())
        else:
            # No baseline available, use raw frequencies
            features.extend(byte_hist.tolist())
    except Exception:
        features.extend([0.0] * 256)

    # === 11. Aggregate byte anomaly stats (4 features) ===
    # Summary stats of byte anomalies for quick signal
    try:
        if baseline is not None:
            anomaly_abs = np.abs(anomaly)
            features.append(float(np.mean(anomaly_abs)))  # Mean absolute anomaly
            features.append(float(np.max(anomaly_abs)))   # Max absolute anomaly
            features.append(float(np.sum(anomaly > 1.0))) # Count of highly overrepresented bytes
            features.append(float(np.sum(anomaly < -1.0))) # Count of highly underrepresented bytes
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def build_feature_names() -> list[str]:
    """Build list of feature names for explainability."""
    names = []

    # Capability presence
    for cap_id in KNOWN_CAPABILITIES:
        names.append(f"cap_{cap_id.replace('/', '_')}")

    # Criticality counts
    names.extend([
        "crit_hostile_count",
        "crit_suspicious_count",
        "crit_notable_count",
        "crit_inert_count",
        "crit_total_findings",
        "crit_hostile_ratio",
        "crit_suspicious_ratio",
    ])

    # Import one-hot
    for imp in KNOWN_IMPORTS:
        names.append(f"import_{imp}")
    names.append("import_total_count")

    # String features
    names.extend([
        "string_total_count",
        "string_max_length",
        "string_avg_length",
        "string_url_count",
        "string_ip_count",
        "string_domain_count",
        "string_path_count",
        "string_base64_count",
    ])

    # Text metrics
    names.extend([
        "metric_char_entropy",
        "metric_avg_line_length",
        "metric_max_line_length",
        "metric_non_ascii_ratio",
        "metric_hex_escape_count",
        "metric_unicode_escape_count",
        "metric_total_lines",
        "metric_whitespace_ratio",
        "metric_digit_ratio",
        "metric_repeated_char_sequences",
        "metric_trailing_whitespace_lines",
    ])

    # Identifier metrics (obfuscation detection)
    names.extend([
        "ident_total",
        "ident_unique",
        "ident_reuse_ratio",
        "ident_avg_length",
        "ident_single_char_count",
        "ident_single_char_ratio",
        "ident_avg_entropy",
        "ident_high_entropy_count",
        "ident_high_entropy_ratio",
        "ident_all_uppercase_ratio",
        "ident_underscore_prefix_count",
        "ident_double_underscore_count",
        "ident_sequential_names",
        "ident_has_digit_ratio",
        "ident_numeric_suffix_count",
    ])

    # String metrics from cleave
    names.extend([
        "str_total",
        "str_total_bytes",
        "str_avg_length",
        "str_max_length",
        "str_avg_entropy",
        "str_entropy_stddev",
        "str_shell_command_strings",
        "str_base64_candidates",
        "str_path_count",
    ])

    # Comment metrics
    names.extend([
        "comment_total",
        "comment_lines",
        "comment_to_code_ratio",
    ])

    # Function metrics
    names.extend([
        "func_total",
        "func_anonymous",
        "func_async_count",
        "func_avg_length_lines",
        "func_max_length_lines",
        "func_one_liners",
        "func_avg_params",
        "func_no_params_count",
        "func_single_char_params",
        "func_max_nesting_depth",
        "func_avg_nesting_depth",
        "func_nested_functions",
        "func_density_per_100_lines",
        "func_code_in_functions_ratio",
    ])

    # Language-specific
    names.extend([
        "lang_python_try_except_count",
        "lang_js_arrow_function_count",
    ])

    # Legacy source code metrics
    names.extend([
        "scm_function_count",
        "scm_dynamic_exec_count",
        "scm_obfuscation_indicators",
        "scm_high_entropy_strings",
        "scm_total_lines",
        "scm_code_lines",
        "scm_comment_lines",
        "scm_avg_function_size",
        "scm_max_function_size",
        "scm_string_count",
        "scm_avg_string_entropy",
    ])

    # Suspicious function names
    names.append("suspicious_func_name_count")

    # Supply-chain attack indicators
    names.extend([
        "sc_sensitive_path_count",      # SSH, AWS, cloud creds
        "sc_browser_path_count",        # Browser credential paths
        "sc_exfil_endpoint_count",      # Discord, Telegram, pastebin
        "sc_public_ip_count",           # Hardcoded public IPs
        "sc_unique_public_ip_count",    # Unique public IPs
        "sc_env_access_count",          # os.environ, process.env
        "sc_sensitive_env_var_count",   # API_KEY, TOKEN, etc.
        "sc_crypto_wallet_count",       # BTC, ETH, XMR addresses
        "sc_install_hook_count",        # setup.py, exec, subprocess
        "sc_exfil_method_count",        # requests.post, socket.send
        "sc_obfuscation_count",         # chr(), getattr, __import__
        "sc_persistence_count",         # crontab, systemd, registry
        "sc_evasion_count",             # VM detection, sleep, debugger
        "sc_data_collection_count",     # hostname, platform, user info
        "sc_suspicious_import_count",   # typosquatting, __builtins__
        "sc_stego_count",               # hidden data, compression
        "sc_injection_count",           # ctypes, mmap, dlopen
        "sc_keylog_count",              # pynput, screenshot, clipboard
        "sc_url_count",                 # http:// https:// in code
        "sc_raw_ip_url_count",          # URLs with raw IPs
        "sc_base64_string_count",       # Base64 encoded strings
        "sc_long_base64_count",         # Long base64 (payloads)
    ])

    # Binary properties
    names.extend([
        "binary_has_canary",
        "binary_has_nx",
        "binary_has_pic",
        "binary_is_stripped",
        "binary_uses_crypto",
    ])

    # Structural features
    names.extend([
        "struct_feature_count",
        "struct_section_count",
        "struct_function_count",
        "struct_syscall_count",
        "struct_path_count",
        "struct_env_var_count",
        "struct_file_size_log",
    ])

    # File type one-hot encoding
    for ft in KNOWN_FILE_TYPES:
        names.append(f"ftype_{ft}")

    # Byte histogram anomaly scores
    for i in range(256):
        names.append(f"byte_{i:02x}")

    # Byte anomaly aggregate stats
    names.extend([
        "byte_anomaly_mean",
        "byte_anomaly_max",
        "byte_overrep_count",
        "byte_underrep_count",
    ])

    return names


def collect_samples(directory: Path, extensions: Optional[set[str]] = None) -> list[Path]:
    """Recursively collect all files from a directory, optionally filtered by extension."""
    if not directory.exists():
        return []

    samples = []
    for path in directory.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            if extensions is None or path.suffix.lower() in extensions:
                samples.append(path)
    return samples


def load_jsonl_dir(directory: Path) -> list[tuple[dict, Path]]:
    """Load all JSONL files from a training directory.

    Returns list of (report, source_path) tuples.
    Each JSONL file may contain multiple lines (e.g., for archives with multiple members).
    """
    results = []
    if not directory.exists():
        print(f"  Warning: directory {directory} does not exist")
        return results

    jsonl_files = list(directory.glob("*.jsonl"))
    print(f"  Found {len(jsonl_files)} JSONL files in {directory}")

    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path) as f:
                reports = []
                for line in f:
                    line = line.strip()
                    if line:
                        report = json.loads(line)
                        reports.append(report)

                # If multiple reports, merge findings (archive case)
                if len(reports) > 1:
                    merged = reports[0].copy()
                    merged["findings"] = []
                    for r in reports:
                        merged["findings"].extend(r.get("findings", []))
                    results.append((merged, jsonl_path))
                elif len(reports) == 1:
                    results.append((reports[0], jsonl_path))
        except Exception as e:
            print(f"  Warning: failed to load {jsonl_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract features from samples")
    parser.add_argument(
        "--malicious",
        type=Path,
        default=Path("/Users/t/dev/hex/samples"),
        help="Directory containing malicious samples",
    )
    parser.add_argument(
        "--benign",
        type=Path,
        default=Path("./benign_samples"),
        help="Directory containing benign samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features.npz"),
        help="Output file for features",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per class (for testing)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated file extensions to include (e.g., .py,.js,.sh,.exe,.dll)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Directory to cache cleave reports (speeds up re-runs)",
    )
    parser.add_argument(
        "--cleave-dir",
        type=Path,
        default=None,
        help="cleave source directory (contains traits/). If not set, uses CLEAVE_DIR env var.",
    )
    # New: support for pre-computed cleave output from training directories
    parser.add_argument(
        "--training-good",
        type=Path,
        default=None,
        help="Directory with pre-computed cleave JSONL for good/benign samples (from trait-basher --training-data)",
    )
    parser.add_argument(
        "--training-bad",
        type=Path,
        default=None,
        help="Directory with pre-computed cleave JSONL for bad/malicious samples (from trait-basher --training-data)",
    )
    args = parser.parse_args()

    global KNOWN_CAPABILITIES, KNOWN_IMPORTS, FEATURE_NAMES, CACHE_DIR, BYTE_BASELINES, CLEAVE_DIR

    # Check if using pre-computed training data mode
    use_training_dirs = args.training_good is not None or args.training_bad is not None
    if use_training_dirs:
        if args.training_good is None:
            args.training_good = Path.home() / "data" / "training" / "good"
        if args.training_bad is None:
            args.training_bad = Path.home() / "data" / "training" / "bad"

        print("Using pre-computed cleave output from training directories:")
        print(f"  Good: {args.training_good}")
        print(f"  Bad:  {args.training_bad}")

        # Load pre-computed reports
        print("\nLoading pre-computed cleave reports...")
        good_reports = load_jsonl_dir(args.training_good)
        bad_reports = load_jsonl_dir(args.training_bad)

        print(f"\nLoaded {len(good_reports)} good, {len(bad_reports)} bad reports")

        if not good_reports:
            print("Error: No good/benign reports found!", file=sys.stderr)
            sys.exit(1)
        if not bad_reports:
            print("Error: No bad/malicious reports found!", file=sys.stderr)
            sys.exit(1)

        # Collect vocabulary from pre-computed reports
        print("\nCollecting vocabulary from pre-computed reports...")

        def collect_vocab_from_reports(reports: list, label: str):
            """Collect vocabulary from pre-computed reports."""
            capabilities = set()
            imports = set()
            for report, _ in reports:
                for finding in report.get("findings", []):
                    cap_id = finding.get("id", "")
                    capabilities.update(expand_capability_hierarchy(cap_id))
                for imp in report.get("imports", []):
                    imports.add(imp.get("symbol", ""))
            print(f"  {label}: {len(capabilities)} capabilities, {len(imports)} imports")
            return capabilities, imports

        good_caps, good_imports = collect_vocab_from_reports(good_reports, "Good")
        bad_caps, bad_imports = collect_vocab_from_reports(bad_reports, "Bad")

        KNOWN_CAPABILITIES = sorted(good_caps | bad_caps)
        KNOWN_IMPORTS = sorted(good_imports | bad_imports)
        # Use uniform byte baselines for pre-computed mode (no raw file access)
        BYTE_BASELINES = {ft: np.ones(256, dtype=np.float64) / 256 for ft in KNOWN_FILE_TYPES}

        print(f"\nVocabulary: {len(KNOWN_CAPABILITIES)} capabilities, {len(KNOWN_IMPORTS)} imports")

        FEATURE_NAMES = build_feature_names()
        print(f"Total features: {len(FEATURE_NAMES)}")

        # Extract features from pre-computed reports
        features_list = []
        labels_list = []
        paths_list = []

        print("\nExtracting features from good samples...")
        for report, source_path in tqdm(good_reports, desc="Good"):
            # Use a dummy path for byte features (no raw file access)
            features = extract_features(report, source_path)
            features_list.append(features)
            labels_list.append(0)  # 0 = benign
            paths_list.append(report.get("path", str(source_path)))

        print("\nExtracting features from bad samples...")
        for report, source_path in tqdm(bad_reports, desc="Bad"):
            features = extract_features(report, source_path)
            features_list.append(features)
            labels_list.append(1)  # 1 = malicious
            paths_list.append(report.get("path", str(source_path)))

        # Convert to numpy arrays and save (skip the rest of main())
        if not features_list:
            print("Error: No features extracted!", file=sys.stderr)
            sys.exit(1)

        X = np.stack(features_list)
        y = np.array(labels_list, dtype=np.int32)

        print(f"\nDataset shape: {X.shape}")
        print(f"Labels: {sum(y == 0)} benign, {sum(y == 1)} hostile")

        baseline_types = list(BYTE_BASELINES.keys())
        baseline_arrays = np.stack([BYTE_BASELINES[ft] for ft in baseline_types])

        np.savez_compressed(
            args.output,
            X=X,
            y=y,
            feature_names=FEATURE_NAMES,
            paths=paths_list,
            capabilities=KNOWN_CAPABILITIES,
            imports=KNOWN_IMPORTS,
            byte_baseline_types=baseline_types,
            byte_baseline_arrays=baseline_arrays,
        )
        print(f"Saved features to {args.output}")

        feature_names_path = args.output.with_suffix(".json")
        with open(feature_names_path, "w") as f:
            json.dump(FEATURE_NAMES, f)
        print(f"Saved feature names to {feature_names_path}")
        return

    # Original mode: run cleave on sample files
    # Set up cleave directory
    import os
    if args.cleave_dir:
        CLEAVE_DIR = args.cleave_dir
    elif os.environ.get("CLEAVE_DIR"):
        CLEAVE_DIR = Path(os.environ["CLEAVE_DIR"])
    else:
        CLEAVE_DIR = None
    if CLEAVE_DIR:
        print(f"Using cleave from: {CLEAVE_DIR}")

    # Set up cache directory
    if args.cache:
        CACHE_DIR = args.cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {CACHE_DIR}")

    # Parse extensions filter
    ext_filter = None
    if args.extensions:
        ext_filter = set(e.strip() if e.startswith('.') else f'.{e.strip()}'
                        for e in args.extensions.split(','))
        print(f"Filtering by extensions: {ext_filter}")

    # Collect samples
    malicious_samples = collect_samples(args.malicious, ext_filter)
    benign_samples = collect_samples(args.benign, ext_filter)

    # Shuffle for random sampling
    import random
    random.seed(42)  # Reproducible randomness
    random.shuffle(malicious_samples)
    random.shuffle(benign_samples)

    if args.max_samples:
        malicious_samples = malicious_samples[: args.max_samples]
        benign_samples = benign_samples[: args.max_samples]

    print(f"Found {len(malicious_samples)} malicious samples")
    print(f"Found {len(benign_samples)} benign samples")

    if not malicious_samples:
        print("Error: No malicious samples found!", file=sys.stderr)
        sys.exit(1)

    # First pass: collect vocabulary and byte baselines
    mal_caps, mal_imports, mal_stats, mal_baselines = collect_vocabulary(malicious_samples, "malicious")
    ben_caps, ben_imports, ben_stats, ben_baselines = collect_vocabulary(benign_samples, "benign")

    # Print stats summary
    print(f"\nMalicious: {mal_stats['ok']} ok, {mal_stats['fail']} failed")
    if mal_stats['errors']:
        for err, count in sorted(mal_stats['errors'].items(), key=lambda x: -x[1])[:5]:
            print(f"  - {err}: {count}")
    print(f"Benign: {ben_stats['ok']} ok, {ben_stats['fail']} failed")
    if ben_stats['errors']:
        for err, count in sorted(ben_stats['errors'].items(), key=lambda x: -x[1])[:5]:
            print(f"  - {err}: {count}")

    # Combine and sort for consistent ordering
    KNOWN_CAPABILITIES = sorted(mal_caps | ben_caps)
    KNOWN_IMPORTS = sorted(mal_imports | ben_imports)

    # Merge byte baselines (prefer benign baselines as they represent "normal" behavior)
    BYTE_BASELINES = {}
    for ft in KNOWN_FILE_TYPES:
        if ft in ben_baselines:
            BYTE_BASELINES[ft] = ben_baselines[ft]
        elif ft in mal_baselines:
            BYTE_BASELINES[ft] = mal_baselines[ft]
        else:
            BYTE_BASELINES[ft] = np.ones(256, dtype=np.float64) / 256

    print(f"\nVocabulary: {len(KNOWN_CAPABILITIES)} capabilities, {len(KNOWN_IMPORTS)} imports")

    # Build feature names
    FEATURE_NAMES = build_feature_names()
    print(f"Total features: {len(FEATURE_NAMES)}")

    # Second pass: extract features
    features_list = []
    labels_list = []
    paths_list = []

    print("\nExtracting features from malicious samples...")
    for file_path in tqdm(malicious_samples, desc="Malicious"):
        report, _, _ = run_cleave(file_path)
        if report is None:
            continue
        features = extract_features(report, file_path)
        features_list.append(features)
        labels_list.append(1)  # 1 = hostile
        paths_list.append(str(file_path))

    print("\nExtracting features from benign samples...")
    for file_path in tqdm(benign_samples, desc="Benign"):
        report, _, _ = run_cleave(file_path)
        if report is None:
            continue
        features = extract_features(report, file_path)
        features_list.append(features)
        labels_list.append(0)  # 0 = benign
        paths_list.append(str(file_path))

    if not features_list:
        print("Error: No features extracted!", file=sys.stderr)
        sys.exit(1)

    # Convert to numpy arrays
    X = np.stack(features_list)
    y = np.array(labels_list, dtype=np.int32)

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: {sum(y == 0)} benign, {sum(y == 1)} hostile")

    # Save to compressed file
    # Convert byte baselines dict to arrays for storage
    baseline_types = list(BYTE_BASELINES.keys())
    baseline_arrays = np.stack([BYTE_BASELINES[ft] for ft in baseline_types])

    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        feature_names=FEATURE_NAMES,
        paths=paths_list,
        capabilities=KNOWN_CAPABILITIES,
        imports=KNOWN_IMPORTS,
        byte_baseline_types=baseline_types,
        byte_baseline_arrays=baseline_arrays,
    )
    print(f"Saved features to {args.output}")

    # Also save feature names as JSON for Rust
    feature_names_path = args.output.with_suffix(".json")
    with open(feature_names_path, "w") as f:
        json.dump(FEATURE_NAMES, f)
    print(f"Saved feature names to {feature_names_path}")


if __name__ == "__main__":
    main()
