#!/usr/bin/env python3
"""
Sommelier Setup Script

Checks prerequisites, installs dependencies, configures API keys,
and optionally sets up Claude Desktop integration.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Verify Python 3.12+."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 12):
        print(f"  Python 3.12+ required, found {major}.{minor}")
        print(f"  Download from https://www.python.org/downloads/")
        return False
    print(f"  Python {major}.{minor} OK")
    return True


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("  requirements.txt not found -- skipping.")
        return False
    print("  Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  pip install failed:\n{result.stderr[:500]}")
        return False
    print("  Dependencies installed.")
    return True


def configure_api_keys():
    """Prompt for API keys and write .env file."""
    env_file = Path(__file__).parent / ".env"
    existing = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()

    providers = [
        ("GEMINI_API_KEY", "Gemini (recommended -- cheapest, native video/PDF)"),
        ("OPENAI_API_KEY", "OpenAI (GPT-4o / GPT-4.1)"),
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
    ]

    print("\n  At least one AI provider API key is required.")
    if existing:
        configured = [name for name, _ in providers if existing.get(name)]
        if configured:
            print(f"  Already configured: {', '.join(configured)}")

    keys = dict(existing)
    for env_var, label in providers:
        current = existing.get(env_var, "")
        if current:
            change = input(f"  {label}: [keep current] ").strip()
            if change:
                keys[env_var] = change
        else:
            value = input(f"  {label}: [skip] ").strip()
            if value:
                keys[env_var] = value

    has_any = any(keys.get(name) for name, _ in providers)
    if not has_any:
        print("\n  No API keys configured. You'll need at least one to use Sommelier.")
        return False

    lines = []
    for env_var, label in providers:
        if keys.get(env_var):
            lines.append(f"{env_var}={keys[env_var]}")
    # Preserve any other keys from existing .env
    for key, value in keys.items():
        if key not in {name for name, _ in providers}:
            lines.append(f"{key}={value}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved to {env_file}")
    return True


def find_claude_desktop_config() -> Path | None:
    """Find Claude Desktop config file location."""
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(xdg) / "Claude" / "claude_desktop_config.json"
    return None


def configure_claude_desktop():
    """Add Sommelier to Claude Desktop MCP config."""
    config_path = find_claude_desktop_config()
    if config_path is None:
        print("  Could not detect Claude Desktop config location.")
        return False

    project_dir = str(Path(__file__).parent.resolve())
    mcp_entry = {
        "command": "python",
        "args": ["mcp_server.py"],
        "env": {
            "PYTHONPATH": project_dir,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    }

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
    else:
        config = {}

    servers = config.setdefault("mcpServers", {})
    if "sommelier" in servers:
        print(f"  Sommelier already in {config_path}")
        update = input("  Overwrite? [y/N] ").strip().lower()
        if update != "y":
            return True

    servers["sommelier"] = mcp_entry
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"  Added Sommelier to {config_path}")
    return True


def main():
    print("\nSommelier Setup")
    print("=" * 40)

    # Step 1: Python version
    print("\n[1/4] Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Dependencies
    print("\n[2/4] Installing dependencies...")
    install_dependencies()

    # Step 3: API keys
    print("\n[3/4] Configuring API keys...")
    configure_api_keys()

    # Step 4: Claude Desktop
    print("\n[4/4] Claude Desktop integration...")
    config_path = find_claude_desktop_config()
    if config_path:
        print(f"  Config location: {config_path}")
        setup_claude = input("  Add Sommelier to Claude Desktop? [Y/n] ").strip().lower()
        if setup_claude != "n":
            configure_claude_desktop()
        else:
            print("  Skipped.")
    else:
        print("  Claude Desktop config not found -- skipping.")

    # Done
    print("\n" + "=" * 40)
    print("Setup complete!")
    print()
    print('Open Claude Desktop and say: "Check my Sommelier status"')
    print()


if __name__ == "__main__":
    main()
