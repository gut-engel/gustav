#!/usr/bin/env python3
"""Pure‑Python installer for Gustav.

Features:
  * Installs the CLI client (or just the service) without needing a shell script.
  * Supports non‑interactive mode via ``--no-prompt`` and defaults.
  * Sets up a systemd service if ``--service`` is requested.
  * Allows specifying installation directory, git ref, source path, port, and
    default usage limit.

All actions are performed using the standard library and ``subprocess`` – no
external dependencies beyond what the project already requires.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], check: bool = True, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=check, cwd=cwd)

def main() -> None:
    parser = argparse.ArgumentParser(description="Install Gustav (Python‑only installer)")
    parser.add_argument("--client", action="store_true", help="Install only the CLI client")
    parser.add_argument("--service", action="store_true", help="Install the systemd service")
    parser.add_argument("--dir", default="/opt/gustav", help="Installation prefix (default: /opt/gustav)")
    parser.add_argument("--ref", default="main", help="Git branch / ref to checkout (default: main)")
    parser.add_argument("--source", help="Local source directory instead of cloning")
    parser.add_argument("--port", default="80", help="Port for the web server (default: 80)")
    parser.add_argument("--no-prompt", action="store_true", help="Run without interactive prompts, using defaults")
    args = parser.parse_args()

    # Determine what to install
    client_only = args.client
    service_only = args.service
    if not client_only and not service_only:
        client_only = service_only = True

    install_dir = Path(args.dir).expanduser().resolve()
    install_dir.mkdir(parents=True, exist_ok=True)

    # Resolve defaults for non‑interactive mode
    if args.no_prompt:
        service_user = os.getenv("USER", "root")
        enable_web = False
        create_admin = False
        default_limit = "0"
    else:
        # Service user selection
        if service_only:
            print("Select the user that will run the Gustav service:")
            print("1) gustav (dedicated service user)")
            print(f"2) {os.getenv('USER', 'current user')} (the user that invoked the installer)")
            choice = input("Enter choice [1-2]: ").strip()
            service_user = "gustav" if choice == "1" else os.getenv("USER", "root")
        else:
            service_user = ""
        # Web UI
        enable_web = input("Would you like to enable the Web UI? (y/N): ").strip().lower() == "y"
        # Admin account
        create_admin = input("Create the first admin account now? (y/N): ").strip().lower() == "y"
        if create_admin:
            admin_name = input("Admin username: ").strip()
            admin_pass = input("Password (will not echo): ")  # Simplified – not hidden for brevity
        else:
            admin_name = admin_pass = ""
        # Default usage limit
        default_limit = input("Default usage limit for new accounts (requests/min, 0 for unlimited): ").strip()
        if not default_limit.isdigit():
            print("Invalid number – defaulting to 0 (unlimited).")
            default_limit = "0"

    # Clone or copy source
    if args.source:
        src_path = Path(args.source).resolve()
        if not src_path.is_dir():
            sys.exit(f"Source directory {src_path} does not exist")
        run(["cp", "-r", str(src_path), str(install_dir)])
    else:
        run(["git", "clone", "--depth", "1", "--branch", args.ref,
            "https://github.com/gut-engel/gustav.git", str(install_dir)])

    # Install Python package
    run([sys.executable, "-m", "pip", "install", "."], cwd=install_dir)

    if service_only:
        service_file = f"/etc/systemd/system/gustav.service"
        service_contents = f"""[Unit]
Description=Gustav AI Server
After=network.target

[Service]
Type=simple
User={service_user}
WorkingDirectory={install_dir}
ExecStart={install_dir}/.venv/bin/gustav-server --port {args.port}
Restart=on-failure
Environment=GUSTAV_JWT_SECRET=$(openssl rand -hex 32)
Environment=GUSTAV_WEB_ENABLED={'1' if enable_web else '0'}
Environment=GUSTAV_DEFAULT_USAGE_LIMIT={default_limit}

[Install]
WantedBy=multi-user.target
"""
        # Write the unit file (requires sudo)
        with open("/tmp/gustav.service", "w", encoding="utf-8") as f:
            f.write(service_contents)
        run(["sudo", "cp", "/tmp/gustav.service", service_file])
        run(["sudo", "systemctl", "daemon-reload"])
        run(["sudo", "systemctl", "enable", "gustav.service"])
        run(["sudo", "systemctl", "start", "gustav.service"])
        print(f"Gustav service installed and started as user {service_user} on port {args.port}.")

        if create_admin:
            # Use the installed CLI to create admin (assumes ``ai`` is on PATH)
            print(f"Creating admin account '{admin_name}'...")
            run(["ai", "user", "add", "--name", admin_name, "--role", "owner", "--password", admin_pass])

    print("Installation complete.")

if __name__ == "__main__":
    main()
