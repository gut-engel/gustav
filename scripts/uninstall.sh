#!/usr/bin/env bash

set -euo pipefail

# Default values
INSTALL_DIR="/opt/gustav"
INTERACTIVE=true

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --dir <path>            Installation directory to remove (default: $INSTALL_DIR).
  --interactive           Run with prompts (default is non‑interactive).
  --help                  Show this help message.
EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --interactive) INTERACTIVE=true; shift ;;
    --help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [ "$INTERACTIVE" = true ]; then
  read -p "Are you sure you want to uninstall Gustav from $INSTALL_DIR? This will stop the service and delete files. (y/N): " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
  fi
fi

# Stop and disable systemd service if it exists
if systemctl list-unit-files | grep -q "gustav.service"; then
  echo "Stopping Gustav service..."
  sudo systemctl stop gustav.service || true
  echo "Disabling Gustav service..."
  sudo systemctl disable gustav.service || true
  sudo rm -f /etc/systemd/system/gustav.service
  sudo systemctl daemon-reload
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
  echo "Removing installation directory $INSTALL_DIR..."
  sudo rm -rf "$INSTALL_DIR"
fi

# Optionally remove the dedicated service user if it exists and has no remaining files
if id "gustav" &>/dev/null; then
  if [ "$INTERACTIVE" = true ]; then
    read -p "Remove the dedicated service user 'gustav'? (y/N): " remove_user
    if [[ "$remove_user" =~ ^[Yy]$ ]]; then
      sudo userdel -r gustav || true
    fi
  else
    # Non‑interactive: remove user without prompting if it exists
    sudo userdel -r gustav || true
  fi
fi

echo "Uninstall complete."
