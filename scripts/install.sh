#!/usr/bin/env bash

set -euo pipefail

# If not running as root, re-execute with sudo to gain privileges for system directories
if [ "$(id -u)" -ne 0 ]; then
  echo "Re-running script with sudo..."
  exec sudo "$0" "$@"
fi

# Default values
INSTALL_DIR="/opt/gustav"
SERVICE_USER=""
PORT="80"
CLIENT_ONLY=false
SERVICE_ONLY=false
SOURCE=""
REF="main"
INTERACTIVE=false

default_limit="0"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --client                Install only the CLI client.
  --service               Install and configure the Gustav server as a systemd service.
  --dir <path>            Installation prefix (default: $INSTALL_DIR).
  --ref <git-ref>         Git reference/branch to checkout (default: $REF).
  --source <path>         Use a local source directory instead of cloning.
  --port <port>           Port for the web server (default: $PORT).
  --no-prompt             Run without interactive prompts, using defaults.
  --one-liner             Print a one‑liner curl command for quick install.
  --help                  Show this help message.
EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --client) CLIENT_ONLY=true; shift ;;
    --service) SERVICE_ONLY=true; shift ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --interactive) INTERACTIVE=true; shift ;;
    --one-liner) echo "curl -fsSL https://raw.githubusercontent.com/gut-engel/gustav/main/scripts/install.sh | sudo bash -s -- ${*:2}"; exit 0 ;;
    --help) usage ;;
    --*) echo "Unknown option: $1"; usage ;;
    *) echo "Unexpected argument: $1"; usage ;;
  esac
done

# Determine if we need both client and service
if [ "$CLIENT_ONLY" = false ] && [ "$SERVICE_ONLY" = false ]; then
  CLIENT_ONLY=true
  SERVICE_ONLY=true
fi

# Defaults when not interactive
if [ "${INTERACTIVE}" = true ]; then
  # Prompt for service user if installing service
  if [ "$SERVICE_ONLY" = true ]; then
    echo "Select the user that will run the Gustav service:"
    echo "1) gustav (dedicated service user)"
    echo "2) $(whoami) (the user that invoked the installer)"
    read -p "Enter choice [1-2]: " choice
    case "$choice" in
      1) SERVICE_USER="gustav" ;;
      2) SERVICE_USER="$(whoami)" ;;
      *) echo "Invalid choice, defaulting to $(whoami)"; SERVICE_USER="$(whoami)" ;;
    esac
  fi

  # Prompt for Web UI setup
  read -p "Would you like to enable the Web UI? (y/N): " enable_web
  if [[ "$enable_web" =~ ^[Yy]$ ]]; then
    ENABLE_WEB=true
  else
    ENABLE_WEB=false
  fi

  # Prompt for first admin user
  read -p "Create the first admin account now? (y/N): " create_admin
  if [[ "$create_admin" =~ ^[Yy]$ ]]; then
    read -p "Admin username: " admin_name
    read -s -p "Password: " admin_pass
    echo
    read -s -p "Confirm password: " admin_pass2
    echo
    if [ "$admin_pass" != "$admin_pass2" ]; then
      echo "Passwords do not match – aborting admin creation."
      exit 1
    fi
    CREATE_ADMIN=true
  else
    CREATE_ADMIN=false
  fi

  # Prompt for usage limits (requests per minute)
  read -p "Default usage limit for new accounts (requests/min, 0 for unlimited): " default_limit
  if ! [[ "$default_limit" =~ ^[0-9]+$ ]]; then
    echo "Invalid number – defaulting to 0 (unlimited)."
    default_limit=0
  fi
else
  # Non‑interactive defaults
  SERVICE_USER="$(whoami)"
  ENABLE_WEB=false
  CREATE_ADMIN=false
  default_limit="0"
fi

# Ensure install directory exists (needs root)
sudo mkdir -p "$INSTALL_DIR"

# Install CLI client
if [ "$CLIENT_ONLY" = true ]; then
  echo "Installing CLI client..."
  if [ -n "$SOURCE" ]; then
    sudo cp -r "$SOURCE" "$INSTALL_DIR"
  else
    sudo git clone --depth 1 --branch "$REF" https://github.com/gut-engel/gustav.git "$INSTALL_DIR"
  fi
  # Create a virtual environment and install the package in editable mode
  PYEXE=$(command -v python3 || command -v python || echo "python")
  sudo "$PYEXE" -m venv "$INSTALL_DIR/.venv"
  sudo "$INSTALL_DIR/.venv/bin/pip" install --upgrade pip
  sudo "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR"
  echo "CLI installed. Add \"$INSTALL_DIR/.local/bin\" to your PATH if needed."
fi

# Install systemd service
if [ "$SERVICE_ONLY" = true ]; then
  echo "Setting up systemd service..."
  SERVICE_DIR="/etc/systemd/system"
  SERVICE_FILE="gustav.service"
  cat > "/tmp/$SERVICE_FILE" <<EOT
[Unit]
Description=Gustav AI Server
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/gustav-server --port $PORT
Restart=on-failure
Environment=GUSTAV_JWT_SECRET=$(openssl rand -hex 32)
Environment=GUSTAV_WEB_ENABLED=$( [ "$ENABLE_WEB" = true ] && echo 1 || echo 0 )
Environment=GUSTAV_DEFAULT_USAGE_LIMIT=$default_limit

[Install]
WantedBy=multi-user.target
EOT
  sudo cp "/tmp/$SERVICE_FILE" "$SERVICE_DIR/"
  sudo systemctl daemon-reload
  sudo systemctl enable gustav.service
  sudo systemctl start gustav.service
  echo "Gustav service installed and started as user $SERVICE_USER on port $PORT."

  # Create first admin if requested
  if [ "$CREATE_ADMIN" = true ]; then
    echo "Creating admin account..."
    ai user add --name "$admin_name" --role owner --password "$admin_pass"
    echo "Admin account '$admin_name' created."
  fi
fi

echo "Installation complete."
