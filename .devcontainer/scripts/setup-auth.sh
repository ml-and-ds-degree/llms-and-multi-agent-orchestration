#!/bin/bash
set -e

AUTH_FILE="$HOME/.local/share/opencode/auth.json"
MOUNT_DIR="/tmp/opencode-auth-mount"
MOUNT_FILE="$MOUNT_DIR/auth.json"

# 1. Create directory with strict permissions (700 = only owner can read/write/execute)
# This prevents other users on the host from accessing the mount point
mkdir -p "$MOUNT_DIR"
chmod 700 "$MOUNT_DIR"

if [ -f "$AUTH_FILE" ]; then
    echo "Found auth.json at $AUTH_FILE. Creating secure symlink..."
    # Create a symlink pointing to the real file
    ln -sf "$AUTH_FILE" "$MOUNT_FILE"
else
    echo "Warning: auth.json not found at $AUTH_FILE. Creating empty placeholder."
    # Create an empty file if the secret doesn't exist to prevent Docker mount errors
    rm -f "$MOUNT_FILE"
    touch "$MOUNT_FILE"
fi
