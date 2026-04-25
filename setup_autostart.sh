#!/bin/bash
# SmartBin Auto-Start Setup for Raspberry Pi 5
# Run once: bash setup_autostart.sh

PROJECT_DIR="/home/pi/FULL_PI"
PYTHON_BIN="$(which python3)"
RUN_USER="$(whoami)"
SERVICE_NAME="smartbin"

echo "=================================================="
echo " SmartBin Auto-Start Setup"
echo "=================================================="
echo " Project : $PROJECT_DIR"
echo " Python  : $PYTHON_BIN"
echo " User    : $RUN_USER"
echo " Service : $SERVICE_NAME"
echo "=================================================="

# Check project files exist
if [ ! -f "$PROJECT_DIR/server.py" ]; then
  echo "[ERROR] $PROJECT_DIR/server.py not found"
  echo "        Edit PROJECT_DIR in this script and re-run"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/main.py" ]; then
  echo "[ERROR] $PROJECT_DIR/main.py not found"
  exit 1
fi

echo "[OK] Found server.py and main.py"

# Step 1: GPIO permission
echo ""
echo "[1/4] Adding $RUN_USER to gpio group..."
sudo usermod -aG gpio "$RUN_USER"
echo "[OK] GPIO group"

# Step 2: Network wait
echo "[2/4] Enabling network-online.target..."
sudo systemctl enable systemd-networkd-wait-online.service 2>/dev/null
echo "[OK] Network wait"

# Step 3: Create service file
echo "[3/4] Creating /etc/systemd/system/${SERVICE_NAME}.service..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=SmartBin V2 (server + engine)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN $PROJECT_DIR/server.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
KillSignal=SIGTERM
StandardOutput=journal
StandardError=journal
SyslogIdentifier=smartbin

[Install]
WantedBy=multi-user.target
EOF

echo "[OK] Service file created"

# Step 4: Enable and Start
echo "[4/4] Enabling and starting service..."
sudo systemctl daemon-reload

sudo systemctl enable ${SERVICE_NAME}.service
if [ $? -eq 0 ]; then
  echo "[OK] Service enabled"
else
  echo "[WARN] Enable failed"
fi

sudo systemctl start ${SERVICE_NAME}.service
if [ $? -eq 0 ]; then
  echo "[OK] Service started"
else
  echo "[WARN] Service failed to start - check logs below"
fi

# Wait and show status
echo ""
echo "Waiting 4 seconds for service to initialize..."
sleep 4

echo ""
echo "=================================================="
echo " Service Status"
echo "=================================================="
sudo systemctl status ${SERVICE_NAME}.service --no-pager -l

echo ""
echo "=================================================="
echo " Setup Complete"
echo "=================================================="
echo ""
echo " Useful commands:"
echo "   View live log : journalctl -u ${SERVICE_NAME} -f"
echo "   Status        : sudo systemctl status ${SERVICE_NAME}"
echo "   Stop          : sudo systemctl stop ${SERVICE_NAME}"
echo "   Restart       : sudo systemctl restart ${SERVICE_NAME}"
echo "   Disable       : sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo " Test auto-start:"
echo "   sudo reboot"
echo ""
echo " After reboot, check:"
echo "   journalctl -u ${SERVICE_NAME} -b --no-pager | head -60"
echo "=================================================="