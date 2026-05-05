#!/bin/bash
# SmartBin Auto-Start Setup for Raspberry Pi 5 (Hotspot mode)
# Run once: bash setup_autostart.sh

PROJECT_DIR="/home/pi/FULL_PI"
PYTHON_BIN="$(which python3)"
RUN_USER="$(whoami)"
SERVICE_NAME="smartbin"

echo "=================================================="
echo " SmartBin Auto-Start Setup (Hotspot mode)"
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
echo "[1/5] Adding $RUN_USER to gpio group..."
sudo usermod -aG gpio "$RUN_USER"
echo "[OK] GPIO group"

# Step 2: Disable network-online (causes slow boot)
echo "[2/5] Disabling network-online.target (fixes slow boot)..."
sudo systemctl disable systemd-networkd-wait-online.service 2>/dev/null
echo "[OK] network-online disabled"

# Step 3: Enable hostapd
echo "[3/5] Enabling hostapd..."
sudo systemctl enable hostapd 2>/dev/null
if [ $? -eq 0 ]; then
  echo "[OK] hostapd enabled"
else
  echo "[WARN] hostapd not found - smartbin will use network.target instead"
fi

# Step 4: Create service file
echo "[4/5] Creating /etc/systemd/system/${SERVICE_NAME}.service..."

# Check if hostapd exists
if systemctl list-unit-files | grep -q "hostapd.service"; then
  AFTER_TARGET="hostapd.service"
  WANTS_TARGET="hostapd.service"
else
  AFTER_TARGET="network.target"
  WANTS_TARGET="network.target"
fi

echo "      Using After=$AFTER_TARGET"

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=SmartBin V2 (server + engine)
After=$AFTER_TARGET
Wants=$WANTS_TARGET

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

# Step 5: Enable and Start
echo "[5/5] Enabling and starting service..."
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