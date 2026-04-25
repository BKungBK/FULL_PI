#!/bin/bash
# ============================================================
# SmartBin Auto-Start Setup (corrected) — รันครั้งเดียวบน Pi 5
# ============================================================
# สิ่งที่สคริปนี้ทำ:
#   1. เพิ่ม user เข้า group gpio
#   2. เปิด network-online.target
#   3. สร้าง service เดียว (server.py) — engine รันเป็น thread ข้างใน
#   4. Enable + Start service
#
# Usage:
#   bash setup_autostart.sh
# ============================================================

set -e

# ── ตั้งค่า — แก้ตรงนี้ถ้า path ไม่ตรง ──────────────────────
PROJECT_DIR="/home/pi/FULL_PI"
PYTHON_BIN="$(which python3)"
RUN_USER="$(whoami)"
SERVICE_NAME="smartbin"
# ─────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo "=================================================="
echo "  SmartBin Auto-Start Setup"
echo "=================================================="
echo "  Project : $PROJECT_DIR"
echo "  Python  : $PYTHON_BIN"
echo "  User    : $RUN_USER"
echo "  Service : $SERVICE_NAME"
echo "=================================================="
echo ""

# ── ตรวจ path ────────────────────────────────────────────────
[ -f "$PROJECT_DIR/server.py" ] || fail "ไม่พบ $PROJECT_DIR/server.py — แก้ PROJECT_DIR แล้วรันใหม่"
[ -f "$PROJECT_DIR/main.py"   ] || fail "ไม่พบ $PROJECT_DIR/main.py"
ok "พบไฟล์ server.py และ main.py"

# ── ตรวจ dependencies ────────────────────────────────────────
echo ""
echo "[CHECK] ตรวจ Python dependencies..."
MISSING=()
for pkg in fastapi uvicorn cv2 lgpio tflite_runtime ncnn; do
  if ! "$PYTHON_BIN" -c "import $pkg" 2>/dev/null; then
    MISSING+=("$pkg")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  warn "package ที่ยังไม่ได้ติดตั้ง: ${MISSING[*]}"
  warn "ติดตั้งก่อนด้วย: pip install ${MISSING[*]} --break-system-packages"
  echo ""
  read -p "ดำเนินการต่อโดยไม่ติดตั้งก่อน? (y/N): " CONT
  [[ "$CONT" =~ ^[Yy]$ ]] || exit 1
else
  ok "dependencies ครบ"
fi

# ── 1. GPIO permission ───────────────────────────────────────
echo ""
echo "[1/4] เพิ่ม $RUN_USER เข้า group gpio..."
sudo usermod -aG gpio "$RUN_USER"
ok "GPIO group"

# ── 2. Network wait ──────────────────────────────────────────
echo "[2/4] เปิด network-online.target..."
sudo systemctl enable systemd-networkd-wait-online.service 2>/dev/null || \
  warn "systemd-networkd-wait-online ไม่พบ — ข้ามขั้นตอนนี้"
ok "Network wait"

# ── 3. สร้าง service (เดียว) ─────────────────────────────────
echo "[3/4] สร้าง /etc/systemd/system/${SERVICE_NAME}.service..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=SmartBin V2 (server + engine)
Documentation=https://github.com/BKungBK/FULL_PI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR

# รัน server.py เดียว — engine เปิดเป็น thread อัตโนมัติผ่าน app_lifespan
ExecStart=$PYTHON_BIN $PROJECT_DIR/server.py

# Restart ถ้า crash โดยรอ 5 วินาที
Restart=on-failure
RestartSec=5

# ส่ง SIGTERM ก่อน แล้วรอ 15 วินาที ถ้ายังไม่ดับค่อย SIGKILL
TimeoutStopSec=15
KillSignal=SIGTERM

# Log ไปที่ journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=smartbin

[Install]
WantedBy=multi-user.target
EOF

ok "สร้าง service file แล้ว"

# ── 4. Enable + Start ─────────────────────────────────────────
echo "[4/4] Enable และ Start service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service
sudo systemctl start ${SERVICE_NAME}.service

# ── รอสักครู่แล้วตรวจสถานะ ──────────────────────────────────
sleep 3
echo ""
echo "=================================================="
echo "  สถานะ service"
echo "=================================================="
sudo systemctl status ${SERVICE_NAME}.service --no-pager -l || true

# ── สรุป ─────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup เสร็จแล้ว"
echo "=================================================="
echo ""
echo "คำสั่งที่ใช้บ่อย:"
echo ""
echo "  ดู log realtime  : journalctl -u ${SERVICE_NAME} -f"
echo "  ดู status        : sudo systemctl status ${SERVICE_NAME}"
echo "  หยุด             : sudo systemctl stop ${SERVICE_NAME}"
echo "  เริ่มใหม่        : sudo systemctl restart ${SERVICE_NAME}"
echo "  ถอน autostart    : sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "ทดสอบ boot:"
echo "  sudo reboot"
echo ""
echo "หลัง reboot เช็ค:"
echo "  journalctl -u ${SERVICE_NAME} -b --no-pager | head -50"
echo "=================================================="
