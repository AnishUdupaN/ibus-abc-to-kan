#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HOME}/.local/share/indicxlit-kn"
MODEL_DIR="${TARGET_DIR}/models"
CONTAINER_NAME="indicxlit-kn-builder"

for cmd in podman uv python3 ibus; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd"; exit 1; }
done

cleanup() {
    echo "[host] Killing and removing container '${CONTAINER_NAME}'..."
    podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[host] 1/5 Starting temporary fedora:latest container..."
podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
podman run -d --name "${CONTAINER_NAME}" registry.fedoraproject.org/fedora:latest sleep infinity >/dev/null

echo "[host] 2/5 Copying export scripts into container and running build..."
podman exec "${CONTAINER_NAME}" mkdir -p /work /out
podman cp "${SCRIPT_DIR}/export_kn_onnx.py" "${CONTAINER_NAME}:/work/export_kn_onnx.py"
podman cp "${SCRIPT_DIR}/container_export.sh" "${CONTAINER_NAME}:/work/container_export.sh"
podman exec "${CONTAINER_NAME}" bash /work/container_export.sh

echo "[host] 3/5 Copying exported models back to ${MODEL_DIR}..."
rm -rf "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"
podman cp "${CONTAINER_NAME}:/out/." "${MODEL_DIR}/"

# Remove container immediately before setting up host runtime
cleanup
trap - EXIT

echo "[host] 4/5 Creating isolated host runtime venv (zero system Python packages)..."
rm -rf "${TARGET_DIR}/venv"
uv venv "${TARGET_DIR}/venv" --python /usr/bin/python3
uv pip install --python "${TARGET_DIR}/venv/bin/python" onnxruntime numpy

cp "${SCRIPT_DIR}/kn_onnx_runtime.py" "${TARGET_DIR}/kn_onnx_runtime.py"
cp "${SCRIPT_DIR}/ibus_indicxlit.py" "${TARGET_DIR}/ibus_indicxlit.py"
chmod +x "${TARGET_DIR}/ibus_indicxlit.py"

echo "[host] 5/5 Registering IBus component and restarting IBus..."
sudo tee /usr/share/ibus/component/indicxlit-kn.xml > /dev/null << EOF
<?xml version="1.0" encoding="utf-8"?>
<component>
    <name>org.freedesktop.IBus.IndicXlitKn</name>
    <description>IndicXlit Kannada Engine</description>
    <exec>/usr/bin/env MALLOC_ARENA_MAX=2 ${TARGET_DIR}/ibus_indicxlit.py --ibus</exec>
    <version>1.0</version>
    <author>Anish</author>
    <license>MIT</license>
    <homepage>https://github.com/AI4Bharat/IndicXlit</homepage>
    <textdomain>indicxlit-kn</textdomain>
    <engines>
        <engine>
            <name>indicxlit-kn</name>
            <language>kn</language>
            <license>MIT</license>
            <author>Anish</author>
            <layout>us</layout>
            <longname>Kannada (IndicXlit)</longname>
            <description>Neural Phonetic Kannada Typing</description>
            <rank>99</rank>
        </engine>
    </engines>
</component>
EOF

ibus write-cache

# Non-destructively append indicxlit-kn to GNOME input sources if not already present
CURRENT_SOURCES="$(gsettings get org.gnome.desktop.input-sources sources)"
if [[ "${CURRENT_SOURCES}" != *"'indicxlit-kn'"* ]]; then
    if [[ "${CURRENT_SOURCES}" == "@a(ss) []" ]]; then
        gsettings set org.gnome.desktop.input-sources sources "[('xkb', 'us'), ('ibus', 'indicxlit-kn')]"
    else
        UPDATED_SOURCES="${CURRENT_SOURCES%]}, ('ibus', 'indicxlit-kn')]"
        gsettings set org.gnome.desktop.input-sources sources "${UPDATED_SOURCES}"
    fi
fi

ibus restart

echo ""
echo "=== Setup Complete ==="
du -sh "${TARGET_DIR}" "${TARGET_DIR}/"*
"${TARGET_DIR}/venv/bin/python" "${TARGET_DIR}/kn_onnx_runtime.py"
