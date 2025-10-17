
ARCH=$(uname -m)
MODEL_PATH="D:\models\openPangu-Embedded-7B-V1.1"
cd "$MODEL_PATH" || exit 1
if [ "$ARCH" = "arm64" ]; then
    sha256sum checklist.chk
else
    sha256sum -c checklist.chk
fi