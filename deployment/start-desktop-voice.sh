#!/bin/bash
# Start after desktop login, using the user's audio/Bluetooth session.
set -e
assistant_project_dir="/home/aitcs/Documents/ait_project"
cd "$assistant_project_dir"
mkdir -p data
# Prevent duplicate launches from repeated autostart requests.
exec 9>data/desktop-voice.lock
if ! flock -n 9; then
    exit 0
fi
printf 'Starting the college voice assistant after desktop audio settles...\n'
sleep 8
if command -v wpctl >/dev/null 2>&1; then
    for attempt in 1 2 3 4 5; do
        if wpctl inspect @DEFAULT_AUDIO_SINK@ >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
fi
exec "$assistant_project_dir/.venv/bin/python" -u "$assistant_project_dir/assistant.py" --mode voice
