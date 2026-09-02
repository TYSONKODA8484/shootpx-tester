#!/bin/bash
# ShootPX Tester Launcher (macOS)
# Double-click this file in Finder to run it.
# (First time only: if macOS blocks it, right-click the file -> Open,
#  or run once in Terminal:  chmod +x "Start ShootPX Tester.command")

cd "$(dirname "$0")" || exit 1

# Prefer python3's streamlit if the bare "streamlit" command isn't on PATH
if command -v streamlit >/dev/null 2>&1; then
    STREAMLIT="streamlit"
elif python3 -m streamlit --version >/dev/null 2>&1; then
    STREAMLIT="python3 -m streamlit"
else
    echo ""
    echo "  Streamlit isn't installed for this Python yet."
    echo "  Open a terminal in this folder and run:  pip3 install -r requirements.txt"
    echo ""
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

while true; do
    clear
    echo ""
    echo "  ShootPX Tester"
    echo "  ================"
    echo ""
    echo "  1. On-Model Shots"
    echo "  2. Catalog Photoshoot"
    echo "  3. Creative Photoshoot"
    echo "  4. Recolor"
    echo "  5. Exit"
    echo ""
    read -r -p "  Choose a number: " choice

    case "$choice" in
        1) app="streamlit_app.py" ;;
        2) app="catalog_streamlit_app.py" ;;
        3) app="creative_streamlit_app.py" ;;
        4) app="recolor_streamlit_app.py" ;;
        5) exit 0 ;;
        *)
            echo ""
            echo "  Not a valid choice - try again."
            read -n 1 -s -r
            continue
            ;;
    esac

    clear
    echo ""
    echo "  Starting $app ..."
    echo "  Your browser will open in a few seconds."
    echo "  To stop this tool, close this window or press Ctrl+C."
    echo ""
    $STREAMLIT run "$app"
    echo ""
    echo "  $app has stopped."
    echo ""
    read -n 1 -s -r -p "Press any key to continue..."
done
