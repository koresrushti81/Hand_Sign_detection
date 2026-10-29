#!/bin/bash

# Hand Sign Recognition System Launcher
# Linux/Mac Shell Script

clear
echo "============================================"
echo "  Hand Sign Recognition System"
echo "============================================"
echo ""
echo "Choose an option:"
echo ""
echo "1. Run GUI Application (Recommended)"
echo "2. Run Recognition Test"
echo "3. Run Data Collection Tool"
echo "4. Install Dependencies"
echo "5. Exit"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        echo "Starting GUI Application..."
        python app.py
        ;;
    2)
        echo "Starting Recognition Test..."
        python test_improved.py
        ;;
    3)
        echo "Starting Data Collection..."
        python datacollection_improved.py
        ;;
    4)
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo "Done!"
        ;;
    5)
        exit
        ;;
    *)
        echo "Invalid option"
        ;;
esac
