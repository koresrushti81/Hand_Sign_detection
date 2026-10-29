import sys
import os

# Add src to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.modern_app import main

if __name__ == "__main__":
    main()
