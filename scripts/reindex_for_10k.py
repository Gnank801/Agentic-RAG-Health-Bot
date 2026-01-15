import sys
import os

# Add parent dir to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingestion import ingest_from_csv, reset_index

def main():
    print("WARNING: This will DELETE all existing data in the vector database.")
    print("Ensure your 10k Synthea CSV files are in 'data/' directory.")
    confirm = input("Type 'yes' to proceed: ")
    
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    # 1. Clear Index
    reset_index()

    # 2. Ingest 10k Records
    print("\nStarting ingestion for 10,000 records...")
    ingest_from_csv(data_dir='data', max_patients=10000)
    print("\nDone! 10k ingestion complete.")

if __name__ == "__main__":
    main()
