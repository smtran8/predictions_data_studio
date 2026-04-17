import pandas as pd
import os
import sys
import argparse
import glob

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing

# ── Keyword list ──────────────────────────────────────────────────────────────
PREDICTION_KEYWORDS = [
    # Core prediction verbs
    'predict', 'forecast', 'anticipate', 'expect',
    'estimate', 'foresee', 'extrapolate',
    # Belief / opinion
    'assume', 
    # Future-oriented
    'will', 'shall', 'may', 'might', 'could',
    'plan', 'intend', 'aim', 'hope',
    # Warning / likelihood
    'warn', 'caution', 'alert', 'signal',
    'likely', 'unlikely', 'probable', 'possible', 'inevitable',
    'risk', 'chance', 'odds',
    # Projection nouns
    'prediction', 'forecast', 'outlook', 'prognosis',
    'expectation', 'estimate', 'scenario', 
]

def contains_prediction_keyword(sentence: str) -> int:
    """Return 1 if sentence contains any prediction keyword, else 0."""
    
    for keyword in PREDICTION_KEYWORDS:
        # Whole-word match to avoid 'will' matching 'William'
        import re
        if re.search(rf'\b{keyword}\b', sentence):
            return 1
    return 0

def auto_label_file(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if 'Sentence' not in df.columns:
        raise ValueError(f"'Sentence' column not found in {filepath}")

    df['Label'] = df['Sentence'].apply(contains_prediction_keyword)
    return df

def main(input_dir, output_dir, num_files = None):
    # Find all labeled CSVs
    csv_files = glob.glob(os.path.join(input_dir, '*_labeled*.csv'))
    total = len(csv_files)
    csv_files = sorted(csv_files)
    if num_files:
        csv_files = csv_files[:num_files]

    if total == 0:
        print(f"No *_labeled.csv files found in {input_dir}")
        return

    print(f"Found {total} labeled CSV file(s). Auto-labeling...\n")

    for idx, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        base_name = filename.replace('_labeled.csv', '')

        df = auto_label_file(filepath)

        # Stats
        n_prediction = df['Label'].sum()
        n_total = len(df)
        print(f"  [{idx}/{total}] {filename}")
        print(f"           → {n_prediction}/{n_total} sentences flagged as prediction ({n_prediction/n_total*100:.1f}%)")

        # Save as ABC19980108.1830.0711_auto_labeled.csv
        save_name = f"{base_name}_auto_labeled"
        DataProcessing.save_to_file(df, output_dir, save_name, 'csv')

    print(f"\nDone. Auto-labeled CSVs saved to: {output_dir}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("AUTO LABELER — PREDICTION KEYWORDS")
    print("="*50 + "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    parser = argparse.ArgumentParser(description='Auto-label sentences using prediction keywords')
    parser.add_argument('--input_dir', type=str,
                        default='timebank_1_2/data/labeled_sentences',
                        help='Folder containing *_labeled.csv files')
    parser.add_argument('--num_files', type=int, default=None,
                    help='Number of files to process (default: all). Use to skip already labeled files.')
    parser.add_argument('--output_dir', type=str,
                        default='timebank_1_2/data/auto_labeled_sentences',
                        help='Folder to save auto-labeled CSVs')
    args = parser.parse_args()

    input_dir = os.path.join(base_data_path, 'timebank_1_2/data/labeled_sentences')
    output_dir = os.path.join(base_data_path, 'timebank_1_2/data/auto_labeled_sentences')


    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}\n")

    main(input_dir, output_dir, args.num_files)