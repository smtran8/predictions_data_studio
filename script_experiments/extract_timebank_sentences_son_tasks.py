import pandas as pd
import re
import os
import sys
import argparse
import spacy 
# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing

def clean_header_noise(text):
    """Remove document header noise like 'ABC19980108.1830.0711 NEWS STORY'"""
    text = re.sub(r'^[\w\d.\-]+\s+NEWS STORY\s*', '', text)
    return text.strip()

def extract_sentences(content, nlp):
    # Remove XML declaration and TimeML tags (reused from original script)
    content = re.sub(r'<\?xml.*?\?>', '', content)
    content = re.sub(r'<TimeML[^>]*>', '', content)
    content = re.sub(r'</TimeML>', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\s+', ' ', content).strip()

    # Clean header noise
    content = clean_header_noise(content)

    # spaCy sentence splitting
    doc = nlp(content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def process_single_file(filepath, filename, nlp, output_dir):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = extract_sentences(content, nlp)

    df = pd.DataFrame({
        'File Name': filename,
        'Sentence': sentences,
        'Label': ''  # to be filled manually
    })

    # Save as ABC19980108.1830.0711_labeled.csv
    base_name = os.path.splitext(filename)[0]
    save_name = f"{base_name}_labeled"
    DataProcessing.save_to_file(df, output_dir, save_name, 'csv')

    return df

def main(input_folder, output_dir, single_file=None, num_files = 1):
    nlp = spacy.load("en_core_web_sm")

    if single_file:
        all_files = [single_file]
    else:
        all_files = [f for f in os.listdir(input_folder) if f.endswith('.tml')]
        all_files = all_files[num_files:]

    total_files = len(all_files)
    print(f"Processing {total_files} file(s)...")

    for idx, filename in enumerate(all_files, 1):
        filepath = os.path.join(input_folder, filename)
        df = process_single_file(filepath, filename, nlp, output_dir)
        print(f"  [{idx}/{total_files}] {filename} => {len(df)} sentences")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TIMEBANK SENTENCE LABELING")
    print("="*50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    parser = argparse.ArgumentParser(description='Split TimeBank .tml files into labeled sentences')
    parser.add_argument('--single_file', type=str, default='ABC19980108.1830.0711.tml',
                        help='Process a single .tml file by name for testing')
    parser.add_argument('--num_files', type=int, default=1,
                    help='Number of files to process (default: 1)')
    parser.add_argument('--all', action='store_true',
                        help='Process all .tml files instead of a single file')
    args = parser.parse_args()

    input_folder = os.path.join(base_data_path, 'timebank_1_2/data/timeml')
    output_dir = os.path.join(base_data_path, 'timebank_1_2/data/labeled_sentences')

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_dir}\n")

    single_file = None if (args.all or args.num_files > 1) else args.single_file
    main(input_folder, output_dir, single_file, args.num_files)