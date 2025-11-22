#!/usr/bin/env python3
"""
Script to extract lines from a file and remove specific characters.
"""
import argparse
from tqdm import tqdm
from normalize import normalize

# Default characters to remove
CHARS_TO_REMOVE = '|\u05bd'


def prepare_data(input_file, output_file, num_lines, chars_to_remove):
    """
    Extract specified number of lines from input file and remove specified characters.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        num_lines: Number of lines to extract
        chars_to_remove: String of characters to remove
    """
    # Count total lines in file
    print(f"Counting lines in {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines in file: {total_lines:,}")
    lines_to_process = min(num_lines, total_lines)
    print(f"Processing {lines_to_process:,} lines...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(tqdm(infile, total=lines_to_process, desc="Processing")):
            if i >= num_lines:
                break
            
            # Remove unwanted characters
            cleaned_line = line
            for char in chars_to_remove:
                cleaned_line = cleaned_line.replace(char, '')
            
            # Normalize the text (sort diacritics, clean dagesh, deduplicate)
            cleaned_line = normalize(cleaned_line)
            
            outfile.write(cleaned_line)
    
    print(f"\n✓ Processed {lines_to_process:,} lines from {input_file} to {output_file}")
    print(f"✓ Removed characters: {repr(chars_to_remove)}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract lines from a file and remove specific characters'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='knesset_nikud_v6.txt',
        help='Input file path (default: knesset_nikud_v6.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output.txt',
        help='Output file path (default: output.txt)'
    )
    
    parser.add_argument(
        '--lines',
        type=int,
        default=1_000_000,
        help='Number of lines to extract (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--remove-chars',
        type=str,
        default=CHARS_TO_REMOVE,
        help=f'Characters to remove (default: "{CHARS_TO_REMOVE}")'
    )
    
    args = parser.parse_args()
    
    prepare_data(
        input_file=args.input,
        output_file=args.output,
        num_lines=args.lines,
        chars_to_remove=args.remove_chars
    )


if __name__ == '__main__':
    main()

