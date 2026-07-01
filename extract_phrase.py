import json
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
import copy
import os
import sys

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_subject_object(sentence):
    """
    Extract subject and object phrases from a sentence

    Parameters:
    sentence (str): input sentence

    Returns:
    list: list of extracted subject and object phrases
    """
    if not sentence or not isinstance(sentence, str):
        print(f"Warning: invalid sentence input: {sentence}")
        return []

    try:
        # Tokenize and POS-tag
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)

        # Define simplified grammar rules - only recognize noun phrases
        grammar = """
NP: {<DT>?<JJ.*>*<NN.*>+}
    {<PRP>}
"""

        # Create parser
        parser = RegexpParser(grammar)

        # Parse sentence
        parsed = parser.parse(tagged)

        # Find subjects and objects (noun phrases)
        noun_phrases = []

        # Traverse parse tree, extract noun phrases
        for subtree in parsed.subtrees():
            if subtree.label() == 'NP':  # noun phrase (may be subject or object)
                phrase = ' '.join(word for word, tag in subtree.leaves())
                noun_phrases.append(phrase)

        return noun_phrases
    except Exception as e:
        print(f"Error extracting phrases: {e}")
        return []

def process_annotation_json(input_json_path, output_json_path):
    """
    Process input annotation_sentence.json and save as annotation_word.json
    """
    try:
        # Read input JSON file
        print(f"Reading {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create output JSON structure (copy original structure)
        output_data = copy.deepcopy(data)

        total_items = sum(len(split_data) for split_name, split_data in data.items())
        processed_items = 0

        # Iterate over all data entries
        for split_name, split_data in data.items():
            print(f"Processing {len(split_data)} entries in dataset '{split_name}'...")

            for i, item in enumerate(split_data):
                # Show progress
                processed_items += 1
                if processed_items % 100 == 0 or processed_items == total_items:
                    print(f"Progress: {processed_items}/{total_items} ({processed_items/total_items*100:.1f}%)")

                # Create a list to store extracted phrases for each sentence
                word_phrases = []

                # Ensure 'report_sentences' field exists
                if 'report_sentences' not in item:
                    print(f"Warning: entry {item.get('id', i)} has no 'report_sentences' field, skipping")
                    continue

                # Process each sentence
                for sentence in item['report_sentences']:
                    # Extract subjects and objects
                    phrases = extract_subject_object(sentence)
                    # Add extracted phrases to the list
                    word_phrases.extend(phrases)

                # Replace original 'report_sentences' with extracted phrases
                output_data[split_name][i]['report_words'] = word_phrases
                # Delete original 'report_sentences' field
                if 'report_sentences' in output_data[split_name][i]:
                    del output_data[split_name][i]['report_sentences']

        # Save output JSON file
        print(f"Saving results to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        return output_data
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description='Extract phrases from annotation_sentence.json and generate annotation_phrase.json')
    parser.add_argument('--input', dest='input_json_path',
                        default='data/iu_xray/annotation_sentence.json',
                        help='path to input annotation_sentence.json')
    parser.add_argument('--output', dest='output_json_path',
                        default='data/iu_xray/annotation_phrase.json',
                        help='path to output annotation_phrase.json')
    args = parser.parse_args()
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path

    # Check if input file exists
    if not os.path.exists(input_json_path):
        print(f"Error: input file '{input_json_path}' not found")
        print("Please verify the file path is correct or update the path in the code.")
        return

    # Process JSON file
    output_data = process_annotation_json(input_json_path, output_json_path)

    if output_data:
        print(f"Processing complete. Results saved to {output_json_path}")

# Test cases
def test_extract():
    """
    Test phrase extraction functionality
    """
    test_sentences = [
        "The heart size and pulmonary vascularity appear within normal limits.",
        "A large hiatal hernia is noted.",
        "The lungs are free of focal airspace disease.",
        "No pneumothorax or pleural effusion is seen.",
        "Degenerative changes are present in the spine."
    ]

    print("Testing subject and object extraction:")
    for sentence in test_sentences:
        phrases = extract_subject_object(sentence)
        print(f"\nOriginal sentence: {sentence}")
        print(f"Extracted subjects and objects: {phrases}")

if __name__ == "__main__":
    # Uncomment the following line to test phrase extraction
    #test_extract()

    # Run main program
    main()