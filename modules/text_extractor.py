import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class TextExtractor:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize the text extractor with Bio-ClinicalBERT model
        
        Args:
            model_name (str): Name or path of the pre-trained model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
    def extract_cls_embeddings(self, texts):
        """
        Extract [CLS] token embeddings for a list of texts
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Array of [CLS] embeddings
        """
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Print progress for long operations
            print(f"Processing {len(texts)} texts through BERT...")
            outputs = self.model(**encoded_input)
            
        # Get the [CLS] embeddings (first token of each sequence)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_embeddings

    def process_report_annotations(self, report_path):
        """
        Process report-level annotations and extract paragraph features
        
        Args:
            report_path (str): Path to the report annotations JSON file
            
        Returns:
            dict: Dictionary with report IDs as keys and paragraph embeddings as values
        """
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        report_features = {}
        report_texts = []
        report_ids = []
        
        # The JSON contains train/test/val splits
        for split in ['train', 'test', 'val']:
            if split in data:
                print(f"Processing {len(data[split])} reports from {split} split")
                for report_item in data[split]:
                    report_id = report_item.get('id')
                    # Check different possible keys for the report text
                    text = None
                    for key in ['report', 'text', 'impression', 'findings']:
                        if key in report_item:
                            text = report_item[key]
                            break
                    
                    if text and report_id:
                        report_texts.append(text)
                        report_ids.append(report_id)
        
        print(f"Total extracted reports: {len(report_texts)}")
        
        # Get embeddings for all paragraphs
        if report_texts:
            # Process in batches to avoid memory issues with large datasets
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(report_texts), batch_size):
                batch_texts = report_texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(report_texts)-1)//batch_size + 1} with {len(batch_texts)} reports")
                batch_embeddings = self.extract_cls_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch embeddings
            if len(all_embeddings) > 1:
                paragraph_embeddings = np.vstack(all_embeddings)
            else:
                paragraph_embeddings = all_embeddings[0]
            
            # Map embeddings back to report IDs
            for i, report_id in enumerate(report_ids):
                report_features[report_id] = paragraph_embeddings[i]
        
        print(f"Generated features for {len(report_features)} reports")
        return report_features
    
    def process_sentence_annotations(self, sentence_path):
        """
        Process sentence-level annotations and extract sentence features
        
        Args:
            sentence_path (str): Path to the sentence annotations JSON file
            
        Returns:
            dict: Dictionary with sentence IDs as keys and sentence embeddings as values
        """
        with open(sentence_path, 'r') as f:
            data = json.load(f)
        
        sentence_features = {}
        all_sentences = []  # List of (report_id, sentence_idx, sentence_text) tuples
        
        # Process data organized by train/test/val splits
        for split in ['train', 'test', 'val']:
            if split in data:
                print(f"Processing {len(data[split])} reports with sentences from {split} split")
                for report_item in data[split]:
                    report_id = report_item.get('id')
                    
                    # Check for report_sentences key which contains the list of sentences
                    if 'report_sentences' in report_item and report_id:
                        sentences = report_item['report_sentences']
                        for i, sentence in enumerate(sentences):
                            if sentence:  # Skip empty sentences
                                # Create a unique ID for each sentence
                                sentence_id = f"{report_id}_s{i}"
                                all_sentences.append((report_id, i, sentence, sentence_id))
        
        print(f"Total extracted sentences: {len(all_sentences)}")
        
        # Extract features for sentences
        if all_sentences:
            # Process in batches to avoid memory issues with large datasets
            batch_size = 64
            sentence_texts = [s[2] for s in all_sentences]
            sentence_ids = [s[3] for s in all_sentences]
            
            all_embeddings = []
            for i in range(0, len(sentence_texts), batch_size):
                batch_texts = sentence_texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(sentence_texts)-1)//batch_size + 1} with {len(batch_texts)} sentences")
                batch_embeddings = self.extract_cls_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch embeddings
            if len(all_embeddings) > 1:
                sentence_embeddings = np.vstack(all_embeddings)
            else:
                sentence_embeddings = all_embeddings[0]
            
            # Map embeddings back to sentence IDs
            for i, sentence_id in enumerate(sentence_ids):
                sentence_features[sentence_id] = sentence_embeddings[i]
        
        print(f"Generated features for {len(sentence_features)} sentences")
        return sentence_features
    
    def process_word_annotations(self, word_path):
        """
        Process word-level annotations and extract word features.
        Extracts words directly from the report_words list in the annotation_word JSON.
        
        Args:
            word_path (str): Path to the word annotations JSON file
            
        Returns:
            dict: Dictionary with word IDs as keys and word embeddings as values
        """
        with open(word_path, 'r') as f:
            data = json.load(f)
        
        word_features = {}
        all_words = []  # List of (report_id, word_idx, word_text, word_id) tuples
        
        # Extract words directly from report_words in the data
        for split in ['train', 'test', 'val']:
            if split in data:
                print(f"Processing words from {len(data[split])} reports in {split} split")
                for report_item in data[split]:
                    report_id = report_item.get('id')
                    
                    # Extract words from report_words if available
                    if 'report_words' in report_item and report_id:
                        words = report_item['report_words']
                        for word_idx, word in enumerate(words):
                            if word:  # Skip empty words
                                # Create a unique ID for each word
                                word_id = f"{report_id}_w{word_idx}"
                                all_words.append((report_id, word_idx, word, word_id))
        
        print(f"Total extracted words: {len(all_words)}")
        
        # Process word features
        if all_words:
            # Create contextualized words for better embeddings
            word_texts = [f"The term is {w[2]}." for w in all_words]
            word_ids = [w[3] for w in all_words]
            
            # Process in batches
            batch_size = 128
            all_embeddings = []
            
            for i in range(0, len(word_texts), batch_size):
                batch_texts = word_texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(word_texts)-1)//batch_size + 1} with {len(batch_texts)} words")
                batch_embeddings = self.extract_cls_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch embeddings
            if len(all_embeddings) > 1:
                word_embeddings = np.vstack(all_embeddings)
            else:
                word_embeddings = all_embeddings[0]
            
            # Map embeddings back to word IDs
            for i, word_id in enumerate(word_ids):
                word_features[word_id] = word_embeddings[i]
        
        print(f"Generated features for {len(word_features)} words")
        return word_features
    
    def extract_all_features(self, report_path, sentence_path, word_path):
        """
        Extract features at all levels: paragraph, sentence, and word
        
        Args:
            report_path (str): Path to the report annotations JSON file
            sentence_path (str): Path to the sentence annotations JSON file
            word_path (str, optional): Path to the word annotations JSON file. 
                                     If None, will use sentence_path to extract words.
            
        Returns:
            tuple: (paragraph_features, sentence_features, word_features)
        """
        print("Processing report-level features...")
        paragraph_features = self.process_report_annotations(report_path)
        
        print("\nProcessing sentence-level features...")
        sentence_features = self.process_sentence_annotations(sentence_path)
        
        print("\nProcessing word-level features...")
        word_features = self.process_word_annotations(word_path)
        
        return paragraph_features, sentence_features, word_features
    
    def save_features(self, output_dir, paragraph_features, sentence_features, word_features):
        """
        Save extracted features to JSON files
        
        Args:
            output_dir (str): Directory to save output files
            paragraph_features (dict): Paragraph-level features
            sentence_features (dict): Sentence-level features
            word_features (dict): Word-level features
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_paragraph_features = {k: v.tolist() for k, v in paragraph_features.items()}
        serializable_sentence_features = {k: v.tolist() for k, v in sentence_features.items()}
        serializable_word_features = {k: v.tolist() for k, v in word_features.items()}
        
        # Save to files
        with open(f"{output_dir}/paragraph_features.json", 'w') as f:
            json.dump(serializable_paragraph_features, f)
            
        with open(f"{output_dir}/sentence_features.json", 'w') as f:
            json.dump(serializable_sentence_features, f)
            
        with open(f"{output_dir}/word_features.json", 'w') as f:
            json.dump(serializable_word_features, f)


def main():
    parser = argparse.ArgumentParser(description="Extract multi-level text features using Bio-ClinicalBERT")
    parser.add_argument("--report_path", default='data/iu_xray/annotation.json', help="Path to report annotations JSON file")
    parser.add_argument("--sentence_path", default='data/iu_xray/annotation_sentence.json', help="Path to sentence annotations JSON file")
    parser.add_argument("--word_path", default='data/iu_xray/annotation_phrase.json', help="Path to word annotations JSON file (optional, will use sentence_path if not provided)")
    parser.add_argument("--output_dir", default="data/features", help="Directory to save output features")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT", help="Pre-trained model name")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    print(f"Initializing TextExtractor with model: {args.model_name}")
    extractor = TextExtractor(model_name=args.model_name)
    
    # Extract features
    print("Starting feature extraction...")
    paragraph_features, sentence_features, word_features = extractor.extract_all_features(
        args.report_path, args.sentence_path, args.word_path
    )
    
    # Save features
    print(f"Saving features to {args.output_dir}...")
    extractor.save_features(args.output_dir, paragraph_features, sentence_features, word_features)
    
    print(f"Features successfully extracted and saved to {args.output_dir}")
    print(f"Paragraph features: {len(paragraph_features)} paragraphs")
    print(f"Sentence features: {len(sentence_features)} sentences")
    print(f"Word features: {len(word_features)} words")


if __name__ == "__main__":
    main()