import csv
import os
import requests
from urllib.parse import urlparse
import ollama
from pathlib import Path

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def describe_image(path_to_image):
    try:
        res = ollama.chat(
            model="llava:13b",
            messages=[
                {
                    'role': 'system',
                    'content': """
                    You are an AI model specialized in image recognition and creating concise yet detailed textual descriptions. Your primary goal is to analyze the content of an image and produce a short, accurate summary suitable for LoRA (Low-Rank Adaptation) training.
                    """
                },
                {
                    'role': 'user',
                    'content': """Provide a concise description of this image for LoRA training. Focus on main objects, style, colors, and any notable visual features.""",
                    'images': [path_to_image]
                }
            ]
        )
        return res['message']['content']
    except Exception as e:
        print(f"Failed to describe image {path_to_image}: {e}")
        return "Description unavailable"

def process_memes(csv_path, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        
        for row in reader:
            image_url = row['Archived URL']
            base_meme = row['Base Meme Name']
            alt_text = row['Alternate Text']
            
            # Extract filename from URL
            parsed_url = urlparse(image_url)
            image_filename = os.path.basename(parsed_url.path)
            image_path = os.path.join(output_dir, image_filename)
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            
            # Skip if txt file already exists
            if os.path.exists(txt_path):
                print(f"Skipping {image_filename} - caption already exists")
                continue
            
            # Download image
            if not download_image(image_url, image_path):
                continue
                
            # Get description from LLaVA
            description = describe_image(image_path)
            
            # Write caption file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Base meme: {base_meme}\n")
                f.write(f"Text on image: {alt_text}\n")
                f.write(f"Description: {description}\n")
            
            print(f"Processed {image_filename}")

if __name__ == "__main__":
    # Path to your CSV file
    csv_file = 'memes.csv'
    
    # Directory to save images and captions
    output_directory = 'lora_captions'
    
    process_memes(csv_file, output_directory)
