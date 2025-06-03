# scripts/generate_metadata.py
import json
from src.sitemap_processor import generate_source_document

def main():
    sitemap_url = "https://maelove.com/sitemap.xml"  # Replace with your sitemap
    metadata = generate_source_document(sitemap_url, max_urls=10, delay=0.5)
    output_path = "results/source_document.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Generated source document with {len(metadata)} entries at {output_path}")

if __name__ == "__main__":
    main()