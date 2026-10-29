"""
Helper script to convert topics.json to kb_chunks.json format
This allows you to use your existing mental health topics data
"""

import json
import os

def convert_topics_to_chunks(topics_file="data/topics.json", output_file="kb_chunks.json"):
    """Convert topics.json to kb_chunks.json format"""
    
    print(f"Reading {topics_file}...")
    
    if not os.path.exists(topics_file):
        print(f"Error: {topics_file} not found!")
        return
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    topics = data.get('topics', [])
    chunks = []
    
    print(f"Converting {len(topics)} topics to chunks...")
    
    for topic_idx, topic in enumerate(topics):
        title = topic.get('title', f'Topic {topic_idx + 1}')
        content_list = topic.get('content', [])
        
        # Create a chunk for each content item
        for content_idx, content in enumerate(content_list):
            chunk = {
                "text": content,
                "source": title,
                "metadata": {
                    "topic_index": topic_idx,
                    "content_index": content_idx,
                    "topic_title": title,
                    "chunk_id": len(chunks)
                }
            }
            chunks.append(chunk)
    
    # Save to kb_chunks.json
    print(f"Writing {len(chunks)} chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_file} with {len(chunks)} chunks!")
    print(f"\nYou can now run: python ingest.py")

if __name__ == "__main__":
    convert_topics_to_chunks()

