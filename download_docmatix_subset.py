import os
import glob
import json
import jsonlines
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import multiprocessing
from multiprocessing import Manager, Value
from functools import partial
import random

def clear_directory(directory):
    for file in glob.glob(f'{directory}/*'):
        os.remove(file)

def save_image(image, path):
    image.save(path)

def create_conversation(query, response, num_images):
    return [
        {"from": "human", "value": "<image>\n" * num_images + query},
        {"from": "gpt", "value": response}
    ]

def process_partition(args):

    partition_indices, jsonl_file, dataset, image_folder, counter, lock = args
    max_images_per_count = 150000
  
    
    with jsonlines.open(jsonl_file, mode='w') as writer:
        for partition_index in tqdm(partition_indices):
            try:
                example = dataset['train'][partition_index]
                image_paths = []
                image_dimensions = []
                
                num_images = len(example['images'])
                

                with lock:
                    # if counter for all num_images is greater than threshold break
                    if all([counter[i] >= max_images_per_count for i in range(1, 5)]):
                        break

                    if counter[num_images] >= max_images_per_count:
                        continue
                    

                for cnt , img in enumerate(example['images']):
                    image_path = f'{partition_index}_{cnt}.jpg'
                    save_image(img, os.path.join(image_folder, image_path))
                    image_paths.append(image_path)
                    image_dimensions.append((img.width, img.height))
                
                conversations = []
                for text in example['texts']:
                    conversations.extend(create_conversation(text['user'], text['assistant'], len(image_paths)))

                json_data = {
                    'id': partition_index,
                    'image': image_paths,
                    'width_list': [dim[0] for dim in image_dimensions],
                    'height_list': [dim[1] for dim in image_dimensions],
                    'conversations': conversations
                }
                writer.write(json_data)

                with lock:
                    counter[num_images] += 1

            except Exception as e:
                print(f"Error processing example {partition_index}: {e}")
                

def merge_jsonl_files(output_file, jsonl_files):
    with jsonlines.open(output_file, mode='w') as writer:
        for jsonl_file in jsonl_files:
            with jsonlines.open(jsonl_file) as reader:
                for obj in reader:
                    writer.write(obj)

def create_train_json(image_folder, jsonl_file, dataset_length, output_file):
    data = {
        'docmatix': {
            'root': image_folder,
            'annotation': jsonl_file,
            'data_augment': False,
            'repeat_time': 1,
            'length': dataset_length
        }
    }
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile)

def main():
    dataset = load_dataset("HuggingFaceM4/Docmatix", "images")
    
    base_dir = '/home/ubuntu/InternVL/playground/'
    image_folder = os.path.join(base_dir, 'data/docmatix')
    jsonl_folder = os.path.join(base_dir, 'jsonl_parts')
    final_jsonl_file = os.path.join(base_dir, 'opensource/docmatix.jsonl')
    train_file = os.path.join(base_dir, 'train.json')
    
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(jsonl_folder, exist_ok=True)
    clear_directory(image_folder)
    clear_directory(jsonl_folder)
    
    total = len(dataset['train'])
    num_workers = multiprocessing.cpu_count()-5
    
    indices = list(range(total))
    random.shuffle(indices)
    partition_size = total // num_workers
    partitions = [indices[i:i + partition_size] for i in range(0, total, partition_size)]
    
    manager = Manager()
    counter = manager.dict({i: 0 for i in range(1, 5)})
    lock = manager.Lock()
    
    pool = multiprocessing.Pool(processes=num_workers)
    
    jsonl_files = [os.path.join(jsonl_folder, f'part_{i}.jsonl') for i in range(num_workers)]
    
    # Preparing arguments to pass as a tuple
    args = [(partitions[i], jsonl_files[i], dataset, image_folder, counter, lock) for i in range(num_workers)]
    
    pool.map(process_partition, args)
    
    pool.close()
    pool.join()

    merge_jsonl_files(final_jsonl_file, jsonl_files)
    create_train_json(image_folder, final_jsonl_file, 800000, train_file)

    print(f"Processing complete. Images saved in {image_folder}")
    print(f"Final JSONL file created: {final_jsonl_file}")
    print(f"Train JSON file created: {train_file}")

if __name__ == "__main__":
    main()
