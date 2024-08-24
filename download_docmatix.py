import os
import glob
import json
import jsonlines
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random
import multiprocessing as mp
import jsonlines
from multiprocessing import Manager


threshold = 150000

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

def is_image_not_plain_white(image, pixel_threshold=250):
    img_array = np.array(image)
    return np.any(img_array < pixel_threshold)

def process_example(args):
    try:
        i, dataset, image_folder, counter = args
        example = dataset['train'][i]
        image_paths = []
        image_dimensions = []
        ignore = False
        num_images = len(example['images'])
        
        if counter[num_images] > threshold:
            return None
        
        """for img in example['images']:
            if not is_image_not_plain_white(img):
                ignore = True
                break
        
        if ignore:
            return None"""
        

        for cnt , img in enumerate(example['images']):
            image_path = f'{i}_{cnt}.jpg'
            save_image(img, os.path.join(image_folder, image_path))
            image_paths.append(image_path)
            image_dimensions.append((img.width, img.height))
           
        
        conversations = []
        for text in example['texts']:
            conversations.extend(create_conversation(text['user'], text['assistant'], len(image_paths)))
        
        json_data = {
            'id': i,
            'image': image_paths,
            'width_list': [dim[0] for dim in image_dimensions],
            'height_list': [dim[1] for dim in image_dimensions],
            'conversations': conversations
        }
        
        return json_data, num_images
    except Exception as e:
        print(f"Error processing example {i}: {str(e)}")
        return None

def process_dataset(dataset, image_folder, jsonl_file):

    os.makedirs(image_folder, exist_ok=True)
    clear_directory(image_folder)
    if os.path.exists(jsonl_file):
        os.remove(jsonl_file)
    
    total = len(dataset['train'])
    manager = Manager()
    counter = manager.dict()
    for i in range(1, 5):
        counter[i] = 0


    random_index = list(range(0, total))
    random.shuffle(random_index)
    
    pool = mp.Pool(processes=mp.cpu_count())
    
    args = [(i, dataset, image_folder, counter) for i in random_index]
    with jsonlines.open(jsonl_file, mode='w') as writer:
        for result in tqdm(pool.imap_unordered(process_example, args), 
                           total=total):
            
            if result is not None:
                json_data, num_images = result
                counter[num_images] += 1
                writer.write(json_data)
            
            if all([counter[x] > threshold for x in range(1, 5)]):
                print(counter)
                break


    return 

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
    jsonl_file = os.path.join(base_dir, 'opensource/docmatix.jsonl')
    train_file = os.path.join(base_dir, 'train.json')
    
    process_dataset(dataset, image_folder, jsonl_file)
    
    create_train_json(image_folder, jsonl_file, threshold*4, train_file)
    
    print(f"Processing complete. Images saved in {image_folder}")
    print(f"JSONL file created: {jsonl_file}")
    print(f"Train JSON file created: {train_file}")

if __name__ == "__main__":
    main()