from utils import *
from dataset import VideoSumDataset
import numpy as np

def find_ranges(user_summary):
    """
    Convert binary user summary into index ranges where the value is 1.

    Args:
        user_summary (np.ndarray): Binary array indicating important frames.

    Returns:
        np.ndarray: List of [start, end) ranges.
    """
    indices = np.where(user_summary == 1)[0]
    ranges = []
    start = indices[0]
    prev = indices[0]
    
    for i in indices[1:]:
        if i != prev + 1:
            ranges.append([start, prev + 1])
            start = i
        prev = i
        
    ranges.append([start, prev + 1])
    return np.array(ranges)

def score_gt_label(file_path, original_anno_score):
    """
    Create a dictionary of captions with corresponding frame scores.

    Args:
        file_path (str): Path to the caption file.
        original_anno_score (list): Score for each caption.

    Returns:
        str: JSON string of caption: score mapping.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    result_dict = {lines[i].strip(): int(original_anno_score[i]) for i in range(min(len(lines), len(original_anno_score)))}
    return json.dumps(result_dict, indent=4)

def create_training_data(args, annoindex):
    """
    Generate exampled prompt.

    Args:
        args: Command-line or script arguments containing dataset info.
        annoindex (int): Index of annotator to use.

    Returns:
        str: Constructed few-shot training prompt.
    """
    yaml_path = f"../datasets/{args.dataset}/FS_videos.yml"
    exampled_prompt = open(f"./prompt_{args.dataset}.txt","r").read()
    
    keys_info = load_yaml(yaml_path)
    train_keys = keys_info[0]['train_keys']  
    train_video_dataset =  VideoSumDataset(train_keys, args.dataset)

    for i in range(len(train_keys)):
        video_name, video_file, change_points, n_frames, n_frame_per_seg, picks, user_summary, caption_path, user_score, original_user_score, caption_lines  = train_video_dataset[i]
        annotator_gt = [[int(value) for value in sublist] for sublist in find_ranges(user_summary[annoindex])/15]
        
        if args.dataset=="TVSum":
            title, genre, query = get_metadata(args, video_name)
            original_anno_score = original_user_score[annoindex][::15]
            gt_label = score_gt_label(caption_path, original_anno_score)
            exampled_prompt = exampled_prompt.replace(f'[VIDEO {(i+1)}]', gt_label).replace(f'[TITLE {(i+1)}]',title)  
        
    return exampled_prompt

def reduce_captions(captions, ratio): 
    """
    Downsample a list of captions by a specified ratio.

    Args:
        captions (list): List of caption strings.
        ratio (int): Step size for reduction.

    Returns:
        list: Reduced list of captions.
    """
    return captions[::ratio]

def convert2dict(caption_path, ratio):
    """
    Convert captions into a dictionary with binary labels.

    Args:
        caption_path (str): Path to caption file.
        ratio (int): Reduction step.

    Returns:
        str: JSON string of caption: 1 mapping.
    """
    with open(caption_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    reduced_lines = reduce_captions(lines,ratio) 
    lines = reduced_lines 
    result_dict = {line.strip(): 1 for line in lines}
    return json.dumps(result_dict, indent=4)

def pseudo_label(caption_path, ratio):
    """
    Assign pseudo labels (1 to 5) to captions in a repeating pattern.

    Args:
        caption_path (str): Path to caption file.
        ratio (int): Reduction step.

    Returns:
        str: JSON string of caption: pseudo score mapping.
    """
    with open(caption_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    result_dict = {}
    lines = reduce_captions(lines,ratio) 
    for index, line in enumerate(lines):
        value = (index % 5) + 1
        result_dict[line.strip()] = value
    return json.dumps(result_dict, indent=4)

def process_captions_scores(caption_path, ratio):
    """
    Process captions and assign pseudo labels.

    Args:
        caption_path (str): Path to caption file.
        ratio (int): Reduction step.

    Returns:
        tuple: (original_captions, reduced_captions, reduced_captions_str, pseudo_labels_json)
    """
    original_captions = read_caption(caption_path)
    reduced_captions = reduce_captions(original_captions,ratio)
    psuedo_labels = pseudo_label(caption_path, ratio)
    reduced_captions_str = '\n'.join(reduced_captions)
    return original_captions, reduced_captions, reduced_captions_str, psuedo_labels
    