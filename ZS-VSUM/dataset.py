from utils import *
import h5py

class VideoSumDataset(object):
    def __init__(self, keys, dataset):
        """
        Dataset class for loading video summarization dataset.

        Args:
            keys (list): List of video keys
            dataset (str): Dataset name 
        """
        self.keys = keys
        self.dataset = dataset
        self.data_root = f'../datasets/{dataset}'
        self.video_dict = h5py.File(f'{self.data_root}/{dataset.lower()}.h5', 'r')
        self.comparison_path = f"{self.data_root}/comparison_{dataset}.json"
            
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        """
        Returns the information of each video.
        """
        key = self.keys[index]
        video_name = key.split('/')[-1]
        
        with open(self.comparison_path, 'r') as f:
            comparsion_video_dict = json.load(f)
        num2name = {v: k for k, v in comparsion_video_dict.items()}
        video_original_name = num2name[video_name]
        
        video_file = self.video_dict[video_name]
        change_points = video_file['change_points'][...].astype(np.int32) # [S, 2], S: number of segments, each row stores indices of a segment
        n_frames = video_file['n_frames'][...].astype(np.int32) # [N], N: number of frames, N = T * 15
        n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32) # [S], indicates number of frames in each segment
        picks = video_file['picks'][...].astype(np.int32) # [T], posotions of subsampled frames in original video
        user_summary = video_file['user_summary'][...].astype(np.float32)
        user_score = video_file['user_scores'][...].astype(np.float32)  
        original_user_score = user_score * (5 - 1) + 1
        
        caption_path = os.path.join(self.data_root, "videos", video_original_name, "captions.txt") 
        caption_lines = read_caption(caption_path)

        return video_original_name, video_file, change_points, n_frames, n_frame_per_seg, picks, user_summary, caption_path, user_score, original_user_score, caption_lines
