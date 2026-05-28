# %%
import pandas as pd 
import torch 
import os 
import clearml 
import pathlib
import configparser
import copy
from pathlib import Path
from typing import List, Any
from torch.utils.data import random_split

# %% [markdown]
# # Dataset 

# %%

class PitchSenseDataset(torch.utils.data.Dataset):
    def __init__(self,roots:List[Any]):
        self.roots=roots
        self.imgs=[]
        for root in self.roots:
            root = Path(root)
            if not root.exists():
                continue
            for match_id in sorted(os.listdir(root)):
                match_path = root / match_id
                if not match_path.is_dir():
                    continue
                # print(match_path)
                sample = self._build_match(match_path)
                
                if sample:
                    self.imgs.extend(self._match_to_frames(sample))
    
    def _match_to_frames(self,sample):
        df_gt,df_det,path=sample['gt'],sample['det'],sample['match_path']
        frames=df_gt['frame'].unique()

        return [self._filter_dfs(sample,df_gt,df_det,path,frame) for frame in frames]

    def _filter_dfs(self,sample,df_gt,df_det,path,frame):
        filtered_gt=df_gt[df_gt['frame']==frame]
        filtered_det=df_det[df_det['frame']==frame]
        img_path=path / "img1" / f"{frame:06d}.jpg"
        
        return {
            "img_path": img_path,
            "gt": filtered_gt,
            "det": filtered_det,
            "config": sample['config']
        }

    def _build_match(self,match_path):
        tree=[f for f in match_path.iterdir()]
        game_config,seq_config = self.get_inis(tree)
        tree=[f for f in match_path.iterdir() if f.is_dir()]
        match_config=game_config | seq_config
        gt,det,tree=self.boiler_plate(tree,match_config)

        return {
            "match_path": match_path,
            "gt": gt,
            "det": det,
            "config": match_config
        }
    
    def _read_ini_to_dict(self,path: Path) -> dict:
        """Read an .ini file into a nested dictionary {section: {key: value}}"""
        parser = configparser.ConfigParser()
        parser.read(path)
        return {section: dict(parser[section]) for section in parser.sections()}

    def get_inis(self,tree):
        for sub_path in tree:
            if sub_path.is_file():
                if sub_path.stem == "gameinfo":
                    game_config = self._read_ini_to_dict(sub_path)['Sequence']
                elif sub_path.stem == "seqinfo":
                    seq_config = self._read_ini_to_dict(sub_path)['Sequence']
        return game_config,seq_config

    def get_mapper(self,match_config):
        tracklet_map = {}
        for key, value in match_config.items():  
            if key.startswith("trackletid_"):
                idx = int(key.split("_")[1])
                name = value.split(";")[0]  
                tracklet_map[idx] = name
        return tracklet_map

    def boiler_plate(self,tree,match_config):
        for item in copy.copy(tree):
            if item.stem=='gt':
                gt = pd.read_csv(item/"gt.txt", header=None)
                tree.remove(item)
            elif item.stem=='det':
                det = pd.read_csv(item/"det.txt", header=None)
                tree.remove(item)
        gt.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'class_id', 'f1','f2','f3']
        det.columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'class_id', 'f1','f2','f3']
        tracklet_map=self.get_mapper(match_config)
        
        gt['name'] = gt['track_id'].map(tracklet_map)
        det['name'] = gt['track_id'].map(tracklet_map)
        return gt,det,tree


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


