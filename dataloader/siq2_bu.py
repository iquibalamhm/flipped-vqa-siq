import torch
from .base_dataset import BaseDataset
import json
import copy
import pysrt
import pandas as pd
import re


class SIQ2(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        # USING JSON
        # json_path = f'./data/tvqa/tvqa_{split}.jsonl'
        # feature_path = f'./data/tvqa/clipvitl14.pth'
    
        # with open(json_path, "r") as f: 
            # data_list = list(f)
        # self.data = [json.loads(x) for x in data_list]
        # USING CSV
        csv_path = f'./data/{args.dataset}/{split}.csv'
        self.data = pd.read_csv(csv_path)
        self.features = torch.load(f'./data/{args.dataset}/clipvitl14.pth')
        self.subtitle_path = f'./data/{args.dataset}/transcript_trimmed/' # provided as castle_s01e01_seg02_clip_00.srt
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3 : '(D)'}
        self.num_options = 4
        # self.sub = args.sub
        self.sub = args.sub
        self.args = args
        print(f"Num {split} data: {len(self.data)} from {csv_path}") 

    def _get_text(self, idx, choices, vid, start, end):
        # question = self.data[idx]["q"].capitalize().strip()
        question = self.data["question"].values[idx].capitalize().strip()

        if question[-1] != "?":
            question = str(question) + "?"
        
        if self.sub:
            dialogue = ''
            with open(self.subtitle_path+f'{vid}'+'.vtt', "r") as file:
                content = file.read()
            lines = content.strip().split("\n")
            lines = lines[1:]
            i = 0
            prev_line = None
            while i < len(lines):
                    text_lines = []

                    # Check if the line matches a timestamp
                    match = re.match(r"(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)", lines[i])
                    if match:
                        start_time, end_time = match.groups()
                        st = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(start_time.split(":"))))
                        et = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(end_time.split(":"))))
                        i += 1

                        while i < len(lines) and not re.match(r"(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)", lines[i]) and lines[i] != "":
                            if lines[i] == ' ': # if the line is a space, skip it
                                # print("line empty",lines[i])
                                pass
                            else:                    
                                if lines[i] != prev_line:
                                    text_lines.append(lines[i])
                                prev_line = lines[i]
                                # print("line ",lines[i])
                            i += 1

                        current_text = " ".join(text_lines).strip()
                        if (st >= start and et <= end) or (st <= start and et  <= end and start <= et):
                            if current_text:
                                dialogue += ' ' + current_text
                    else:
                        i += 1
            if self.args.blank_subs or self.args.no_modality_inp:
                dialogue = "gibberish"
            if dialogue != '': d_text = f"Dialogue: {dialogue}\n"
            else: d_text =  ''
            # print(f'vid: {vid}, d_text {d_text}')
        else: 
            d_text = ""
        
        q_text = f"Question: {question}\n"
        o_text = f"Choices: \n"
        
        assert len(choices) == self.num_options, "Double check number of choices"
        for i, option in enumerate(choices):
            o_text += f"{self.answer_mapping[i]} {option}\n"
        
        a_text = f"Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'd_text': d_text}
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            if self.args.blank_video or self.args.no_modality_inp:
                video = torch.zeros(1, self.features_dim)
            else:
                video = self.features[video_id][start * 1: (end + 1) * 1, :].float() # 1fps from extraction
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len

    def _get_padding_id(self, text_id, prefix_index, prefix_i, prefix_main, type):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        
        prefix = prefix_index
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
                prefix = prefix_index
            else:
                if self.sub and prefix_i != prefix_main:
                    pad = self.max_seq_len - ((prefix_i) + (len(tid) - prefix_main))
                    padding_text_id[i, :prefix_i] = tid[:prefix_i]
                    padding_text_id[i, prefix_i: prefix_i + pad] = tid[prefix_i: prefix_i + pad]
                    padding_text_id[i, prefix_i + pad :] = tid[prefix_main:]

                    if type == "vqa":
                        prefix = len(padding_text_id[i]) - 4
                    elif type == "vaq":
                        if self.split == "train":
                            try:
                                prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1].item() + 2
                            except:
                                prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1][0].item() + 2
                        else:
                            prefix = (padding_text_id == self.tokenizer.q_token_id).nonzero(as_tuple=True)[1][0].item() + 2
                    else:
                        prefix = len(padding_text_id[i]) - self.max_feats - 1
                else:
                    padding_text_id[i] = tid[:self.max_seq_len]
                    prefix = prefix_index
                print('max sequence length overflow')

        return padding_text_id, prefix

    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_video_start, vqa_prefix_i, vqa_prefix_q = self.tokenizer.encode_dvqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vaq_id, vaq_prefix_index, vaq_video_start, vaq_prefix_i, vaq_prefix_q = self.tokenizer.encode_dvaq(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        qav_id, qav_prefix_index, qav_prefix_i, qav_prefix_q = self.tokenizer.encode_dqav(text=text, max_feats=self.max_feats, max_seq_len=self.max_seq_len, split=self.split, answer_mapping=self.answer_mapping, answer=answer)

        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]
        qav_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in qav_id]
        
        vqa_padding_text_id, vqa_prefix_index = self._get_padding_id(vqa_id, vqa_prefix_index, vqa_prefix_i, vqa_prefix_q, "vqa")
        vaq_padding_text_id, vaq_prefix_index = self._get_padding_id(vaq_id, vaq_prefix_index, vaq_prefix_i, vaq_prefix_q, "vaq")
        qav_padding_text_id, qav_prefix_index = self._get_padding_id(qav_id, qav_prefix_index, qav_prefix_i, qav_prefix_q, "qav")

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float() 
        
        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()
        
        qav_label = torch.ones_like(qav_padding_text_id) * -1
        qav_label[:, qav_prefix_index:qav_prefix_index+self.max_feats] = torch.arange(self.max_feats)
        qav_label_mask = torch.zeros_like(qav_padding_text_id)
        qav_label_mask[:, qav_prefix_index] = 1
        qav_label_mask = qav_label_mask.float()
                
        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0
        qav_text_mask = qav_padding_text_id.ge(0)
        qav_padding_text_id[~qav_text_mask] = 0
        
        # video index
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats)
        qav_video_index = torch.arange(qav_prefix_index, qav_prefix_index + self.max_feats)
        
        text_id = {'vqa': vqa_padding_text_id, 'vaq': vaq_padding_text_id, 'qav': qav_padding_text_id}
        label = {'vqa': vqa_label, 'vaq': vaq_label, 'qav': qav_label}
        video_start = {'vqa': vqa_video_start, 'vaq': vaq_video_start, 'qav': qav_prefix_index}
        video_index = {'vqa': vqa_video_index, 'vaq': vaq_video_index, 'qav': qav_video_index}
        label_mask = {'vqa': vqa_label_mask, 'vaq': vaq_label_mask, 'qav': qav_label_mask}
        return text_id, label, video_start, video_index, label_mask

    def __getitem__(self, idx):
        vid = self.data['video_id'].values[idx]
        qtype = -1
        # print(f'vid: {vid}')
        choices =  [ self.data[f'a{i}'].values[idx] for i in range(self.num_options)]
        # print(f'choices: {choices}')
        # answer =  self.data[idx]['answer_idx']
        answer = self.data['answer_id'].values[idx]

        # start, end = map(float, self.data['ts'].split('-'))

        # try: 
        #     start, end = round(start), round(end)
        # except: 
        #     start, end = -1000, 1000
        try:
            start = self.data['start'].values[idx]
            end = self.data['end'].values[idx]
        except:
            start, end = -1000, 1000

        video, video_len = self._get_video(f'{vid}', start, end)
        text = self._get_text(idx, choices, f'{vid}', start, end)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}


    def __len__(self):
        return len(self.data)