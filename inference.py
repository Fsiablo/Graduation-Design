from settings import parse_args
import torch
import numpy as np
import json
from model_v2 import Model
def inference_midi(input_lyrics):
    args=parse_args()
    batch_size=args.batch_size
    lyrics = []
    for i, lyric in enumerate(input_lyrics.replace('\n', '')):
        if i % batch_size == 0:
            lyrics.append([])
        lyrics[i // batch_size].append(ord(lyric))
    while len(lyrics[-1]) % batch_size != 0:
        lyrics[-1].append(ord('#'))
    lyrics = torch.tensor(lyrics)

    params_dict = torch.load('Midi_Model/best_model')
    midi_model=Model(args)
    midi_model.load_state_dict(params_dict)

    # 设置为评估模式
    midi_model.eval()

    # 模型推理
    out = midi_model(lyrics)

    # 结果转换
    with torch.no_grad():
        results = []
        for _ in np.argmax(out.numpy(), -1).reshape(-1):
            results.append(_)

    return results

def inference_dur(results):
    midis = []
    args=parse_args()
    batch_size=args.batch_size
    dur_dic = {}
    with open('dur_dic.json', 'r') as f:
        dur_str = f.readline()
        dur_dic = json.loads(dur_str)
    for i, midi in enumerate(results):
        if i % batch_size == 0:
            midis.append([])
        midis[i // batch_size].append(midi) if midi <= 200 else midis[i // batch_size].append(0)
    while len(midis[-1]) % batch_size != 0:
        midis[-1].append(0)
    midis = torch.tensor(midis)

    params_dict = torch.load('Duration_Model/best_model')
    dur_model = Model(args)
    dur_model.load_state_dict(params_dict)

    # 设置为评估模式
    dur_model.eval()

    # 模型推理
    # out = nn.Softmax(dur_model(midis))
    out = dur_model(midis)

    # 结果转换
    with torch.no_grad():
        durations = []
        for _ in np.argmax(out.numpy(), -1).reshape(-1):
            durations.append(_)

    return durations