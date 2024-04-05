# -*- coding:utf-8 -*-
import os
import jieba
import torch
PATH=r"D:\study\Graduation project\fluidsynth\bin"
def poem2seq(poem):
    from jiayan import load_lm
    from jiayan import CharHMMTokenizer

    lm = load_lm('CRF_model/jiayan.klm')
    tokenizer = CharHMMTokenizer(lm)
    print(list(tokenizer.tokenize(poem)))
def seq2mid(seq,interval=1/4):
    pattern=''
    seq_drum=drum(pattern='i:0.5,S,S,K,S,S,i:0.5,H,r:4',default_interval=interval)
    write(seq_drum,name='mids/seq_drum.mid')
def mid2wav():
    command='fluidsynth -ni ".fluidsynth/dgx62mbsf.sf2" "out.midi" -F wavs/seq_drum.wav -r 44100'
    os.environ['Path']=(os.environ['Path']+';'+PATH)
    os.system(command)
mid2wav()