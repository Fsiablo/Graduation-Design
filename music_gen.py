from music21 import *
import json
from inference import inference_midi,inference_dur
def music_gen(input_lyrics):
    results=inference_midi(input_lyrics)
    durations=inference_dur(results)
    dur_dic = {}
    with open('dur_dic.json', 'r') as f:
        dur_str = f.readline()
        dur_dic = json.loads(dur_str)
        print(dur_dic)

    stream1 = stream.Stream()
    for i, lyric in enumerate(input_lyrics.replace('\n', '')):
        if results[i] != 0:
            n1 = note.Note(results[i])
        else:
            n1 = note.Rest()
        n1.addLyric(lyric)
        n1.duration = duration.Duration(dur_dic[str(durations[i])])
        stream1.append(n1)
    stream1.write("xml", "out.xml")
    stream1.write('midi', 'out.midi')