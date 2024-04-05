import os
from music21 import converter
import fractions
import json
import torch
class DataStorage:
    path='data/'
    def get_data(self):
        midis = []
        lyrics = []
        durations = []
        for i in self.generate():
            midis.append(i[0])
            durations.append(i[1])
            lyrics.append(i[2])
        return (midis,lyrics,durations)
    def generate(self,DIGITS=10):
        dur_dic = {}
        for file in os.listdir(self.path):
            lyrics = []
            midis = []
            durations = []
            xml = converter.parseFile(os.path.join(self.path, file))
            for i, note in enumerate(xml.recurse().notesAndRests):
                if i % DIGITS == 0:
                    lyrics.append([])
                    midis.append([])
                    durations.append([])
                lyric = note._getLyric()
                if lyric == None:
                    lyric = '#'
                lyrics[i // DIGITS].append(ord(lyric))
                try:
                    midis[i // DIGITS].append(note.pitch.midi)
                except:
                    midis[i // DIGITS].append(0)
                durations[i // DIGITS].append(note.duration.quarterLength)
                if type(note.duration.quarterLength) == fractions.Fraction and float(
                        note.duration.quarterLength) not in list(dur_dic.values()):
                    dur_dic[len(dur_dic)] = float(note.duration.quarterLength)
                elif type(
                        note.duration.quarterLength) != fractions.Fraction and note.duration.quarterLength not in list(
                        dur_dic.values()):
                    dur_dic[len(dur_dic)] = note.duration.quarterLength
            yield [midis, durations, lyrics]
        with open('dur_dic.json', 'w') as f:
            f.write(json.dumps(dur_dic))