import re
import os
from kgc_base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
dataset = 'FB15k-237'
cased = True


class FB15k237_Processor(Processor):
    def __init__(self, in_folder, out_folder, dataset):
        super().__init__(in_folder, out_folder, dataset)

    def create_ent2name(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            ent, name = line.split('\t')
            name = name.strip('\"').replace(r'\n', '').replace(r'\t', '').replace('\\', '')
            if ent not in self.ent2name:
                self.ent2name[ent] = name
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2descrip(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            ent, descrip = line.split('\t')
            if descrip.endswith('@en'):
                descrip = descrip[:-3]
            descrip = descrip.strip('\"').replace(r'\n', '').replace(r'\t', '').replace('\\', '')
            if ent not in self.ent2descrip:
                self.ent2descrip[ent] = descrip
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2id(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            ent = line.strip()
            self.ent2id[ent] = i
            self.entid2name[i] = self.ent2name[ent]
            if ent in self.ent2descrip:
                self.entid2descrip[i] = self.ent2descrip[ent]
            else:
                self.entid2descrip[i] = ''

    def create_rel2id(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            rel = line.split('\t')[0]
            self.rel2id[rel] = i
            self.relid2name[i] = rel

processor = FB15k237_Processor(in_folder, out_folder, dataset)
processor.create_out_folder()
processor.create_ent2name('entity2text.txt')
processor.create_ent2descrip('entity2textlong.txt')
processor.create_ent2id('entities.txt')
processor.create_rel2id('relation2text.txt')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt')
processor.write_file('relationid2name.txt', func=lambda x: x.strip('/').replace('/', ' , ').replace('_', ' '))
processor.write_file('entityid2description.txt')

in_files = ['train.tsv', 'dev.tsv', 'test.tsv']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(3):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file)
    processor.write_triples(out_file, triples)