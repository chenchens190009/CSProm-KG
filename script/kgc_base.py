import os
import json


class Processor:
    def __init__(self, in_folder, out_folder, dataset):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.dataset = dataset
        self.ent2id = {}
        self.id2ent = {}
        self.rel2id = {}
        self.id2rel = {}
        self.ent2name = {}
        self.ent2descrip = {}
        self.entid2name = {}
        self.entid2descrip = {}
        self.relid2name = {}

    def create_out_folder(self):
        os.makedirs(os.path.join(self.out_folder, self.dataset), exist_ok=True)

    def read_file(self, filename, in_folder=None, dataset=None):
        in_folder = self.in_folder if in_folder is None else in_folder
        dataset = self.dataset if dataset is None else dataset
        with open(os.path.join(in_folder, dataset, filename), encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
        return lines

    def read_json_file(self, filename, in_folder=None, dataset=None):
        in_folder = self.in_folder if in_folder is None else in_folder
        dataset = self.dataset if dataset is None else dataset
        return json.load(open(os.path.join(in_folder, dataset, filename)))

    def write_file(self, filename, out_folder=None, dataset=None, sort_key=None, func=lambda x: x):
        out_folder = self.out_folder if out_folder is None else out_folder
        dataset = self.dataset if dataset is None else dataset
        if filename == 'entity2id.txt':
            target_dict = self.ent2id
        elif filename == 'entityid2name.txt':
            target_dict = self.entid2name
        elif filename == 'entityid2description.txt':
            target_dict = self.entid2descrip
        elif filename == 'relation2id.txt':
            target_dict = self.rel2id
        elif filename == 'relationid2name.txt':
            target_dict = self.relid2name
        else:
            raise ValueError('Unknown file name!')
        target_list = sorted(list(target_dict.items()), key=sort_key)
        with open(os.path.join(out_folder, dataset, filename), 'w', encoding='utf-8') as file:
            file.write(str(len(target_list)) + '\n')
            for x, y in target_list:
                y = func(y)
                file.write(str(x) + '\t' + str(y) + '\n')

    def read_triples(self, filename, in_folder=None, dataset=None):
        in_folder = self.in_folder if in_folder is None else in_folder
        dataset = self.dataset if dataset is None else dataset
        triples = list()
        lines = self.read_file(filename, in_folder, dataset)
        for line in lines:
            head, rel, tail = line.split('\t')
            if rel in self.rel2id:
                head_id, rel_id, tail_id = self.ent2id[head], self.rel2id[rel], self.ent2id[tail]
                triples.append([head_id, tail_id, rel_id])
        return triples

    def write_triples(self, filename, triples, out_folder=None, dataset=None):
        out_folder = self.out_folder if out_folder is None else out_folder
        dataset = self.dataset if dataset is None else dataset
        name_filename = filename.split('.')[0] + '_name.txt'
        with open(os.path.join(out_folder, dataset, filename), 'w', encoding='utf-8') as file, \
                open(os.path.join(out_folder, dataset, name_filename), 'w', encoding='utf-8') as name_file:
            file.write(str(len(triples)) + '\n')
            name_file.write(str(len(triples)) + '\n')
            for head, tail, rel in triples:
                file.write(str(head) + ' ' + str(tail) + ' ' + str(rel) + '\n')
                name_file.write(self.entid2name[head] + ' | ' + self.entid2name[tail] + ' | ' + self.relid2name[rel] + '\n')