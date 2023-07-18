import os
from kgc_base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
dataset = 'ICEWS14'


class ICEWS14_Processor(Processor):
    def __init__(self, in_folder, out_folder, dataset):
        super().__init__(in_folder, out_folder, dataset)
        self.month2text = {
            '01': ['January', 'Jan'],
            '02': ['February', 'Feb'],
            '03': ['March', 'Mar'],
            '04': ['April', 'Apr'],
            '05': ['May', 'May'],
            '06': ['June', 'Jun'],
            '07': ['July', 'Jul'],
            '08': ['August', 'Aug'],
            '09': ['September', 'Sep'],
            '10': ['October', 'Oct'],
            '11': ['November', 'Nov'],
            '12': ['December', 'Dec'],
        }

    def create_ent2id_rel2id_entid2name_relid2name(self):
        train_lines = self.read_file('icews_2014_train.txt')
        valid_lines = self.read_file('icews_2014_valid.txt')
        test_lines = self.read_file('icews_2014_test.txt')
        lines = train_lines + valid_lines + test_lines
        for line in lines:
            h, r, t, time = line.split('\t')
            h, r, t = h.strip(), r.strip(), t.strip()
            if h not in self.ent2id:
                self.ent2id[h] = len(self.ent2id)
                self.id2ent[len(self.ent2id)] = h
                self.entid2name[len(self.entid2name)] = h
            if t not in self.ent2id:
                self.ent2id[t] = len(self.ent2id)
                self.id2ent[len(self.ent2id)] = t
                self.entid2name[len(self.entid2name)] = t
            if r not in self.rel2id:
                self.rel2id[r] = len(self.rel2id)
                self.id2rel[len(self.rel2id)] = r
                self.relid2name[len(self.relid2name)] = r

    def create_entid2descrip(self, filename):
        lines = self.read_file(filename)
        all_ent2descrip = {}
        for line in lines:
            split = line.split('\t')
            src_name, src_descrip, src_country = split[2].strip(), split[3].strip(), split[4].strip()
            tgt_name, tgt_descrip, tgt_country = split[8].strip(), split[9].strip(), split[10].strip()
            if src_name not in all_ent2descrip:
                all_ent2descrip[src_name] = src_descrip + ' (' + src_country + ')'
            if tgt_name not in all_ent2descrip:
                all_ent2descrip[tgt_name] = tgt_descrip + ' (' + tgt_country + ')'
        for entid, entname in self.entid2name.items():
            if entname in all_ent2descrip:
                self.entid2descrip[entid] = all_ent2descrip[entname]
            else:
                self.entid2descrip[entid] = ' '

    def read_triples(self, filename, in_folder=None, dataset=None):
        in_folder = self.in_folder if in_folder is None else in_folder
        dataset = self.dataset if dataset is None else dataset
        triples = list()
        lines = self.read_file(filename, in_folder, dataset)
        for i, line in enumerate(lines):

            head, rel, tail, time = line.split('\t')

            if rel in self.rel2id:
                head, rel, tail = head.strip(), rel.strip(), tail.strip()
                head_id, rel_id, tail_id = self.ent2id[head], self.rel2id[rel], self.ent2id[tail]
                triples.append([head_id, tail_id, rel_id, time])

        return triples

    def write_triples(self, filename, triples, out_folder=None, dataset=None):
        out_folder = self.out_folder if out_folder is None else out_folder
        dataset = self.dataset if dataset is None else dataset
        name_filename = filename.split('.')[0] + '_name.txt'
        with open(os.path.join(out_folder, dataset, filename), 'w', encoding='utf-8') as file, \
                open(os.path.join(out_folder, dataset, name_filename), 'w', encoding='utf-8') as name_file:
            file.write(str(len(triples)) + '\n')
            name_file.write(str(len(triples)) + '\n')
            for head, tail, rel, time in triples:
                file.write(str(head) + ' ' + str(tail) + ' ' + str(rel) + ' ' + self.convert_time(time) + '\n')
                name_file.write(self.entid2name[head] + ' | ' + self.entid2name[tail] + ' | ' + self.relid2name[rel] + ' | ' + self.convert_time(time) + '\n')

    def convert_time(self, time):
        y, m, d = time.split('-')
        return self.month2text[m][1] + '-' + d

print('preprocessing ICEWS14...')
processor = ICEWS14_Processor(in_folder, out_folder, dataset)
processor.create_out_folder()
processor.create_ent2id_rel2id_entid2name_relid2name()
processor.create_entid2descrip('icews14_data_source.tab')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt')
processor.write_file('relationid2name.txt')
processor.write_file('entityid2description.txt')

in_files = ['icews_2014_train.txt', 'icews_2014_valid.txt', 'icews_2014_test.txt']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(3):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file)
    processor.write_triples(out_file, triples)