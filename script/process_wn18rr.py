from kgc_base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
dataset = 'WN18RR'


class WN18RR_Processor(Processor):
    def __init__(self, in_folder, out_folder, dataset):
        super().__init__(in_folder, out_folder, dataset)
        self.non_capitalized_list = ['a', 'an', 'the', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'at', 'around', 'by', 'after', 'along', 'from', 'of', 'on', 'to', 'with', 'without']

    def create_ent2name_ent2descrip(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            ent, name, description = line.split('\t')
            if ent not in self.ent2name:
                self.ent2name[ent] = name
                self.ent2descrip[ent] = description
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2id(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            ent = line.strip()
            if ent in self.ent2name:
                self.ent2id[ent] = i
                self.entid2name[i] = self.ent2name[ent]
                self.entid2descrip[i] = self.ent2descrip[ent]
            else:
                raise ValueError('%s entity not in entity list' % line)

    def create_rel2id(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            rel = line.strip()
            self.rel2id[rel] = i
            self.relid2name[i] = rel

    @staticmethod
    def convert_ent_name(name):
        split = name.strip('_').split('_')
        tokens, POS, num = split[:-2], split[-2], split[-1]
        tokens = ' '.join(tokens)
        name = tokens + ' , ' + POS + ' , ' + num
        return name

print('preprocessing WN18RR...')
processor = WN18RR_Processor(in_folder, out_folder, dataset)
processor.create_out_folder()
processor.create_ent2name_ent2descrip('wordnet-mlj12-definitions.txt')
processor.create_ent2id('entities.txt')
processor.create_rel2id('relations.txt')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt', func=processor.convert_ent_name)
processor.write_file('relationid2name.txt', func=lambda x: x.strip('_').replace('_', ' '))
processor.write_file('entityid2description.txt')

in_files = ['train.tsv', 'dev.tsv', 'test.tsv']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(len(in_files)):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file)
    processor.write_triples(out_file, triples)