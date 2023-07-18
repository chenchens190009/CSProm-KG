from kgc_base import Processor

in_folder = '../data/raw'
out_folder = '../data/processed'
dataset = 'wikidata5m_transductive'
cased = True

class Wikidata5m_Processor(Processor):
    def __init__(self, in_folder, out_folder, dataset):
        super().__init__(in_folder, out_folder, dataset)
        self.cased = cased


    def create_ent2name(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            split = line.split('\t')
            ent, name = split[0], split[1]
            if ent not in self.ent2name:
                self.ent2name[ent] = name
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2descrip(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            split = line.split('\t')
            ent, descrip = split[0], ' '.join(split[1:])
            descrip = descrip[:400]
            if ent not in self.ent2descrip:
                self.ent2descrip[ent] = descrip
            else:
                raise ValueError('%s dupliated entities!' % ent)

    def create_ent2id(self, filenames):
        all_ents = set()
        for filename in filenames:
            lines = self.read_file(filename)
            for line in lines:
                split = line.split('\t')
                all_ents.add(split[0])
                all_ents.add(split[2])
        lines = list(all_ents)
        for i, line in enumerate(lines):
            ent = line.strip()
            self.ent2id[ent] = i
            if ent in self.ent2name:
                self.entid2name[i] = self.ent2name[ent]
            else:
                self.entid2name[i] = ' '
            if ent in self.ent2descrip:
                self.entid2descrip[i] = self.ent2descrip[ent]
            else:
                self.entid2descrip[i] = ''

    def create_rel2id(self, filename):
        lines = self.read_file(filename)
        for i, line in enumerate(lines):
            split = line.split('\t')
            rel, relname = split[0], split[1]
            self.rel2id[rel] = i
            self.relid2name[i] = relname

    def check_ent2name_ent2descrip(self):
        for i in range(len(self.ent2id)):
            if i not in self.ent2name:
                self.ent2name[i] = ''
            if i not in self.ent2descrip:
                self.entid2descrip[i] = ''

processor = Wikidata5m_Processor(in_folder, out_folder, dataset)
processor.create_out_folder()
processor.create_ent2name('wikidata5m_entity.txt')
processor.create_ent2descrip('wikidata5m_text.txt')
processor.create_ent2id(['wikidata5m_transductive_train.txt', 'wikidata5m_transductive_valid.txt', 'wikidata5m_transductive_test.txt'])
processor.create_rel2id('wikidata5m_relation.txt')
processor.write_file('entity2id.txt', sort_key=lambda x: x[1])
processor.write_file('relation2id.txt', sort_key=lambda x: x[1])
processor.write_file('entityid2name.txt')
processor.write_file('relationid2name.txt', func=lambda x: x)
processor.write_file('entityid2description.txt')

in_files = ['wikidata5m_transductive_train.txt', 'wikidata5m_transductive_valid.txt', 'wikidata5m_transductive_test.txt']
out_files = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
for i in range(3):
    in_file, out_file = in_files[i], out_files[i]
    triples = processor.read_triples(in_file)
    processor.write_triples(out_file, triples)