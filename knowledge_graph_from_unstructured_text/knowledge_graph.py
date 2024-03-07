import nltk
import sys
import pickle
import os
import json
from collections import defaultdict
import glob
from collections import Counter
import en_core_web_sm
from pprint import pprint
import spacy
from stanfordcorenlp import StanfordCoreNLP

class StanfordNER:
    def __init__(self):
        self.get_stanford_ner_location()

    def get_stanford_ner_location(self):
        print("Provide (relative/absolute) path to stanford ner package.\nPress carriage return to use './stanford-ner-2018-10-16' as path:")
        loc = input()
        print("... Running stanford for NER; this may take some time ...")
        if not loc:
            loc = "./stanford-ner-2018-10-16"
        self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(
            loc+'/classifiers/english.all.3class.distsim.crf.ser.gz',
            loc+'/stanford-ner.jar')

    def ner(self, doc):
        sentences = nltk.sent_tokenize(doc)
        result = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            tagged = self.stanford_ner_tagger.tag(words)
            result.append(tagged)
        return result

    def display(self, ner):
        print(ner)
        print("\n")

class SpacyNER:
    def ner(self, doc):
        nlp = en_core_web_sm.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]

    def ner_to_dict(self, ner):
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict

    def display(self, ner):
        print(ner)
        print("\n")

class NltkNER:
    def ner(self, doc):
        pos_tagged = self.assign_pos_tags(doc)
        result = []
        for sent in pos_tagged:
            result.append(nltk.ne_chunk(sent))
        return result

    def assign_pos_tags(self, doc):
        sentences = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        pos_tagged = [nltk.pos_tag(word) for word in words]
        return pos_tagged

    def display(self, ner):
        print("\n\nTagged: \n\n")
        pprint(ner)
        print("\n\nTree: \n\n ")
        for leaves in ner:
            print(leaves)
        print("\n")

class CoreferenceResolver:
    def generate_coreferences(self, doc, stanford_core_nlp_path, verbose):
        nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=not verbose)
        props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
        annotated = nlp.annotate(doc, properties=props)
        print("\nannotated\n\n", annotated, "\n\n")
        result = json.loads(annotated)
        pickle.dump(result, open("coref_res.pickle", "wb"))
        nlp.close()
        return result

    def display_dict(self, result):
        for key in result:
            print(key, ":\n", result[key])
            print("\n")

    def unpickle(self):
        result = pickle.load(open("coref_res.pickle", "rb"))
        return result

    def resolve_coreferences(self, corefs, doc, ner, verbose):
        corefs = corefs['corefs']
        if verbose:
            print("Coreferences found: ", len(corefs),
                  "\nThe coreferences are:")
            self.display_dict(corefs)
            print("Named entities:")
            print(ner.keys())

        replace_coref_with = []
        sentence_wise_replacements = defaultdict(list)

        sentences = nltk.sent_tokenize(doc)
        for index, coreferences in enumerate(corefs.values()):
            replace_with = coreferences[0]
            for reference in coreferences:
                if reference["text"] in ner.keys() or reference["text"][reference["headIndex"]-reference["startIndex"]] in ner.keys():
                    replace_with = reference
                sentence_wise_replacements[reference["sentNum"]-1].append(
                    (reference, index))
            replace_coref_with.append(replace_with["text"])

        sentence_wise_replacements[0].sort(
            key=lambda tup: tup[0]["startIndex"])

        if verbose:
            for key, val in sentence_wise_replacements.items():
                print("Sent no# ", key)
                for item in val:
                    print(item[0]["text"], " ", item[0]["startIndex"],
                          " ", item[0]["endIndex"], " -> ", replace_coref_with[item[1]], " replacement correl #", item[1], end="   ||| ")
                print("\n")

        for index, sent in enumerate(sentences):
            replacement_list = sentence_wise_replacements[index]
            for item in replacement_list[::-1]:
                to_replace = item[0]
                replace_with = replace_coref_with[item[1]]
                replaced_sent = ""
                words = nltk.word_tokenize(sent)

                for i in range(len(words)-1, to_replace["endIndex"]-2, -1):
                    replaced_sent = words[i] + " " + replaced_sent
                replaced_sent = replace_with + " " + replaced_sent
                for i in range(to_replace["startIndex"]-2, -1, -1):
                    replaced_sent = words[i] + " " + replaced_sent
                sentences[index] = replaced_sent

        result = ""
        for sent in sentences:
            result += sent
        if verbose:
            print("Original text: \n", doc)
            print("Resolved text:\n ", result)
        return result


def resolve_coreferences(doc, stanford_core_nlp_path, ner, verbose):
    coref_obj = CoreferenceResolver()
    corefs = coref_obj.generate_coreferences(doc, stanford_core_nlp_path, verbose)
    result = coref_obj.resolve_coreferences(corefs, doc, ner, verbose)
    return result


def main():
    if len(sys.argv) == 1:
        print("Usage:   python3 knowledge_graph.py <nltk/stanford/spacy> [optimized,verbose,nltk,stanford,spacy]")
        return None

    verbose = False
    execute_coref_resol = False
    output_path = "./data/output/"
    ner_pickles_op = output_path + "ner/"
    coref_cache_path = output_path + "caches/"
    coref_resolved_op = output_path + "kg/"

    stanford_core_nlp_path = input(
        "\n\nProvide (relative/absolute) path to stanford core nlp package.\n Press carriage return to use './stanford-corenlp-full-2018-10-05' as path:")
    if not stanford_core_nlp_path:
        stanford_core_nlp_path = "./stanford-corenlp-full-2018-10-05"

    file_list = glob.glob('./data/input/*')

    for file in file_list:
        with open(file, "r") as f:
            lines = f.read().splitlines()

        doc = "".join(lines)

        if verbose:
            print("Read: \n", doc)

        for arg in sys.argv[1:]:
            if arg == "nltk":
                print("\nusing NLTK for NER")
                nltk_ner = NltkNER()
                named_entities = nltk_ner.ner(doc)
                nltk_ner.display(named_entities)
                spacy_ner = SpacyNER()
                named_entities = spacy_ner.ner_to_dict(spacy_ner.ner(doc))
            elif arg == "stanford":
                print("using Stanford for NER (may take a while):  \n\n\n")
                stanford_ner = StanfordNER()
                tagged = stanford_ner.ner(doc)
                ner = stanford_ner.ner(doc)
                stanford_ner.display(ner)
                named_entities = spacy_ner.ner_to_dict(spacy_ner.ner(doc))
            elif arg == "spacy":
                print("\nusing Spacy for NER\n")
                spacy_ner = SpacyNER()
                named_entities = spacy_ner.ner(doc)
                spacy_ner.display(named_entities)
                named_entities = spacy_ner.ner_to_dict(named_entities)
            elif arg == "verbose":
                verbose = True
            elif arg == "optimized":
                execute_coref_resol = True

            op_pickle_filename = ner_pickles_op + "named_entity_" + \
                file.split('/')[-1].split('.')[0] + ".pickle"
            os.makedirs(os.path.dirname(op_pickle_filename), exist_ok=True)
            with open(op_pickle_filename, "wb") as f:
                pickle.dump(named_entities, f)

        if execute_coref_resol:
            print("\nResolving Coreferences... (This may take a while)\n")
            doc = resolve_coreferences(
                doc, stanford_core_nlp_path, named_entities, verbose)

        op_filename = coref_resolved_op + file.split('/')[-1]
        os.makedirs(os.path.dirname(op_filename), exist_ok=True)
        with open(op_filename, "w+") as f:
            f.write(doc)



if __name__ == "__main__":
    main()
