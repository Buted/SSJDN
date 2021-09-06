import json
from torchtext.data import Field, Example

from enet.corpus.Corpus import Corpus
from enet.corpus.Sentence import Sentence


class WordField(Field):
    '''
    Process word each sentence, recording the tokenization of every word.
    '''
    def preprocess(self, x):
        dic = {}
        dic["_LIST_"] = x
        for i, w in enumerate(x):
            dic[w] = self.tokenize(w)
            
        return dic
    
    def pad(self, minibatch):
        return minibatch
    
    def numericalize(self, arr, device=None, train=True):
        return arr


class ACE2005Dataset(Corpus):
    """
    Defines a dataset composed of Examples along with its Fields.
    """

    sort_key = None

    def __init__(self, path, fields, min_len=None, keep_events=None, only_keep=False, include_contigous=False,
                 tokenizer=None, word_id=None, **kwargs):
        '''
        Create a corpus given a path, field list, and a filter function.

        :param path: str, Path to the data file
        :param fields: dict[str: tuple(str, Field)],
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
        :param min_len: int, min length of the sentence used for training.
        :param only_keep: bool, whether or not keep the non-event sentence.
        :param include_contigous: bool, whether or not keep the multi-word trigger
        :param word_id: dict, word2id
        :param keep_events: int, minimum sentence events. Default keep all.
        '''
        self.keep_events = keep_events
        self.only_keep = only_keep
        self.min_len = min_len
        self.include_contigous = include_contigous
        self.tokenizer = tokenizer
        self.word_id = word_id
        self.all_word_id = kwargs.pop("all_word_id", None)
        super(ACE2005Dataset, self).__init__(path, fields, **kwargs)

    def parse_example(self, path: str, fields):
        """
        Parse the dataset for the model from file.
        :param path: str, dataset json file
        :param fields: Dict[str, Field]. Data Fields for parsing the sentence.
        :return: the parsed data.
        """
        examples = []

        num_of_trigger_words_more_than_one = 0
        no_event = 0
        one_event = 0
        more_event = 0
        using_no_event = 0
        with open(path, "r", encoding="utf-8") as f:
                jl = json.load(f, encoding="utf-8")
                for js in jl:
                    events = js["golden-event-mentions"]
                    if len(events) != 0:
                        if len(events) == 1:
                            one_event += 1
                        else:
                            more_event += 1
                        
                        for e in events:
                            if e["trigger"]["end"] - e["trigger"]["start"] != 1:
                                num_of_trigger_words_more_than_one += 1
                    else:
                        no_event += 1
                        
                    try:
                        exs = self.parse_sentence(js, fields)
                    except Exception as e:
                        print("JSON_ERROR\n",js)
                        print(e.message)
                        exit(-1)
                        
                    if len(exs) != 0:
                        if len(events) == 0:
                            using_no_event += 1
                        examples.extend(exs)

        print("-"*20)
        print("Sentences:", len(jl))
        print("trigger words length more than 1 :", num_of_trigger_words_more_than_one)
        print("no event:", no_event)
        print("using no event:", using_no_event)
        print("1 event:", one_event)
        print("1+ event:", more_event)
        print("-"*20 + "\n")
        
        return examples

    def parse_sentence(self, js, fields):
        """
        Parse the sample for the model.
        :param js: sample json str
        :param fields: Dict[str, Field]. Data Fields for parsing the sentence.
        :return: the parsed sample.
        """
        WORDS = fields["words"]
        WLIST = fields["word-list"]
        LABELS = fields["golden-event-mentions"]
        TOKENLABELS = fields["golden-token-labels"]
        WIDS = fields["word-ids"]
        WLABELIDS = fields["word-label-ids"]

        sentence = Sentence(json_content=js, include_contigous=self.include_contigous, tokenizer=self.tokenizer,
                            word_id=self.word_id, all_word_id=self.all_word_id)

        exs = []
        if self.keep_events is not None:
            # 事件不足
            if self.only_keep and sentence.containsEvents != self.keep_events:
                return exs
            elif not self.only_keep and sentence.containsEvents < self.keep_events:
                return exs
            elif self.min_len is not None and sentence.word_len < self.min_len:
                if sentence.containsEvents == 0:
                    return exs

        ex = Example()
        setattr(ex, WORDS[0], WORDS[1].preprocess(sentence.sentence))
        setattr(ex, WLIST[0], WLIST[1].preprocess(sentence.wordList))
        setattr(ex, LABELS[0], LABELS[1].preprocess(sentence.triggerLabelList))
        setattr(ex, TOKENLABELS[0], TOKENLABELS[1].preprocess(sentence.tokenLabelList))
        setattr(ex, WIDS[0], WIDS[1].preprocess(sentence.gloveId))
        setattr(ex, WLABELIDS[0], WLABELIDS[1].preprocess(sentence.wordId))
        exs.append(ex)
        return exs
    
    def longest(self):
        return max([len(x.WORDS) for x in self.examples])
