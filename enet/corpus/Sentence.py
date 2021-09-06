from enet.consts import CUTOFF


def pretty_str(a):
    a = a.upper()
    if a[:4] == "TIME":
        a = "TIME"
        
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


class Sentence:
    def __init__(self, json_content, include_contigous=False, tokenizer=None, word_id=None,
                 all_word_id=None):
        """
        Generate the parsed sample, basing the json str.
        :param json_content: json sample str
        :param all_word_id: dict[str, int], word2id, including the word not in trainset but in glove vocabulary
        """
        if tokenizer is None:
            raise ValueError("tokenizer can't be None")

        self.contigous_trigger = include_contigous
        self.word_id = word_id
        self.all_word_id = all_word_id
        # words
        self.wordList = json_content["words"][:CUTOFF]
        self.gloveId = self.generateWordIdList(self.all_word_id)
        self.wordId = self.generateWordIdList(self.word_id)
        # self.tokenList = []
        self.word_len = len(self.wordList)
        
        # 构造后的句子
        self.sentence = " ".join(["[CLS]"] + self.wordList)
        self.tokens_dict = {}
        for w in self.wordList:
            self.tokens_dict[w] = tokenizer.tokenize(w)
        
        # trigger 标签， 针对words，去除了连续词的可能性
        self.triggerLabelList = self.generateTriggerLabelList(json_content["golden-event-mentions"])

        # token labels
        self.tokenLabelList = self.generateTokenLabelList()
        # events
        self.events = self.generateGoldenEvents(json_content["golden-event-mentions"])
        self.containsEvents = len(self.events.keys())
        
        # token
        self.tokenList = self.makeTokenList()

    def generateGoldenEvents(self, eventsJson):
        '''

        {
            (2, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
        }

        '''
        golden_dict = {}
        for eventJson in eventsJson:
            triggerJson = eventJson["trigger"]
            # 去除连续词的可能性
            if triggerJson["start"] >= CUTOFF or triggerJson["end"] - triggerJson["start"] != 1:
                continue
                
            key = (triggerJson["start"], pretty_str(eventJson["event_type"]))
            values = []
            for argumentJson in eventJson["arguments"]:
                if argumentJson["start"] >= CUTOFF:
                    continue
                start = argumentJson["start"]
                end = min(argumentJson["end"], CUTOFF)
                entity = " ".join(self.wordList[start:end])
                value = (start, end, entity, pretty_str(argumentJson["role"]))
                values.append(value)
                
            golden_dict[key] = list(sorted(values))
        return golden_dict

    def generateTriggerLabelList(self, triggerJsonList):
        """
        Generate the labels of words.
        :param triggerJsonList: the event json str
        :return: Label list of words.
        """
        # 针对原来每个词做trigger
        triggerLabel = ["O" for _ in range(self.word_len)]

        def assignTriggerLabel(index, label):
            if index >= CUTOFF:
                return
            triggerLabel[index] = pretty_str(label)
        last_event = None
        for eventJson in triggerJsonList:
            triggerJson = eventJson["trigger"]
            start = triggerJson["start"]
            end = triggerJson["end"]
            if end - start != 1 and self.contigous_trigger == False:
                continue
            etype = eventJson["event_type"]
            for i in range(start, end):
                assignTriggerLabel(i, etype)
            
        return triggerLabel

    def generateTokenLabelList(self):
        """
        Generate label of tokens. not used in SSJDN
        :return:
        """
        # 针对原来每个词做trigger
        tokenLabel = [] # exclude CLS
        for word, label in zip(self.wordList, self.triggerLabelList):
            tokenLabel.extend([label]*len(self.tokens_dict[word]))
        return tokenLabel

    def generateWordIdList(self, word_dict):
        """
        Generate the ids of words.
        :param word_dict: word2id dict
        :return: Id list of words.
        """
        wordId = [] # exclude CLS
        for word in self.wordList:
            if word not in word_dict:
                wordId.append(1)
                continue
            wordId.append(word_dict[word])
        return wordId

    def makeTokenList(self):
        return [Token(self.wordList[i], self.triggerLabelList[i]) for i in range(self.word_len)]

    def __len__(self):
        return self.word_len

    def __iter__(self):
        for x in self.tokenList:
            yield x

    def __getitem__(self, index):
        return self.tokenList[index]


class Token:
    def __init__(self, word, triggerLabel):
        self.word = word
        self.triggerLabel = triggerLabel
        self.predictedLabel = None

    def addPredictedLabel(self, label):
        self.predictedLabel = label
