import collections
import dynet as dy
import numpy as np
from pulp import *
from main import check_overlap
from main import get_all_spans
from trees import InternalParseNode, LeafParseNode
from trees import ParseNode

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
DELETABLE_TAGS = {',', ':', '``', "''", '.'}


def optimal_tree_construction(span_to_label, sentence, span_to_on_score):
    conflicting = set()
    for span_a in span_to_label:
        for span_b in span_to_label:
            if check_overlap(span_a, span_b):
                conflicting.add(span_a)
    cache = {}

    def helper(left, right):
        if (left, right) in cache:
            return cache[(left, right)]

        if (left, right) in span_to_label:
            label = span_to_label[(left, right)]
            assert label != ()
        else:
            assert left != 0 or right != len(sentence)
            label = ()

        if right - left == 1:
            tag, word = sentence[left]
            tree = LeafParseNode(left, tag, word)
            score = 0
            if label:
                tree = InternalParseNode(label, [tree])
                score += span_to_on_score[(left, right)]
            return [tree], score

        split_options = []
        for split in range(right - 1, left, -1):
            if (left, split) in span_to_label:
                split_options.append(split)
                if (left, split) not in conflicting:
                    break
            if split == left + 1:
                split_options.append(left + 1)
        assert len(split_options) > 0
        best_option_score = None
        best_option = None
        for split in split_options:
            left_trees, left_score = helper(left, split)
            right_trees, right_score = helper(split, right)
            children = left_trees + right_trees
            score = left_score + right_score
            if label:
                children = [InternalParseNode(label, children)]
                score += span_to_on_score[(left, right)]

            if best_option_score is None or score > best_option_score:
                best_option_score = score
                best_option = children
        response = best_option, best_option_score
        cache[(left, right)] = response
        return response

    trees, _ = helper(0, len(sentence))
    assert (0, len(sentence)) in span_to_label
    assert len(trees) == 1, len(trees)
    return trees[0]


def optimal_parser(label_log_probabilities_np,
                   span_to_index,
                   sentence,
                   empty_label_index,
                   label_vocab):
    assert (0, len(sentence)) in span_to_index, (span_to_index, len(sentence))
    on_spans = []
    for (start, end), span_index in span_to_index.items():
        off_score = label_log_probabilities_np[empty_label_index, span_index]
        on_score = np.max(label_log_probabilities_np[1:,
                          span_index])
        if on_score > off_score or (start == 0 and end == len(sentence)):
            label_index = label_log_probabilities_np[1:, span_index].argmax() + 1
            on_spans.append((start, end, on_score - off_score, None, label_index))

    span_to_label = {}
    span_to_on_score = {}
    for choice in on_spans:
        span_to_label[(choice[0], choice[1])] = label_vocab.value(choice[4])
        span_to_on_score[(choice[0], choice[1])] = choice[2]

    assert (0, len(sentence)) in span_to_label, span_to_label
    return optimal_tree_construction(span_to_label, sentence, span_to_on_score)


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")
        self.layer_params = []
        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            layer_params = self.model.add_subcollection("Layer" + str(len(self.weights)))
            self.weights.append(layer_params.add_parameters((next_dim, prev_dim)))
            self.biases.append(layer_params.add_parameters(next_dim))
            self.layer_params.append(layer_params)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class Parser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            _,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            __,
            dropout,
            use_elmo=True,
            predict_pos=True
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")
        self.use_elmo = use_elmo
        self.model = model.add_subcollection("Parser")
        self.mlp = self.model.add_subcollection("mlp")

        self.tag_vocab = tag_vocab
        print('tag vocab', tag_vocab.size)
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim
        self.hidden_dim = label_hidden_dim

        lstm_input_dim = word_embedding_dim
        if self.use_elmo:
            self.elmo_weights = self.model.parameters_from_numpy(
                np.array([0.19608361, 0.53294581, -0.00724584]), name='elmo-averaging-weights')
            lstm_input_dim += 1024

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            lstm_input_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_encoding = Feedforward(
            self.mlp, 2 * lstm_dim, [], label_hidden_dim)

        self.f_label = Feedforward(
            self.mlp, label_hidden_dim, [], label_vocab.size)

        self.f_tag = Feedforward(
            self.mlp, label_hidden_dim, [], tag_vocab.size)

        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.dropout = dropout
        self.empty_label = ()
        self.empty_label_index = self.label_vocab.index(self.empty_label)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def _featurize_sentence(self, sentence, is_train, elmo_embeddings):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        cur_word_index = 0
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                unk_word = (np.random.rand() < 1 / (1 + count)) or (np.random.rand() < 0.1)
                if not count or (is_train and unk_word):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            input_components = [word_embedding]
            if self.use_elmo:
                if tag == START or tag == STOP:
                    elmo_embedding = dy.zeros(1024)
                else:
                    elmo_weights = dy.parameter(self.elmo_weights)
                    elmo_embedding = dy.sum_dim(dy.cmult(elmo_weights, dy.pick(elmo_embeddings,
                                                                               index=cur_word_index,
                                                                               dim=1)), [0])
                    cur_word_index += 1
                input_components.append(elmo_embedding)

            embedding = dy.concatenate(input_components)
            if is_train:
                embedding = dy.dropout(embedding, p=0.4)
            embeddings.append(embedding)
        return self.lstm.transduce(embeddings)

    def _get_span_encoding(self, left, right, lstm_outputs):
        forward = (
            lstm_outputs[right][:self.lstm_dim] -
            lstm_outputs[left][:self.lstm_dim])
        backward = (
            lstm_outputs[left + 1][self.lstm_dim:] -
            lstm_outputs[right + 1][self.lstm_dim:])
        return dy.concatenate([forward, backward])

    def parse(self, sentence, elmo_embeddings):
        if isinstance(sentence[0], str):
            sentence = [(None, word) for word in sentence]
        label_log_probabilities, tag_log_probabilities, span_to_index = \
            self._get_scores(sentence, is_train=False, elmo_embeddings=elmo_embeddings)
        label_log_probabilities_np = label_log_probabilities.npvalue()
        tag_log_probabilities_np = tag_log_probabilities.npvalue()
        sentence_with_tags = []
        num_correct = 0
        total = 0
        for word_index, (oracle_tag, word) in enumerate(sentence):
            tag_index = np.argmax(tag_log_probabilities_np[:, word_index])
            tag = self.tag_vocab.value(tag_index)
            oracle_tag_is_deletable = oracle_tag in DELETABLE_TAGS
            predicted_tag_is_deletable = tag in DELETABLE_TAGS
            if oracle_tag is not None and oracle_tag not in self.tag_vocab.indices:
                print(oracle_tag, 'not in tag vocab')
                oracle_tag = None
            if oracle_tag is not None:
                oracle_tag_index = self.tag_vocab.index(oracle_tag)
                if oracle_tag_index == tag_index and tag != oracle_tag:
                    if oracle_tag[0] != '-':
                        print(tag, oracle_tag)
                    tag = oracle_tag
                num_correct += tag_index == oracle_tag_index

            if oracle_tag is not None and oracle_tag_is_deletable != predicted_tag_is_deletable:
                sentence_with_tags.append((oracle_tag, word))
            else:
                sentence_with_tags.append((tag, word))
            total += 1
        assert (0, len(sentence)) in span_to_index, span_to_index
        tree = optimal_parser(label_log_probabilities_np,
                              span_to_index,
                              sentence_with_tags,
                              self.empty_label_index,
                              self.label_vocab)
        return tree

    def _get_scores(self, sentence, is_train, elmo_embeddings):
        lstm_outputs = self._featurize_sentence(sentence, is_train=is_train,
                                                elmo_embeddings=elmo_embeddings)

        other_encodings = []
        single_word_encodings = []
        temporary_span_to_index = {}
        for left in range(len(sentence)):
            for right in range(left + 1, len(sentence) + 1):
                encoding = self._get_span_encoding(left, right, lstm_outputs)
                span = (left, right)
                if right - left == 1:
                    temporary_span_to_index[span] = len(single_word_encodings)
                    single_word_encodings.append(encoding)
                else:
                    temporary_span_to_index[span] = len(other_encodings)
                    other_encodings.append(encoding)

        encodings = single_word_encodings + other_encodings
        span_to_index = {}
        for span, index in temporary_span_to_index.items():
            if span[1] - span[0] == 1:
                new_index = index
            else:
                new_index = index + len(single_word_encodings)
            span_to_index[span] = new_index
        span_encodings = dy.rectify(dy.reshape(self.f_encoding(dy.concatenate_to_batch(encodings)),
                                               (self.hidden_dim, len(encodings))))
        label_scores = self.f_label(span_encodings)
        label_scores_reshaped = dy.reshape(label_scores, (self.label_vocab.size, len(encodings)))
        label_log_probabilities = dy.log_softmax(label_scores_reshaped)
        single_word_span_encodings = dy.select_cols(span_encodings,
                                                    list(range(len(single_word_encodings))))
        tag_scores = self.f_tag(single_word_span_encodings)
        tag_scores_reshaped = dy.reshape(tag_scores,
                                         (self.tag_vocab.size, len(single_word_encodings)))
        tag_log_probabilities = dy.log_softmax(tag_scores_reshaped)
        return label_log_probabilities, tag_log_probabilities, span_to_index

    def span_parser(self, sentence, is_train, elmo_embeddings, gold=None):
        if is_train:
            label_log_probabilities, tag_log_probabilities, span_to_index = self._get_scores(
                sentence, is_train, elmo_embeddings)
            assert isinstance(gold, ParseNode)
            total_loss = dy.zeros(1)
            span_to_gold_label = get_all_spans(gold)
            for span, oracle_label in span_to_gold_label.items():
                oracle_label_index = self.label_vocab.index(oracle_label)
                index = span_to_index[span]
                if span[1] - span[0] == 1:
                    oracle_tag = sentence[span[0]][0]
                    total_loss -= tag_log_probabilities[self.tag_vocab.index(oracle_tag)][index]
                total_loss -= label_log_probabilities[oracle_label_index][index]
            return total_loss
        else:
            return self.parse(sentence, elmo_embeddings)
