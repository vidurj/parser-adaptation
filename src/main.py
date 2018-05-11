import argparse
import itertools
import math
import os.path
import pickle
import random
import time
from collections import defaultdict
from collections import namedtuple
import dynet as dy
import h5py
import numpy as np
import spacy
import evaluate
import parse
import trees
import vocabulary
from trees import InternalParseNode, LeafParseNode, ParseNode

label_nt = namedtuple("label", ["left", "right", "oracle_label_index"])


def check_overlap(span_a, span_b):
    return span_a[0] < span_b[0] < span_a[1] < span_b[1] or \
           span_b[0] < span_a[0] < span_b[1] < span_a[1]


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def get_all_spans(parse):
    """
    Creates a map from every span in a sentence to its label implied the the given constituency
    parse tree.
    """
    assert isinstance(parse, ParseNode)
    sentence = list(parse.leaves)
    parses = [parse]
    span_to_gold_label = {}
    while len(parses) > 0:
        tree = parses.pop()
        if isinstance(tree, LeafParseNode):
            continue
        else:
            assert isinstance(tree, InternalParseNode)
        parses.extend(tree.children)
        span_to_gold_label[(tree.left, tree.right)] = tree.label
    for start in range(0, len(sentence) + 1):
        for end in range(start + 1, len(sentence) + 1):
            span = (start, end)
            if span not in span_to_gold_label:
                span_to_gold_label[span] = ()
    return span_to_gold_label


def load_or_create_model(args, parses_for_vocab):
    components = args.model_path_base.split('/')
    directory = '/'.join(components[:-1])
    if os.path.isdir(directory):
        relevant_files = [f for f in os.listdir(directory) if f.startswith(components[-1])]
    else:
        relevant_files = []
    assert len(relevant_files) <= 2, "Multiple possibilities {}".format(relevant_files)
    if len(relevant_files) > 0:
        print("Loading model from {}...".format(args.model_path_base))

        model = dy.ParameterCollection()
        [parser] = dy.load(args.model_path_base, model)
    else:
        assert parses_for_vocab is not None
        print("Constructing vocabularies using train parses...")

        tag_vocab = vocabulary.Vocabulary()
        tag_vocab.index(parse.START)
        tag_vocab.index(parse.STOP)

        word_vocab = vocabulary.Vocabulary()
        word_vocab.index(parse.START)
        word_vocab.index(parse.STOP)
        word_vocab.index(parse.UNK)

        label_vocab = vocabulary.Vocabulary()
        label_vocab.index(())

        for tree in parses_for_vocab:
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalParseNode):
                    label_vocab.index(node.label)
                    nodes.extend(reversed(node.children))
                else:
                    assert isinstance(node, LeafParseNode)
                    tag_vocab.index(node.tag)
                    word_vocab.index(node.word)

        tag_vocab.freeze()
        word_vocab.freeze()
        label_vocab.freeze()

        print("Initializing model...")
        model = dy.ParameterCollection()
        parser = parse.Parser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            None,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            None,
            args.dropout,
            not args.no_elmo
        )
    return parser, model


def load_parses(file_path):
    print("Loading trees from {}...".format(file_path))
    treebank = trees.load_trees(file_path)
    parses = [tree.convert() for tree in treebank]
    return parses


def evaluate_on_brown_corpus(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    assert parser.use_elmo == args.use_elmo, (parser.use_elmo, args.use_elmo)

    directories = ['cf', 'cg', 'ck', 'cl', 'cm', 'cn', 'cp', 'cr']
    for directory in directories:
        print('-' * 100)
        print(directory)
        input_file = '../brown/' + directory + '/' + directory + '.all.mrg'
        expt_name = args.expt_name + '/' + directory
        if not os.path.exists(expt_name):
            os.mkdir(expt_name)
        cleaned_corpus_path = trees.cleanup_text(input_file)
        treebank = trees.load_trees(cleaned_corpus_path, strip_top=True, filter_none=True)
        sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves] for tree in treebank]
        tokenized_lines = [' '.join([word for pos, word in sentence]) for sentence in sentences]
        if args.use_elmo:
            embedding_file = compute_elmo_embeddings(tokenized_lines, expt_name,
                                                     args.path_to_python)
        else:
            embedding_file = None
        dev_predicted = []
        num_correct = 0
        total = 0
        for tree_index, tree in enumerate(treebank):
            if tree_index % 100 == 0:
                print(tree_index)
                dy.renew_cg()
            sentence = sentences[tree_index]
            if args.use_elmo:
                embeddings_np = embedding_file[str(tree_index)][:, :, :]
                assert embeddings_np.shape[1] == len(sentence), (
                embeddings_np.shape[1], len(sentence))
                embeddings = dy.inputTensor(embeddings_np)
            else:
                embeddings = None
            predicted, (additional_info, c, t) = parser.span_parser(sentence, is_train=False,
                                                                    elmo_embeddings=embeddings)
            num_correct += c
            total += t
            dev_predicted.append(predicted.convert())

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=True,
                                                   name="without-labels",
                                                   expt_name=expt_name)
        print("dev-fscore without labels", dev_fscore_without_labels)

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=True,
                                                   flatten=True,
                                                   name="without-label-flattened",
                                                   expt_name=expt_name)
        print("dev-fscore without labels and flattened", dev_fscore_without_labels)

        dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, dev_predicted,
                                                   args=args,
                                                   erase_labels=False,
                                                   flatten=True,
                                                   name="flattened",
                                                   expt_name=expt_name)
        print("dev-fscore with labels and flattened", dev_fscore_without_labels)

        test_fscore = evaluate.evalb('EVALB/', treebank, dev_predicted, args=args,
                                     name="regular",
                                     expt_name=expt_name)

        print("regular", test_fscore)
        pos_fraction = num_correct / total
        print('pos fraction', pos_fraction)
        with open(expt_name + '/pos_accuracy.txt', 'w') as f:
            f.write(str(pos_fraction))


def run_test_qbank(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    all_trees = trees.load_trees(args.question_bank_trees_path)

    if args.stanford_split == 'true':
        print('using stanford split')
        split_to_indices = {
            'train': list(range(0, 1000)) + list(range(2000, 3000)),
            'dev': list(range(1000, 1500)) + list(range(3000, 3500)),
            'test': list(range(1500, 2000)) + list(range(3500, 4000))
        }
    else:
        print('not using stanford split')
        split_to_indices = {
            'train': range(0, 2000),
            'dev': range(2000, 3000),
            'test': range(3000, 4000)
        }

    test_indices = split_to_indices[args.split]
    qb_embeddings_file = h5py.File('../question-bank.hdf5', 'r')
    dev_predicted = []
    for test_index in test_indices:
        if len(dev_predicted) % 100 == 0:
            dy.renew_cg()
        tree = all_trees[test_index]
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        test_embeddings_np = qb_embeddings_file[str(test_index)][:, :, :]
        assert test_embeddings_np.shape[1] == len(sentence)
        test_embeddings = dy.inputTensor(test_embeddings_np)
        predicted, _ = parser.span_parser(sentence, is_train=False,
                                          elmo_embeddings=test_embeddings)
        dev_predicted.append(predicted.convert())

    test_treebank = [all_trees[index] for index in test_indices]
    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=True,
                                               name="without-labels")
    print("dev-fscore without labels", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=True,
                                               flatten=True,
                                               name="without-label-flattened")
    print("dev-fscore without labels and flattened", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted,
                                               args=args,
                                               erase_labels=False,
                                               flatten=True,
                                               name="flattened")
    print("dev-fscore with labels and flattened", dev_fscore_without_labels)

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, dev_predicted, args=args,
                                 name="regular")

    print("regular", test_fscore)


def compute_elmo_embeddings(tokenized_lines, expt_name, path_to_python):
    if not os.path.exists(expt_name):
        os.mkdir(expt_name)
    elmo_embeddings_file_path = expt_name + '/elmo.hdf5'
    print(elmo_embeddings_file_path)
    if not os.path.exists(elmo_embeddings_file_path):
        tokenized_sentences_file_path = expt_name + '/tokenized_sentences.txt'

        normalized_lines = []
        for line in tokenized_lines:
            cleaned_line = ''
            for word in line.split():
                if word == '-LRB-' or word == 'LRB':
                    word = '('
                elif word == '-RRB-' or word == 'RRB':
                    word = ')'
                elif word == '-LCB-' or word == 'LCB':
                    word = '{'
                elif word == '-RCB-' or word == 'RCB':
                    word = '}'
                elif word == '`':
                    word = "'"
                elif word == "''":
                    word = '"'

                if '\\' in word:
                    print('-' * 100)
                    print(word)
                    print('-' * 100)
                cleaned_line += word + ' '
            normalized_lines.append(cleaned_line.strip())
        with open(tokenized_sentences_file_path, 'w') as f:
            f.write('\n'.join(normalized_lines))

        generate_elmo_vectors = '{} -m allennlp.run elmo {} {} --all'.format(
            path_to_python,
            tokenized_sentences_file_path,
            elmo_embeddings_file_path
        )
        print(generate_elmo_vectors)
        return_code = os.system(generate_elmo_vectors)
        assert return_code == 0, return_code
    else:
        print('Using precomputed embeddings at', elmo_embeddings_file_path)
    embedding_file = h5py.File(elmo_embeddings_file_path, 'r')
    return embedding_file


def test_on_brackets(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    with open(args.input_file, 'r') as f:
        lines = f.read().splitlines()

    nlp = spacy.load('en')

    processed_lines = []
    sentence_number_to_bracketings = defaultdict(set)
    original_brackets = set()
    for sentence_number, line in enumerate(lines):
        tokens = [token.text for token in nlp(line)]
        open_bracket_indices = []
        sentence = []
        print(line)
        print('***')
        constituent_spans = set()
        for token in tokens:
            if token == '[' or token == '{':
                open_bracket_indices.append(len(sentence))
            elif token == ']' or token == '}':
                is_constituent = token == ']'
                span = (open_bracket_indices.pop(), len(sentence))
                temp = sentence + ["*" for _ in range(5)]
                print(temp[span[0]: span[1]], is_constituent)
                sentence_number_to_bracketings[sentence_number].add((span, is_constituent))
                original_brackets.add((sentence_number, span))
                if is_constituent:
                    constituent_spans.add(span)
            else:
                sentence.append(token)

        for start in range(len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span = (start, end)
                for constituent_span in constituent_spans:
                    if check_overlap(span, constituent_span):
                        assert span not in constituent_spans
                        sentence_number_to_bracketings[sentence_number].add((span, False))
        print('-' * 40)
        assert len(open_bracket_indices) == 0, open_bracket_indices
        processed_lines.append(' '.join(sentence))

    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    embedding_file = compute_elmo_embeddings(processed_lines,
                                             args.expt_name + '/test',
                                             args.path_to_python)
    sentences = [[(None, word) for word in line.split()] for line in processed_lines]
    num_correct = 0
    num_correct_original = 0
    num_wrong = 0
    num_perfect_sentences = 0
    num_sentences = 0
    output_string = ''
    for sentence_number, sentence in enumerate(sentences):
        elmo_embeddings_np = embedding_file[str(sentence_number)][:, :, :]
        assert elmo_embeddings_np.shape[1] == len(sentence)
        elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
        lstm_outputs = parser._featurize_sentence(sentence, is_train=False,
                                                  elmo_embeddings=elmo_embeddings)
        encodings = []
        span_to_index = {}
        for ((start, end), is_constituent) in sentence_number_to_bracketings[sentence_number]:
            span_to_index[(start, end)] = len(encodings)
            encodings.append(parser._get_span_encoding(start, end, lstm_outputs))

        if len(encodings) == 0:
            print('skipping sentence', sentence_number)
            continue

        label_scores = parser.f_label(
            dy.rectify(parser.f_encoding(dy.concatenate_to_batch(encodings))))
        label_scores_reshaped = dy.reshape(label_scores,
                                           (parser.label_vocab.size, len(encodings)))
        label_probabilities = dy.softmax(label_scores_reshaped)

        is_perfect = True
        sentence_words = [word for pos, word in sentence]
        for (span, is_constituent) in sentence_number_to_bracketings[sentence_number]:
            span_index = span_to_index[span]
            non_constituent_prob_np = label_probabilities[parser.empty_label_index][
                span_index].scalar_value()
            if is_constituent and non_constituent_prob_np < 0.5:
                num_correct += 1
                if (sentence_number, span) in original_brackets:
                    num_correct_original += 1
            elif not is_constituent and non_constituent_prob_np > 0.5:
                num_correct += 1
                if (sentence_number, span) in original_brackets:
                    num_correct_original += 1
            else:
                print(is_constituent, non_constituent_prob_np,
                      ' '.join(sentence_words[span[0]: span[1]]))
                is_perfect = False
                num_wrong += 1

        parse, _ = parser.span_parser(sentence, is_train=False, elmo_embeddings=elmo_embeddings)
        parse_string = parse.convert().linearize()
        output_string += parse_string + '\n'

        if not is_perfect:
            print(sentence_number)
        num_perfect_sentences += is_perfect
        num_sentences += 1

    print('fraction of original correct spans', num_correct_original, len(original_brackets),
          num_correct_original / float(len(original_brackets)))
    print('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)
    print(
    'num sentences', num_sentences, 'num perfect sentences', num_perfect_sentences, 'fraction',
    num_perfect_sentences / float(num_sentences))
    results = str(args) + '\n' + str(
        ('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)) + \
              str(('num sentences', num_sentences, 'num perfect sentences', num_perfect_sentences,
                   'fraction', num_perfect_sentences / float(num_sentences)))
    with open(args.expt_name + '/trained_test_results.txt', 'w') as f:
        f.write(results)
    with open(args.expt_name + '/parses.txt', 'w') as f:
        f.write(output_string)


def train_on_brackets(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    with open(args.input_file, 'r') as f:
        lines = f.read().splitlines()

    nlp = spacy.load('en')

    processed_lines = []
    sentence_number_to_bracketings = defaultdict(set)
    for sentence_number, line in enumerate(lines):
        line = ' '.join(line.split())
        tokens = [token.text for token in nlp(line)]
        open_bracket_indices = []
        sentence = []
        print(line)
        print('***')
        constituent_spans = set()
        for token in tokens:
            if token == '[' or token == '{':
                open_bracket_indices.append(len(sentence))
            elif token == ']' or token == '}':
                is_constituent = token == ']'
                span = (open_bracket_indices.pop(), len(sentence))
                sentence_number_to_bracketings[sentence_number].add((span, is_constituent))
                if is_constituent:
                    constituent_spans.add(span)
            else:
                sentence.append(token)

        for start in range(len(sentence)):
            for end in range(start + 1, len(sentence) + 1):
                span = (start, end)
                for constituent_span in constituent_spans:
                    if check_overlap(span, constituent_span):
                        assert span not in constituent_spans
                        sentence_number_to_bracketings[sentence_number].add((span, False))
        assert len(open_bracket_indices) == 0, open_bracket_indices
        processed_lines.append(' '.join(sentence))
        stripped_tokens = [x for x in tokens if x != '[' and x != ']' and x != '{' and x != '}']
        assert processed_lines[-1].split() == stripped_tokens, (
        processed_lines[-1].split(), stripped_tokens)

    wsj_train = load_parses(args.wsj_train_trees_path)
    parser, model = load_or_create_model(args, wsj_train)
    trainer = dy.AdamTrainer(model)
    embedding_file = compute_elmo_embeddings(processed_lines, args.expt_name)

    wsj_embedding_file = h5py.File(args.wsj_train_elmo_embeddings_path, 'r')
    sentences = [[(None, word) for word in line.split()] for line in processed_lines]
    wsj_indices = []
    for epoch in itertools.count(start=1):
        dy.renew_cg()

        loss = dy.zeros(1)
        num_correct = 0
        num_wrong = 0

        for sentence_index in range(len(sentences)):
            sentence = sentences[sentence_index]
            elmo_embeddings_np = embedding_file[str(sentence_index)][:, :, :]
            assert elmo_embeddings_np.shape[1] == len(sentence)
            elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
            lstm_outputs = parser._featurize_sentence(sentence, is_train=True,
                                                      elmo_embeddings=elmo_embeddings)
            encodings = []
            span_to_index = {}
            for ((start, end), is_constituent) in sentence_number_to_bracketings[sentence_index]:
                span_to_index[(start, end)] = len(encodings)
                encodings.append(parser._get_span_encoding(start, end, lstm_outputs))

            if len(encodings) == 0:
                print('skipping sentence')
                continue

            label_scores = parser.f_label(
                dy.rectify(parser.f_encoding(dy.concatenate_to_batch(encodings))))
            label_scores_reshaped = dy.reshape(label_scores,
                                               (parser.label_vocab.size, len(encodings)))
            label_probabilities = dy.softmax(label_scores_reshaped)

            for (span, is_constituent) in sentence_number_to_bracketings[sentence_index]:
                span_index = span_to_index[span]
                non_constituent_prob_np = label_probabilities[parser.empty_label_index][
                    span_index].scalar_value()
                if is_constituent:
                    loss -= dy.log(
                        1 - label_probabilities[parser.empty_label_index][span_index] + 10 ** -7)
                else:
                    loss -= dy.log(
                        label_probabilities[parser.empty_label_index][span_index] + 10 ** -7)
                if is_constituent and non_constituent_prob_np < 0.5:
                    num_correct += 1
                elif not is_constituent and non_constituent_prob_np > 0.5:
                    num_correct += 1
                else:
                    num_wrong += 1

        print(loss.scalar_value())
        print('batch number', epoch)
        print('accuracy', num_correct / float(num_correct + num_wrong), num_correct, num_wrong)

        batch_losses = [loss]

        for _ in range(50):
            if not wsj_indices:
                wsj_indices = list(range(39832))
                random.shuffle(wsj_indices)

            index = wsj_indices.pop()
            tree = wsj_train[index]
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
            wsj_embeddings_np = wsj_embedding_file[str(index)][:, :, :]
            assert wsj_embeddings_np.shape[1] == len(sentence)
            wsj_embeddings = dy.inputTensor(wsj_embeddings_np)
            loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                      elmo_embeddings=wsj_embeddings)
            batch_losses.append(loss)

        batch_loss = dy.average(batch_losses)
        batch_loss_value = batch_loss.scalar_value()
        batch_loss.backward()
        trainer.update()

        print(batch_loss_value)
        print('-' * 100)

        if epoch % 10 == 0:
            save_latest_model(args.model_path_base, parser)


def save_latest_model(model_path_base, parser):
    latest_model_path = "{}_latest_model".format(model_path_base)
    for ext in [".data", ".meta"]:
        path = latest_model_path + ext
        if os.path.exists(path):
            print("Removing previous model file {}...".format(path))
            os.remove(path)

    print("Saving new model to {}...".format(latest_model_path))
    dy.save(latest_model_path, [parser])


def parse_trees_to_string_lines(trees):
    tokenized_lines = []
    for tree in trees:
        sentence = [leaf.word for leaf in tree.leaves]
        tokenized_lines.append(' '.join(sentence))
    return tokenized_lines


def train_on_parses(args):
    args.model_path_base = os.path.join(args.expt_name, 'model')
    train_parses = load_parses(os.path.join(args.expt_name, 'train_trees.txt'))
    train_indices = list(range(len(train_parses)))
    if not args.no_elmo:
        train_tokenized_lines = parse_trees_to_string_lines(train_parses)
        train_embeddings = compute_elmo_embeddings(train_tokenized_lines,
                                                   os.path.join(args.expt_name,
                                                                'train_embeddings'),
                                                   args.path_to_python)
    else:
        train_embeddings = None

    dev_trees = trees.load_trees(os.path.join(args.expt_name, 'dev_trees.txt'))

    if not args.no_elmo:
        dev_tokenized_lines = parse_trees_to_string_lines(dev_trees)
        dev_embeddings = compute_elmo_embeddings(dev_tokenized_lines,
                                                 os.path.join(args.expt_name, 'dev_embeddings'),
                                                 args.path_to_python)
    else:
        dev_embeddings = None

    additional_trees_path = os.path.join(args.expt_name, 'additional_trees.txt')
    if os.path.exists(additional_trees_path):
        print('Training on', additional_trees_path)
        additional_train_trees = load_parses(additional_trees_path)
        additional_trees_indices = list(range(len(additional_train_trees)))

        if not args.no_elmo:
            additional_tokenized_lines = parse_trees_to_string_lines(additional_train_trees)
            additional_embeddings_file = compute_elmo_embeddings(additional_tokenized_lines,
                                                                 os.path.join(args.expt_name,
                                                                              'additional_embeddings'),
                                                                 args.path_to_python)
    else:
        print('No additional training trees.')
        additional_train_trees = []
        additional_trees_indices = []

    parser, model = load_or_create_model(args, train_parses + additional_train_trees)
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    for epoch in itertools.count(start=1):
        np.random.shuffle(train_indices)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_indices), args.batch_size):
            dy.renew_cg()

            batch_losses = []
            for tree_index in train_indices[start_index: start_index + args.batch_size]:
                tree = train_parses[tree_index]
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                if train_embeddings is not None:
                    embeddings_np = train_embeddings[str(tree_index)][:, :, :]
                    assert embeddings_np.shape[1] == len(sentence), (
                    embeddings_np.shape, len(sentence), sentence)
                    embeddings = dy.inputTensor(embeddings_np)
                else:
                    embeddings = None
                loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                          elmo_embeddings=embeddings)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            random.shuffle(additional_trees_indices)
            for tree_index in additional_trees_indices[:100]:
                tree = additional_train_trees[tree_index]
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                if additional_embeddings_file is not None:
                    embeddings = dy.inputTensor(
                        additional_embeddings_file[str(tree_index)][:, :, :])
                else:
                    embeddings = None
                loss = parser.span_parser(sentence,
                                          is_train=True,
                                          gold=tree,
                                          elmo_embeddings=embeddings)
                batch_losses.append(loss)

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size,
                    int(np.ceil(len(train_indices) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

        if epoch % int(args.num_epochs_per_check) == 0:
            check_performance_and_save(parser,
                                       best_dev_fscore,
                                       best_dev_model_path,
                                       dev_trees,
                                       dev_embeddings,
                                       args)


def run_train_question_bank(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    all_trees = trees.load_trees(args.question_bank_trees_path)
    all_parses = [tree.convert() for tree in all_trees]

    print("Loaded {:,} trees.".format(len(all_parses)))

    tentative_stanford_train_indices = list(range(0, 1000)) + list(range(2000, 3000))
    stanford_dev_indices = list(range(1000, 1500)) + list(range(3000, 3500))
    stanford_test_indices = list(range(1500, 2000)) + list(range(3500, 4000))

    dev_and_test_sentences = set()
    for index in stanford_dev_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        dev_and_test_sentences.add(tuple(sentence))

    for index in stanford_test_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        dev_and_test_sentences.add(tuple(sentence))

    stanford_train_indices = []
    for index in tentative_stanford_train_indices:
        parse = all_parses[index]
        sentence = [(leaf.tag, leaf.word) for leaf in parse.leaves]
        if tuple(sentence) not in dev_and_test_sentences:
            stanford_train_indices.append(index)

    qb_embeddings_file = h5py.File(args.question_bank_elmo_embeddings_path, 'r')

    print("We have {:,} train trees.".format(len(stanford_train_indices)))

    wsj_train = load_parses(args.wsj_train_trees_path)
    qb_train_parses = [all_parses[index] for index in stanford_train_indices]
    qb_dev_treebank = [all_trees[index] for index in stanford_dev_indices]
    parser, model = load_or_create_model(args, qb_train_parses + wsj_train)

    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    if args.train_on_wsj == 'true':
        print('training on wsj')
        wsj_embeddings_file = h5py.File(args.wsj_train_elmo_embeddings_path, 'r')
        wsj_indices = list(range(39832))
    else:
        print('not training on wsj')

    indices_file_path = args.expt_name + '/train_tree_indices.txt'
    if os.path.exists(indices_file_path):
        with open(indices_file_path, 'r') as f:
            tree_indices = [int(x) for x in f.read().splitlines()]
        print('loaded', len(tree_indices), 'indices from file', indices_file_path)
    elif args.num_samples != 'false':
        print('restricting to', args.num_samples, 'samples')
        random.shuffle(stanford_train_indices)
        tree_indices = stanford_train_indices[:int(args.num_samples)]
    else:
        print('training on original data')
        tree_indices = stanford_train_indices

    if args.num_samples != 'false':
        assert int(args.num_samples) == len(tree_indices), (args.num_samples, len(tree_indices))

    with open(indices_file_path, 'w') as f:
        f.write('\n'.join([str(x) for x in tree_indices]))

    for epoch in itertools.count(start=1):
        np.random.shuffle(tree_indices)
        epoch_start_time = time.time()

        for start_index in range(0, len(tree_indices), args.batch_size):
            dy.renew_cg()

            batch_losses = []
            for tree_index in tree_indices[start_index: start_index + args.batch_size]:
                tree = all_parses[tree_index]
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                embeddings_np = qb_embeddings_file[str(tree_index)][:, :, :]
                assert embeddings_np.shape[1] == len(sentence), (
                embeddings_np.shape, len(sentence), sentence)
                embeddings = dy.inputTensor(embeddings_np)
                loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                          elmo_embeddings=embeddings)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            if args.train_on_wsj == 'true':
                random.shuffle(wsj_indices)
                for tree_index in wsj_indices[:100]:
                    tree = wsj_train[tree_index]
                    sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
                    embeddings = dy.inputTensor(wsj_embeddings_file[str(tree_index)][:, :, :])
                    loss = parser.span_parser(sentence, is_train=True, gold=tree,
                                              elmo_embeddings=embeddings)
                    batch_losses.append(loss)

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size,
                    int(np.ceil(len(tree_indices) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

        if epoch % 10 == 0:
            check_performance_and_save(parser,
                                       best_dev_fscore,
                                       best_dev_model_path,
                                       qb_dev_treebank,
                                       qb_embeddings_file,
                                       args)


def check_performance(parser, treebank, sentence_embeddings, args):
    dev_start_time = time.time()

    dev_predicted = []
    for tree_index, tree in enumerate(treebank):
        if tree_index % 100 == 0:
            dy.renew_cg()
        if sentence_embeddings is not None:
            elmo_embeddings = dy.inputTensor(sentence_embeddings[str(tree_index)][:, :, :])
        else:
            elmo_embeddings = None
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        predicted = parser.span_parser(sentence,
                                       is_train=False,
                                       elmo_embeddings=elmo_embeddings)
        dev_predicted.append(predicted.convert())

    dev_fscore = evaluate.evalb('EVALB', treebank, dev_predicted, args=args, name="dev")
    return dev_fscore, dev_start_time


def check_performance_and_save(parser,
                               best_dev_fscore,
                               best_dev_model_path,
                               treebank,
                               sentence_embeddings,
                               args):
    dev_fscore, dev_start_time = check_performance(parser,
                                                   treebank,
                                                   sentence_embeddings,
                                                   args)

    print('dev-fscore {}\ndev-elapsed {}'.format(dev_fscore, format_elapsed(dev_start_time)))

    if dev_fscore.fscore > best_dev_fscore:
        if best_dev_model_path is not None:
            for ext in [".data", ".meta"]:
                path = best_dev_model_path + ext
                if os.path.exists(path):
                    print("Removing previous model file {}...".format(path))
                    os.remove(path)

        best_dev_fscore = dev_fscore.fscore
        best_dev_model_path = "{}_dev={:.2f}".format(args.model_path_base, dev_fscore.fscore)
        print("Saving new best model to {}...".format(best_dev_model_path))
        dy.save(best_dev_model_path, [parser])
    return best_dev_fscore, best_dev_model_path


# TODO support no elmo case
def run_test(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)
    print("Loading test trees from {}...".format(args.trees_path))
    sentence_embeddings = h5py.File(args.elmo_embeddings_path, 'r')
    test_treebank = trees.load_trees(args.trees_path)

    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")
    check_performance(parser, test_treebank, range(len(test_treebank)), sentence_embeddings, args)


def produce_parse_lists(args):
    if args.dev_parses:
        print('using dev')
        indices = list(range(1000, 1500)) + list(range(3000, 3500))
    else:
        with open(args.index_file_path, 'r') as f:
            indices = [int(x) for x in f.read().splitlines()]
    with open(args.parses_file, 'r') as f:
        lines = f.read().splitlines()
    relevant_lines = [lines[x] for x in indices]
    with open('qb_subset_' + str(len(relevant_lines)) + '.txt', 'w') as f:
        f.write('\n'.join(relevant_lines) + '\n')


def produce_sentences_from_conll(args):
    sentences = ['']
    with open(args.input_file, 'r') as f:
        lines = f.read().strip().splitlines()
    for line in lines:
        if line == '':
            sentences.append('')
        else:
            tokens = line.split()
            assert len(tokens) == 10, line
            sentences[-1] += ' ' + tokens[1]
    with open(args.output_file, 'w') as f:
        f.write('\n'.join([x.strip() for x in sentences]))


def produce_elmo_for_treebank(args):
    treebank = trees.load_trees(args.input_file, strip_top=True, filter_none=True)
    for tree in treebank:
        parse = tree.convert()
        sentence1 = [leaf.word for leaf in tree.leaves]
        sentence2 = [leaf.word for leaf in parse.leaves]
        assert sentence1 == sentence2, (sentence1, sentence2)
    sentences = [[leaf.word for leaf in tree.leaves] for tree in treebank]
    tokenized_lines = [' '.join(sentence) for sentence in sentences]
    compute_elmo_embeddings(tokenized_lines, args.expt_name, args.path_to_python)


def test_on_parses(args):
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    treebank = trees.load_trees(args.input_file, strip_top=True, filter_none=True)
    output = [tree.linearize() for tree in treebank]
    with open(args.expt_name + '/parses.txt', 'w') as f:
        f.write('\n'.join(output))
    sentence_embeddings = h5py.File(args.elmo_embeddings_file_path, 'r')

    test_predicted = []
    start_time = time.time()
    total_log_likelihood = 0
    total_confusion_matrix = {}
    total_turned_off = 0
    ranks = []
    num_correct = 0
    total = 0
    for tree_index, tree in enumerate(treebank):
        if tree_index % 100 == 0:
            print(tree_index)
            dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves]
        elmo_embeddings_np = sentence_embeddings[str(tree_index)][:, :, :]
        assert elmo_embeddings_np.shape[1] == len(sentence), (
        elmo_embeddings_np.shape[1], len(sentence), [word for pos, word in sentence])
        elmo_embeddings = dy.inputTensor(elmo_embeddings_np)
        predicted, (additional_info, c, t) = parser.span_parser(sentence, is_train=False,
                                                                elmo_embeddings=elmo_embeddings)
        num_correct += c
        total += t
        rank = additional_info[3]
        ranks.append(rank)
        total_log_likelihood += additional_info[-1]
        test_predicted.append(predicted.convert())
    print('pos accuracy', num_correct / total)
    print("total time", time.time() - start_time)
    print("total loglikelihood", total_log_likelihood)
    print("total turned off", total_turned_off)
    print(total_confusion_matrix)

    print(ranks)
    print("avg", np.mean(ranks), "median", np.median(ranks))

    dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, test_predicted,
                                               args=args,
                                               erase_labels=True,
                                               name="without-labels")
    print("dev-fscore without labels", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, test_predicted,
                                               args=args,
                                               erase_labels=True,
                                               flatten=True,
                                               name="without-label-flattened")
    print("dev-fscore without labels and flattened", dev_fscore_without_labels)

    dev_fscore_without_labels = evaluate.evalb('EVALB/', treebank, test_predicted,
                                               args=args,
                                               erase_labels=False,
                                               flatten=True,
                                               name="flattened")
    print("dev-fscore with labels and flattened", dev_fscore_without_labels)

    test_fscore = evaluate.evalb('EVALB/', treebank, test_predicted, args=args,
                                 name="regular")

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
    with open(os.path.join(args.expt_name, "confusion_matrix.pickle"), "wb") as f:
        pickle.dump(total_confusion_matrix, f)


def save_components(args):
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)
    parser.f_label.param_collection().save(args.save_path, '/f-label', append=False)
    parser.f_tag.param_collection().save(args.save_path, '/f-tag', append=True)
    parser.f_encoding.param_collection().save(args.save_path, '/f-encoding', append=True)
    parser.word_embeddings.save(args.save_path, '/word-embedding', append=True)
    parser.lstm.param_collection().save(args.save_path, '/lstm', append=True)


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train-on-parses")
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.set_defaults(callback=train_on_parses)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--no-elmo", action="store_true")
    subparser.add_argument("--path-to-python", default="python3")

    subparser.add_argument("--num-epochs-per-check", default=1)

    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--batch-size", type=int, default=10)

    subparser = subparsers.add_parser("test-on-trees")
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.set_defaults(callback=test_on_parses)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--elmo-embeddings-file-path", required=True)
    subparser.add_argument("--model-path-base", required=True)

    subparser = subparsers.add_parser("produce-elmo-for-treebank")
    subparser.set_defaults(callback=produce_elmo_for_treebank)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--path-to-python",
                           required=True,
                           help='Path to python environment with Allennlp. '
                                'Example: /home/user/miniconda3/envs/allennlp/bin/python3')

    subparser = subparsers.add_parser("produce-parse-lists")
    subparser.set_defaults(callback=produce_parse_lists)
    subparser.add_argument("--index-file-path", required=True)
    subparser.add_argument("--parses-file", required=True)
    subparser.add_argument("--dev-parses", action="store_true")

    subparser = subparsers.add_parser("sentences-from-conll")
    subparser.set_defaults(callback=produce_sentences_from_conll)
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--output-file", required=True)

    subparser = subparsers.add_parser("evaluate-brown-corpus")
    subparser.set_defaults(callback=evaluate_on_brown_corpus)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--use-elmo", action="store_true")
    subparser.add_argument("--input-file", required=True)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--path-to-python",
                           required=True,
                           help='Path to python environment with Allennlp. '
                                'Example: /home/user/miniconda3/envs/allennlp/bin/python3')

    subparser = subparsers.add_parser("train-on-brackets")
    subparser.set_defaults(callback=train_on_brackets)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--input-file",
                           required=True,
                           help='File with sentences and partial bracketings.')
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("test-on-brackets")
    subparser.set_defaults(callback=test_on_brackets)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--input-file",
                           required=True,
                           help='File with sentences and partial bracketings.')
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--expt-name", required=True)
    subparser.add_argument("--path-to-python",
                           required=True,
                           help='Path to python environment with Allennlp. '
                                'Example: /home/user/miniconda3/envs/allennlp/bin/python3')

    subparser = subparsers.add_parser("train-on-question-bank")
    subparser.set_defaults(callback=run_train_question_bank)
    for arg in dynet_args:
        subparser.add_argument(arg)

    subparser.add_argument("--train-on-wsj", required=True,
                           help='Whether or not to train on the WSJ corpus. '
                                'Must be either "true" or "false".')
    subparser.add_argument("--num-samples",
                           required=True,
                           help='Number of sentences to train on from Question Bank.')

    subparser.add_argument("--expt-name",
                           required=True,
                           help='The name of the experiment. '
                                'All results will be stored under this directory.')
    subparser.add_argument("--model-path-base",
                           required=True,
                           help='Path prefix with which to save the model.')
    subparser.add_argument("--question-bank-trees-path",
                           default='questionbank/all_qb_trees.txt')
    subparser.add_argument("--question-bank-elmo-embeddings-path",
                           default='../question-bank.hdf5')
    subparser.add_argument("--wsj-train-trees-path",
                           default='data/train.trees')
    subparser.add_argument("--wsj-train-elmo-embeddings-path",
                           default='../wsj-train.hdf5')
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--batch-size", type=int, default=10)

    subparser = subparsers.add_parser("test-on-question-bank")
    subparser.set_defaults(callback=run_test_qbank)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--stanford-split",
                           required=True,
                           help='Whether to use the Stanford train test split. '
                                'Must be "true" or "false".')
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--question-bank-trees-path",
                           default='questionbank/all_qb_trees.txt')
    subparser.add_argument("--question-bank-elmo-embeddings-path",
                           default='../question-bank.hdf5')
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--split",
                           required=True,
                           help='Which split to test on. Must be "train", "dev" or "test".')
    subparser.add_argument("--expt-name", required=True)


    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--trees-path", required=True)
    subparser.add_argument("--elmo-embeddings-path", required=True)
    subparser.add_argument("--expt-name", required=True)

    subparser = subparsers.add_parser("save-components")
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.set_defaults(callback=save_components)
    subparser.add_argument("--save-path", required=True)
    subparser.add_argument("--model-path-base", required=True)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
