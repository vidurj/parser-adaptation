import collections.abc

deletable_tags = {',', ':', '``', "''", '.'}


class TreebankNode(object):
    pass


class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str), label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)
        self.leaves = [leaf for child in self.children for leaf in child.leaves]

    def linearize(self, erase_labels=False):
        if erase_labels:
            label = "XX"
        else:
            label = self.label
        return "({} {})".format(
            label, " ".join(child.linearize(erase_labels) for child in self.children))

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]
        while len(tree.children) == 1 and isinstance(tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    def flatten(self):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.flatten())

        return InternalTreebankNode("*".join(sublabels), children)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word
        self.leaves = [self]

    def linearize(self, erase_labels=False):
        return "({} {})".format(self.tag, self.word)

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

    def flatten(self):
        return self


class ParseNode(object):
    pass


def create_internal_parse_node(label, children):
    if len(children) == 0:
        return None
    elif len(children) == 1 and isinstance(children[0], InternalParseNode):
        child = children[0]
        return create_internal_parse_node(label + child.label, child.children)
    else:
        return InternalParseNode(label, children)


class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple), label
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(left.right == right.left for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.leaves = [leaf for child in self.children for leaf in child.leaves]
        self._delete_punctuation = None
        self._tree_bank = None

    def clean_up_punctuation(self):
        new_children = []
        for child in self.children:
            if isinstance(child, LeafParseNode):
                new_children.append(child)
            else:
                child = child.clean_up_punctuation()
                child_children = child.children
                while len(child_children) > 0 and isinstance(child_children[0], LeafParseNode) and \
                                child_children[0].tag in deletable_tags:
                    new_children.append(child_children[0])
                    child_children = child_children[1:]

                more_children = []
                while len(child_children) > 0 and isinstance(child_children[-1], LeafParseNode) and \
                                child_children[-1].tag in deletable_tags:
                    more_children.append(child_children[-1])
                    child_children = child_children[:-1]

                if len(child_children) > 0:
                    new_child = create_internal_parse_node(child.label, child_children)
                    assert new_child is not None, (child.left, child.right, child.label)
                    new_children.append(new_child)
                new_children.extend(list(reversed(more_children)))
        new_node = create_internal_parse_node(self.label, new_children)
        assert new_node.left == self.left and new_node.right == self.right, (
            new_node.left, self.left, new_node.right, self.right)
        return new_node

    def delete_punctuation(self):
        if self._delete_punctuation is None:
            new_children = []
            for child in self.children:
                if isinstance(child, InternalParseNode):
                    new_child = child.delete_punctuation()
                    if new_child is not None:
                        new_children.append(new_child)
                else:
                    assert isinstance(child, LeafParseNode)
                    if child.tag not in deletable_tags:
                        new_children.append(child)
            self._delete_punctuation = create_internal_parse_node(self.label, new_children)
        return self._delete_punctuation

    def reset(self, left):
        self.left = left
        cur_left = left
        for child in self.children:
            child.reset(cur_left)
            if isinstance(child, LeafParseNode):
                cur_left += 1
            else:
                cur_left += len(child.leaves)
        self.right = cur_left

    def convert(self):
        if self._tree_bank is None:
            children = [child.convert() for child in self.children]
            tree = InternalTreebankNode(self.label[-1], children)
            for sublabel in reversed(self.label[:-1]):
                tree = InternalTreebankNode(sublabel, [tree])
            self._tree_bank = tree
        return self._tree_bank

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right, (self.left, left, right, self.right)
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
            ]


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int), index
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word
        self.leaves = [self]


    def reset(self, left):
        self.left = left
        self.rigth = left + 1

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def delete_punctuation(self):
        if self.tag in deletable_tags:
            return None
        else:
            return self


def cleanup_text(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    processed_lines = []
    num_open_parens = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '*' or line[0] == ';':
            continue
        else:
            if num_open_parens > 0:
                processed_lines[-1] += line + ' '
            else:
                processed_lines.append(line)

            num_parens_opened = line.count('(')
            num_parens_closed = line.count(')')
            num_open_parens += num_parens_opened - num_parens_closed
    processed_file_path = file_path[:-4] + '_cleaned.txt'
    output_string = ""
    for processed_line in processed_lines:
        output_string += processed_line.strip() + '\n'
    with open(processed_file_path, 'w') as f:
        f.write(output_string.strip())
    return processed_file_path


def load_trees(path, strip_top=True, filter_none=False):
    with open(path) as infile:
        lines = infile.read().splitlines()

    text = ''
    for line in lines:
        line = line.strip()
        if line.startswith('*') or line.startswith(';'):
            continue
        else:
            text += line + '\n'

    tokens = tuple(text.replace("(", " ( ").replace(")", " ) ").split())

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            if label[0] == ':':
                label = ':'
            elif ':' in label:
                label = ':'.join(label.split(':')[:-1])
            if len(label.strip()) < 1:
                print(tokens[index])
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                if not filter_none or (label != '-NONE-' and len(children) > 0):
                    label = label.split('-')[0]
                    trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                if not filter_none or label != '-NONE-':
                    trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    if index < len(tokens):
        print(tokens[index], 'must have been a (', tokens[index - 1])
        assert index == len(tokens), (index, len(tokens))

    if strip_top:
        for i, tree in enumerate(trees):
            if isinstance(tree, InternalTreebankNode) and tree.label == "TOP":
                assert len(tree.children) == 1, [child.label for child in tree.children]
                trees[i] = tree.children[0]

    return trees
