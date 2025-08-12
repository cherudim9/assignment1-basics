from collections.abc import Iterable, Iterator
from cs336_basics.pretokenization import PRETOKENIZATION_PATTERN
import ast
from functools import reduce
import json
import multiprocessing as mp
import regex as re
import time


ENCODE_ITERABLE_CHUNK_SIZE = 2**28


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key = len, reverse = True) if special_tokens else None
        self.vocab_index = {v : k for k, v in vocab.items()}
        self.indexed_merges = [(self.vocab_index[b1], self.vocab_index[b2]) for b1, b2 in self.merges]
        self.pretoken_dict = {}


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab, merges = None, None
        with open(vocab_filepath, 'r') as vocab_f:
            vocab = ast.literal_eval(vocab_f.read())
        with open(merges_filepath) as merges_f:
            merges = ast.literal_eval(merges_f.read())

        return BpeTokenizer(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens
        )


    class BytePointer:
        def __init__(
            self,
            value: int,
        ):
            self.value = value
            self.next = None
            self.prev = None


    class BytePair:
        def __init__(self, p1, p2):
            self.p1 = p1
            self.p2 = p2


        def __hash__(self):
            return hash((self.p1, self.p2))
        

        def __eq__(self, other):
                return self.p1 == other.p1 and self.p2 == other.p2


    def encode_pretoken_set(
        self,
        pretoken_list: list[str],
    ):
        print(f'size of pretoken_list = {len(pretoken_list)}')
        pair_sets = {}
        pretoken_pointers = []
        for pretoken in pretoken_list:
            pretoken_bytes = pretoken.encode('utf_8')

            # the linked list is comprised of each bytes in the pretoken.
            li = [self.BytePointer(self.vocab_index[bytes([b])]) for b in pretoken_bytes]

            # link within the list.
            for b1, b2 in zip(li[:-1], li[1:]):
                b1.next = b2
                b2.prev = b1
                k = (b1.value, b2.value)
                if k not in pair_sets:
                    pair_sets[k] = []
                pair_sets[k].append(self.BytePair(b1, b2))

            dummy_head = self.BytePointer(-1)
            dummy_head.next = li[0]
            li[0].prev = dummy_head
            pretoken_pointers.append(dummy_head)

        for b1, b2 in self.indexed_merges:
            if (b1, b2) in pair_sets:
                for byte_pair in pair_sets[(b1, b2)]:
                    p1 = byte_pair.p1
                    p2 = byte_pair.p2
                    if p1.next != p2 or p2.prev != p1:
                        # one element of this pair is already merged with others.
                        continue

                    new_value = self.vocab_index[self.vocab[b1] + self.vocab[b2]]
                    new_p = self.BytePointer(new_value)
                    
                    # maintain the linked list.
                    if p2.next:
                        new_p.next = p2.next
                        p2.next.prev = new_p
                    if p1.prev:
                        new_p.prev = p1.prev
                        p1.prev.next = new_p

                    # create new pairs within the same pretoken.
                    if new_p.prev != None and new_p.prev.value != -1:
                        new_k = (new_p.prev.value, new_value)
                        if new_k not in pair_sets:
                            pair_sets[new_k] = []
                        pair_sets[new_k].append(self.BytePair(new_p.prev, new_p))
                    if new_p.next != None:
                        new_k = (new_value, new_p.next.value)
                        if new_k not in pair_sets:
                            pair_sets[new_k] = []
                        pair_sets[new_k].append(self.BytePair(new_p, new_p.next))
        
        for idx, p in enumerate(pretoken_pointers):
            p = p.next
            tokens = []
            while p != None:
                tokens.append(p.value)
                p = p.next
            self.pretoken_dict[pretoken_list[idx]] = tokens


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            chunks_splitted = re.split('(' + '|'.join(map(re.escape, self.special_tokens)) + ')', text)
        else:
            chunks_splitted = [text]

        pretoken_set = set()
        for chunk in chunks_splitted:
            if not (self.special_tokens and chunk in self.special_tokens):
                for pretoken in re.finditer(PRETOKENIZATION_PATTERN, chunk):
                    if len(pretoken.group(0)) == 0 or pretoken.group(0) in self.pretoken_dict:
                        continue
                    pretoken_set.add(pretoken.group(0))

        self.encode_pretoken_set(list(pretoken_set))

        results = []
        for chunk in chunks_splitted:
            if self.special_tokens and chunk in self.special_tokens:
                results.append(self.vocab_index[chunk.encode('utf-8')])
            else:
                for pretoken in re.finditer(PRETOKENIZATION_PATTERN, chunk):
                    if len(pretoken) == 0:
                        continue
                    results += self.pretoken_dict[pretoken.group(0)]

        return results
    

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        acc_s = ''
        special_tokens_pattern = re.compile('|'.join(map(re.escape, self.special_tokens))) if self.special_tokens else None
        for s in iterable:
            if special_tokens_pattern:
                match = special_tokens_pattern.search(s)
            else:
                match = True
            if len(acc_s) > ENCODE_ITERABLE_CHUNK_SIZE and match:
                start_time = time.time()
                print(f'processing acc_s of length={len(acc_s)}, sample="{acc_s[:100]}"...')
                if isinstance(match, bool):
                    end_index = len(s)
                else:
                    end_index = match.span()[1]
                acc_s += s[:end_index]
                token_ids = self.encode(acc_s)
                end_time = time.time()
                print(f'runtime = {end_time - start_time}')
                acc_s = s[end_index:]
                for token_id in token_ids:
                    yield token_id
            else:
                acc_s += s

        if len(acc_s) > 0:
            token_ids = self.encode(acc_s)
            acc_s = ''
            for token_id in token_ids:
                yield token_id
    
    def decode(
        self,
        ids: list[int]
    ) -> str:
        if ids and len(ids) > 0:
            return reduce(lambda a, b: a + b, map(lambda x: self.vocab[x], ids)).decode('utf-8', errors = 'replace')
        else:
            return ""