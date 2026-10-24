from ..import_packages import *


class FastBBPE(object):
    """
        ByteBPE
        </start>, token1, token2 ... tokenN </end> </pad> <</pad> </pad>
        """

    def __init__(self, vocab_size=2048, min_frequency=2, load=False, file_path="ByteBpe.json"):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pad, self.start, self.end = "</PAD>", "</START>", "</END>"
        self.pad_id, self.start_id, self.end_id = 0, 0, 0
        self.base_vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(self.base_vocab_size)}  # 存
        self.merges_rank = {}  # 存数组规则，只存规则pair，类似字典{pair: ind,...}
        if load:
            self.load(file_name=file_path)

    def save(self, file_name="ByteBpe.json"):
        vocab = {}
        for k, v in self.vocab.items():
            vocab[k] = list(v)
        merge_rank_reverse = {}
        for k, v in self.merges_rank.items():
            merge_rank_reverse[v] = list(k)
        result = {"vocab": vocab,
                  "merge_rank_reverse": merge_rank_reverse,
                  "vocab_size": self.vocab_size,
                  "pad_id": self.pad_id,
                  "start_id": self.start_id,
                  "end_id": self.end_id}
        with open(file_name, "w") as file:
            file.write(json.dumps(result, ensure_ascii=False))

    def load(self, file_name):
        with open(file_name, "r") as file:
            params = json.loads(file.read())
        vocab = {}
        merge_rank = {}
        for k, v in params["vocab"].items():
            vocab[int(k)] = bytes(v)
        for k, v in params["merge_rank_reverse"].items():
            merge_rank[tuple(v)] = int(k)
        self.vocab = vocab
        self.merges_rank = merge_rank
        self.vocab_size = params["vocab_size"]
        self.start_id = params["start_id"]
        self.pad_id = params["pad_id"]
        self.end_id = params["end_id"]

    def clean_sentence(self, sentence):
        s = sentence.lower()
        s = re.sub(r'[\n\r\t]+', ' ', s)
        return s.strip()

    def preprocess(self, seq, resume=False):
        seq = self.clean_sentence(seq)
        result_seq = list(bytes(seq.encode("utf-8")))
        if resume:
            result_seq = self.__encode(result_seq)
        return result_seq

    def __counter_seq(self, seq, ind):
        return Counter(list(zip(seq, seq[1:]))), ind

    def get_counter_location(self, byte_seq, max_workers=4):
        pair_counter = Counter()
        pair_location = defaultdict(set)
        pair_heap = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = [
                executor.submit(self.__counter_seq, seq, ind)
                for ind, seq in enumerate(tqdm(byte_seq))
            ]
            for future in as_completed(future_to_file):
                counter, ind = future.result()
                pair_counter.update(counter)
                for pair, cnt in counter.items():
                    pair_location[pair].add(ind)
        for pair, cnt in pair_counter.items():
            if cnt <= self.min_frequency:
                continue
            heapq.heappush(pair_heap, (-cnt, pair))
        return pair_counter, pair_location, pair_heap

    def merge_pair_update_seq(self, seq, best_pair, new_id, ind):
        result_seq = []
        old_counter = Counter(list(zip(seq, seq[1:])))
        a, b = best_pair
        length = len(seq)
        i = 0
        while i < length:
            if i < length - 1 and seq[i] == a and seq[i + 1] == b:
                result_seq.append(new_id)
                i += 2
            else:
                result_seq.append(seq[i])
                i += 1
        new_counter = Counter(list(zip(result_seq, result_seq[1:])))
        new_pair_counter = new_counter - old_counter
        del_pair_counter = old_counter - new_counter
        vanished_pairs = set()
        for pair in old_counter:
            if new_counter.get(pair, 0) == 0:
                vanished_pairs.add(pair)
        return result_seq, new_pair_counter, del_pair_counter, vanished_pairs, ind

    def train(self, sentences, save_path="ByteBpe.json", max_workers=4, save_step=1000, min_seq_len=10
              , resume=False):
        base_vocab_size = len(self.vocab)
        if base_vocab_size > self.base_vocab_size:  # 调用了load，会出现resume情况，自动检测
            if self.vocab[base_vocab_size - 1] == bytes(self.end.encode("utf-8")):
                print(f"已有规则，且最后一个为{self.end}, 继续合并会冲突，程序退出")
                return None, None, None
            print("检测到vocab内部已经有合并规则，切换到resume模式，在已有规则上进行合并")
            resume = True
        merge_count = 0
        merge_max_num = self.vocab_size - base_vocab_size - 3
        next_id = base_vocab_size
        byte_seq = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = [
                executor.submit(self.preprocess, s, resume)
                for s in sentences
            ]
            for future in as_completed(future_to_file):
                result_seq = future.result()
                if len(result_seq) <= min_seq_len:
                    continue
                byte_seq.append(result_seq)
        print(f"================训练数据集共{len(byte_seq)}个句子=============================")
        # 构建pair_counter, pair_location, pair_heap
        print(f"================统计全局pair和全局位置=============================")
        pair_counter, pair_location, pair_heap = self.get_counter_location(byte_seq, max_workers=max_workers)
        # 已合并过的 pair，防止同一个 pair 被 heap 重新选中、产生重复 vocab
        bytes_to_id = {v: k for k, v in self.vocab.items()}
        merged_pairs = set()
        for id in range(256, len(self.vocab)):
            v = self.vocab[id]
            for k in range(1, len(v)):
                left, right = v[:k], v[k:]
                if left in bytes_to_id and right in bytes_to_id:
                    merged_pairs.add((bytes_to_id[left], bytes_to_id[right]))
                    break
        for _ in tqdm(range(merge_max_num)):
            best_pair = (-1, -1)
            while pair_heap:
                neg_cnt, pair = heapq.heappop(pair_heap)
                if pair in merged_pairs:
                    # 已经合过了，heap/counter/location 里这条残留一起清掉
                    pair_counter.pop(pair, None)
                    pair_location.pop(pair, None)
                    continue
                current_cnt = pair_counter.get(pair, 0)
                if not pair_location[pair]:
                    pair_counter.pop(pair, None)
                    pair_location.pop(pair, None)
                    continue
                if current_cnt <= self.min_frequency:
                    continue  # 已经没了
                if current_cnt == -neg_cnt:
                    heapq.heappush(pair_heap, (-current_cnt, pair))
                    best_pair = pair
                    break
                else:
                    heapq.heappush(pair_heap, (-current_cnt, pair))
            if not pair_heap:
                break
            if best_pair == (-1, -1):
                break
            if pair_counter[best_pair] <= self.min_frequency:
                break
            merged_pairs.add(best_pair)  # 登记：这个 pair 正式合并，后续不再重复选
            ind_arr = pair_location[best_pair]
            new_pairs = set()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = [
                    executor.submit(self.merge_pair_update_seq, byte_seq[ind], best_pair, next_id, ind)
                    for ind in ind_arr
                ]
                for future in as_completed(future_to_file):
                    result_seq, new_pair_counter, del_pair_counter, vanished_pairs, ind = future.result()
                    byte_seq[ind] = result_seq
                    pair_location[best_pair].discard(ind)
                    pair_counter.update(new_pair_counter)  # 新pair操作
                    for pair, cnt in new_pair_counter.items():
                        pair_location[pair].add(ind)
                        new_pairs.add(pair)
                    # 消失pair操作
                    for pair, cnt in del_pair_counter.items():
                        pair_counter[pair] -= cnt
                        if pair_counter[pair] <= 0 and not pair_location[pair]:
                            pair_counter.pop(pair, None)
                            pair_location.pop(pair, None)
                    for pair in vanished_pairs:
                        pair_location[pair].discard(ind)
            for pair in new_pairs:
                heapq.heappush(pair_heap, (-pair_counter[pair], pair))  # 堆更新new_pair
            a, b = best_pair
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            self.merges_rank[best_pair] = next_id
            merge_count += 1
            next_id += 1
            if len(pair_heap) > len(pair_counter) * 2:  # 清除冗余堆数据
                pair_heap = []
                for pair, cnt in pair_counter.items():
                    heapq.heappush(pair_heap, (-cnt, pair))
            if _ % save_step == 0:
                self.save(file_name=save_path)
        self.vocab[next_id] = bytes(self.pad.encode("utf-8"))
        self.pad_id = next_id
        self.vocab[next_id + 1] = bytes(self.start.encode("utf-8"))
        self.start_id = next_id + 1
        self.vocab[next_id + 2] = bytes(self.end.encode("utf-8"))
        self.end_id = next_id + 2
        self.save(file_name=save_path)
        print(f"\n训练完成！词表大小: {len(self.vocab)}")
        return pair_counter, pair_location, pair_heap

    def __encode(self, seq):
        if len(seq) < 2:
            return seq
        heap = []
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            if pair in self.merges_rank:
                heapq.heappush(heap, (self.merges_rank[pair], i, pair))
        while heap:
            rank, i, pair = heapq.heappop(heap)
            if i + 1 >= len(seq):
                continue
            if (seq[i], seq[i + 1]) != pair:
                continue
            # 合并
            seq[i] = rank
            del seq[i + 1]
            # 新产生的相邻 pair
            if i > 0:
                np = (seq[i - 1], seq[i])
                if np in self.merges_rank:
                    heapq.heappush(heap, (self.merges_rank[np], i - 1, np))
            if i + 1 < len(seq):
                np = (seq[i], seq[i + 1])
                if np in self.merges_rank:
                    heapq.heappush(heap, (self.merges_rank[np], i, np))
            # 关键：i 之后的 pair 全部偏移了，重新入堆
            for j in range(i + 1, len(seq) - 1):
                np = (seq[j], seq[j + 1])
                if np in self.merges_rank:
                    heapq.heappush(heap, (self.merges_rank[np], j, np))
        return seq

    def __decode(self, tokens):
        seq = []
        for token in tokens:
            seq.extend(list(self.vocab[token]))
        return bytes(seq).decode('utf-8', errors='replace').replace(self.pad, '').replace(self.start, '').replace(
            self.end, '')

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = self.preprocess(sentences)
            out = self.__encode(sentences)
        else:
            out = []
            for s in sentences:
                s = self.preprocess(s)
                out.append(self.__encode(s))
        return out

    def decode(self, tokens):
        if isinstance(tokens[0], int):
            out = self.__decode(tokens)
        else:
            out = []
            for t in tokens:
                out.append(self.__decode(t))
        return out


if __name__ == "__main__":
    model = FastBBPE(vocab_size=65536, min_frequency=20, load=False, file_path="./FastBBPE.json")
    file = open("./data.txt", "r")
    model.train(sentences=file, save_path="./FastBBPE.json", max_workers=6, save_step=1000, min_seq_len=10)
    file.close()
