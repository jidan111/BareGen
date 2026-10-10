// fast_bbpe.cpp — Byte-level BPE tokenizer (C++17)
// 转写自 Python FastBBPE，逻辑规则保持一致，性能做了优化：
//   - pair 用 uint64_t 打包代替 tuple 作为 hash key
//   - vocab 用 vector<string> 按 ID 直接索引，避免 map 查找
//   - 并行统计/合并用 std::thread 分块，结果汇总在主线程，无锁
//   - priority_queue 直接做 min-heap，与 Python heapq 行为一致
//
// ====== 内存开关（按需修改后重新编译即可）======
// 词表 <= 65536：保留下面这行，训练序列用 uint16_t，省一半内存
// 词表 > 65536 (如 128K)：注释掉下面这行，自动用 int
#define BPE_SMALL_VOCAB
// ================================================
//
// 编译: g++ -O3 -std=c++17 -pthread fast_bbpe.cpp -o fast_bbpe
// 运行: ./fast_bbpe

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 训练序列中 token ID 的存储类型
#ifdef BPE_SMALL_VOCAB
using bpe_token_t = uint16_t;  // 词表 <= 65536 时用 2 字节，省一半内存
#else
using bpe_token_t = int;       // 通用，词表任意大小
#endif

// ============================================================
//  JSON 最小解析器（仅满足本模型 save/load 的格式）
// ============================================================
struct JsonValue {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue& at(const std::string& key) const {
        for (const auto& kv : obj) if (kv.first == key) return kv.second;
        throw std::runtime_error("JSON missing key: " + key);
    }
    const JsonValue& at(size_t i) const { return arr[i]; }
    size_t size() const { return arr.size(); }
};

struct JsonParser {
    const char* p;
    const char* end;

    explicit JsonParser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    char get() { return *p++; }
    char peek() { return *p; }
    bool match(char c) {
        skip_ws();
        if (p < end && *p == c) { p++; return true; }
        return false;
    }

    JsonValue parse() {
        skip_ws();
        if (p >= end) return {};
        if (*p == '{') { get(); return parse_obj(); }
        if (*p == '[') { get(); return parse_arr(); }
        if (*p == '"') return parse_str();
        if (strncmp(p, "true", 4) == 0) { p += 4; JsonValue v; v.type = JsonValue::Bool; v.b = true; return v; }
        if (strncmp(p, "false", 5) == 0) { p += 5; JsonValue v; v.type = JsonValue::Bool; v.b = false; return v; }
        if (strncmp(p, "null", 4) == 0) { p += 4; return {}; }
        return parse_num();
    }

    JsonValue parse_obj() {
        JsonValue v; v.type = JsonValue::Obj;
        skip_ws();
        if (match('}')) return v;
        while (true) {
            skip_ws();
            std::string key = parse_str_raw();
            match(':');
            JsonValue val = parse();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (match(',')) continue;
            break;
        }
        match('}');
        return v;
    }

    JsonValue parse_arr() {
        JsonValue v; v.type = JsonValue::Arr;
        skip_ws();
        if (match(']')) return v;
        while (true) {
            JsonValue val = parse();
            v.arr.push_back(std::move(val));
            skip_ws();
            if (match(',')) continue;
            break;
        }
        match(']');
        return v;
    }

    std::string parse_str_raw() {
        skip_ws();
        get(); // opening "
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                p++;
                char c = get();
                switch (c) {
                    case 'n': s += '\n'; break;
                    case 't': s += '\t'; break;
                    case 'r': s += '\r'; break;
                    case 'u': {
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; i++) {
                            cp <<= 4;
                            char h = get();
                            if (h >= '0' && h <= '9') cp += h - '0';
                            else if (h >= 'a' && h <= 'f') cp += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') cp += h - 'A' + 10;
                        }
                        if (cp < 0x80) s += (char)cp;
                        else if (cp < 0x800) {
                            s += (char)(0xC0 | (cp >> 6));
                            s += (char)(0x80 | (cp & 0x3F));
                        } else {
                            s += (char)(0xE0 | (cp >> 12));
                            s += (char)(0x80 | ((cp >> 6) & 0x3F));
                            s += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: s += c;
                }
            } else {
                s += get();
            }
        }
        get(); // closing "
        return s;
    }

    JsonValue parse_str() {
        JsonValue v; v.type = JsonValue::Str;
        v.str = parse_str_raw();
        return v;
    }

    JsonValue parse_num() {
        skip_ws();
        const char* start = p;
        while (p < end && (isdigit((unsigned char)*p) || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) p++;
        JsonValue v; v.type = JsonValue::Num;
        v.num = std::stod(std::string(start, p));
        return v;
    }
};

// ============================================================
//  工具函数
// ============================================================

// 判断 UTF-8 位置 i 处是否为 CJK 统一汉字 (U+4E00 ~ U+9FA5)
static bool is_chinese_utf8(const std::string& s, size_t i) {
    if (i + 2 >= s.size()) return false;
    unsigned char b0 = (unsigned char)s[i];
    if (b0 < 0xE4 || b0 > 0xE9) return false;
    unsigned char b1 = (unsigned char)s[i + 1];
    unsigned char b2 = (unsigned char)s[i + 2];
    if (b0 == 0xE4) return (b1 > 0xB8) || (b1 == 0xB8 && b2 >= 0x80);
    if (b0 == 0xE9) return (b1 < 0xBE) || (b1 == 0xBE && b2 <= 0xA5);
    return true; // E5 ~ E8 全部在范围内
}

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

// 把两个 token id 打包成 uint64_t 作为 unordered_map 的 key
static inline uint64_t pack_pair(int a, int b) {
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}
static inline int unpack_a(uint64_t k) { return (int)(uint32_t)(k >> 32); }
static inline int unpack_b(uint64_t k) { return (int)(uint32_t)(k & 0xFFFFFFFF); }

// 与 Python heapq 一致的最小堆：存 (-cnt, pair)，计数相同时选字典序最小的 pair
using BpeHeap = std::priority_queue<
    std::pair<long long, uint64_t>,
    std::vector<std::pair<long long, uint64_t>>,
    std::greater<std::pair<long long, uint64_t>>>;

// 并行 for：分块执行，bucket 为线程编号 0..num_threads-1
template <typename F>
static void parallel_for(int start, int end, int num_threads, F&& f) {
    if (start >= end) return;
    int total = end - start;
    int chunk = (total + num_threads - 1) / num_threads;
    std::vector<std::thread> pool;
    int actual_threads = 0;
    for (int t = 0; t < num_threads; t++) {
        int s = start + t * chunk;
        int e = std::min(s + chunk, end);
        if (s >= e) break;
        actual_threads++;
        pool.emplace_back([t, s, e, &f]() {
            for (int i = s; i < e; i++) f(t, i);
        });
    }
    for (auto& th : pool) th.join();
    (void)actual_threads;
}

// ============================================================
//  tqdm 风格进度条
// ============================================================
class ProgressBar {
public:
    ProgressBar(int total, const std::string& desc = "", int width = 35)
        : total_(total), desc_(desc), width_(width),
          start_(std::chrono::steady_clock::now()) {}

    void update(int n = 1) {
        done_ += n;
        render();
    }

    void set_progress(int done) {
        done_ = done;
        render();
    }

    void finish() {
        done_ = total_;
        render();
        std::cout << "\n";
        std::cout.flush();
    }

private:
    int total_;
    std::string desc_;
    int width_;
    int done_ = 0;
    std::chrono::steady_clock::time_point start_;

    static std::string fmt_time(double seconds) {
        int h = (int)(seconds / 3600);
        int m = (int)((int)seconds % 3600 / 60);
        int s = (int)seconds % 60;
        char buf[32];
        if (h > 0) snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, s);
        else       snprintf(buf, sizeof(buf), "%02d:%02d", m, s);
        return buf;
    }

    void render() {
        float ratio = total_ > 0 ? (float)done_ / total_ : 1.0f;
        if (ratio > 1.0f) ratio = 1.0f;
        int filled = (int)(width_ * ratio);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_).count();

        std::cout << "\r";
        if (!desc_.empty()) std::cout << desc_ << ": ";
        std::cout << std::fixed << std::setprecision(1) << ratio * 100.0f << "%|";
        // tqdm 用 █ 填充，░ 空白
        for (int i = 0; i < width_; i++) {
            std::cout << (i < filled ? "█" : "░");
        }
        std::cout << "| " << done_ << "/" << total_;
        std::cout << " [" << fmt_time(elapsed);

        if (done_ > 0 && done_ < total_ && elapsed > 0.01) {
            double eta = elapsed * (total_ - done_) / done_;
            std::cout << "<" << fmt_time(eta);
            double rate = done_ / elapsed;
            if (rate >= 1.0)
                std::cout << ", " << std::fixed << std::setprecision(2) << rate << "it/s";
            else
                std::cout << ", " << std::fixed << std::setprecision(2) << 1.0 / rate << "s/it";
        } else {
            std::cout << "<--:--";
        }
        std::cout << "]";
        std::cout.flush();
    }
};

// ============================================================
//  FastBBPE 类
// ============================================================
class FastBBPE {
public:
    int vocab_size;
    int min_frequency;
    std::string pad = "</PAD>";
    std::string start = "</START>";
    std::string end = "</END>";
    int pad_id = 0;
    int start_id = 0;
    int end_id = 0;
    static constexpr int base_vocab_size = 256;

    // vocab[id] = 该 token 对应的原始字节串
    std::vector<std::string> vocab;
    // merges_rank[(a,b)] = 合并后的新 token id
    std::unordered_map<uint64_t, int> merges_rank;

    explicit FastBBPE(int vocab_size_ = 2048, int min_frequency_ = 2,
                      bool load = false, const std::string& file_path = "ByteBpe.json")
        : vocab_size(vocab_size_), min_frequency(min_frequency_) {
        vocab.resize(base_vocab_size);
        for (int i = 0; i < base_vocab_size; i++) {
            vocab[i] = std::string(1, (char)(unsigned char)i);
        }
        if (load) load_file(file_path);
    }

    // 由 vocab 重建 merges_rank（与 Python __apply_merges_rank 等价）
    // 不能直接存原始字节，因为 encode 到一半 seq 里是 token ID 不是原始字节。
    // 需要把每个 vocab 条目拆成 (左子tokenID, 右子tokenID) → 新ID。
    // 关键：不能取第一个合法分割（会把单字节左子当成分割点），
    // 要取左子 token ID 最大的那个——BPE 训练时左子是最后创建的，ID 最大。
    void apply_merges_rank() {
        merges_rank.clear();
        std::unordered_map<std::string, int> bytes_to_id;
        for (int id = 0; id < (int)vocab.size(); id++) {
            bytes_to_id[vocab[id]] = id;
        }
        for (int id = 256; id < (int)vocab.size(); id++) {
            const std::string& v = vocab[id];
            int best_left = -1, best_right = -1, best_left_id = -1;
            for (size_t k = 1; k < v.size(); k++) {
                std::string left = v.substr(0, k);
                std::string right = v.substr(k);
                auto it_l = bytes_to_id.find(left);
                auto it_r = bytes_to_id.find(right);
                if (it_l != bytes_to_id.end() && it_r != bytes_to_id.end()) {
                    int lid = it_l->second, rid = it_r->second;
                    // 两个子 token 必须在当前条目之前创建（BPE 用已有 token 合并新 token）
                    if (lid < id && rid < id && lid > best_left_id) {
                        best_left_id = lid;
                        best_left = lid;
                        best_right = rid;
                    }
                }
            }
            if (best_left_id >= 0) {
                merges_rank[pack_pair(best_left, best_right)] = id;
            }
        }
    }

    // ---------- save / load ----------
    void save(const std::string& file_name = "ByteBpe.json") {
        std::ofstream ofs(file_name);
        if (!ofs) throw std::runtime_error("cannot open: " + file_name);
        ofs << "{\"vocab\": {";
        for (size_t i = 0; i < vocab.size(); i++) {
            if (i > 0) ofs << ",";
            ofs << "\"" << i << "\":[";
            for (size_t j = 0; j < vocab[i].size(); j++) {
                if (j > 0) ofs << ",";
                ofs << (int)(unsigned char)vocab[i][j];
            }
            ofs << "]";
        }
        ofs << "},\"merge_rank_reverse\": {";
        bool first = true;
        for (auto& [pair, id] : merges_rank) {
            if (!first) ofs << ",";
            first = false;
            ofs << "\"" << id << "\":[" << unpack_a(pair) << "," << unpack_b(pair) << "]";
        }
        ofs << "},\"vocab_size\":" << vocab_size
            << ",\"pad_id\":" << pad_id
            << ",\"start_id\":" << start_id
            << ",\"end_id\":" << end_id
            << "}";
    }

    void load_file(const std::string& file_name) {
        std::ifstream ifs(file_name);
        if (!ifs) throw std::runtime_error("cannot open: " + file_name);
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string content = ss.str();

        JsonParser parser(content);
        JsonValue root = parser.parse();
        const auto& voc = root.at("vocab");
        vocab.clear();
        for (const auto& kv : voc.obj) {
            int id = std::stoi(kv.first);
            std::string bytes;
            bytes.reserve(kv.second.arr.size());
            for (const auto& num : kv.second.arr) {
                bytes.push_back((char)(int)num.num);
            }
            if ((int)vocab.size() <= id) vocab.resize(id + 1);
            vocab[id] = std::move(bytes);
        }
        vocab_size = (int)root.at("vocab_size").num;
        start_id = (int)root.at("start_id").num;
        pad_id = (int)root.at("pad_id").num;
        end_id = (int)root.at("end_id").num;
        // 从磁盘读 merges_rank，不需要重建
        merges_rank.clear();
        for (const auto& kv : root.obj) {
            if (kv.first == "merge_rank_reverse") {
                for (const auto& mr : kv.second.obj) {
                    int id = std::stoi(mr.first);
                    int a = (int)mr.second.arr[0].num;
                    int b = (int)mr.second.arr[1].num;
                    merges_rank[pack_pair(a, b)] = id;
                }
                break;
            }
        }
    }

    // ---------- 预处理 ----------
    std::string clean_sentence(const std::string& sentence) const {
        std::string s = sentence;
        for (auto& c : s) c = (char)tolower((unsigned char)c);
        // Python str.lower() 会对 Unicode 字符做小写转换。
        // C++ tolower 只处理 ASCII，这里补 Latin-1 Supplement (U+00C0-U+00DE) 的 2 字节 UTF-8 小写。
        // U+00C0-U+00DE 编码为 C3 80-C3 9E（除 C3 97 = ×），小写为 C3 A0-C3 BE。
        for (size_t i = 0; i + 1 < s.size(); i++) {
            unsigned char b0 = (unsigned char)s[i];
            if (b0 == 0xC3) {
                unsigned char b1 = (unsigned char)s[i+1];
                if (b1 >= 0x80 && b1 <= 0x9E && b1 != 0x97) {
                    s[i+1] = (char)(b1 + 0x20);
                    i++; // 跳过第二个字节
                }
            }
        }
        for (auto& c : s) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        // strip
        size_t a = s.find_first_not_of(' ');
        size_t b = s.find_last_not_of(' ');
        if (a == std::string::npos) return "";
        return s.substr(a, b - a + 1);
    }

    // 与 Python preprocess 等价：clean_sentence 后直接 UTF-8 字节，不再正则切分
    std::vector<bpe_token_t> preprocess(const std::string& sentence) const {
        std::string s = clean_sentence(sentence);
        std::vector<bpe_token_t> result;
        result.reserve(s.size());
        for (unsigned char c : s) result.push_back(c);
        return result;
    }

    // BPE encode：优先级队列按 rank 顺序合并（与 Python 版一致）
    struct HeapEntry {
        int rank;
        int pos;
        uint64_t pair;
        bool operator>(const HeapEntry& o) const { return rank > o.rank; }
    };

    std::vector<bpe_token_t> bpe_merge(const std::vector<bpe_token_t>& seq) const {
        std::vector<bpe_token_t> s = seq;
        if ((int)s.size() < 2) return s;

        // 最小堆：rank 小的先合
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

        // 建堆
        for (int i = 0; i + 1 < (int)s.size(); i++) {
            uint64_t pair = pack_pair(s[i], s[i + 1]);
            auto it = merges_rank.find(pair);
            if (it != merges_rank.end()) {
                heap.push({it->second, i, pair});
            }
        }

        while (!heap.empty()) {
            auto [rank, i, pair] = heap.top();
            heap.pop();

            // 验证 pair 还有效
            if (i + 1 >= (int)s.size()) continue;
            if (pack_pair(s[i], s[i + 1]) != pair) continue;

            // 合并
            s[i] = (bpe_token_t)rank;
            s.erase(s.begin() + i + 1);

            // 新产生的相邻 pair
            if (i > 0) {
                uint64_t np = pack_pair(s[i - 1], s[i]);
                auto it = merges_rank.find(np);
                if (it != merges_rank.end()) {
                    heap.push({it->second, i - 1, np});
                }
            }
            if (i + 1 < (int)s.size()) {
                uint64_t np = pack_pair(s[i], s[i + 1]);
                auto it = merges_rank.find(np);
                if (it != merges_rank.end()) {
                    heap.push({it->second, i, np});
                }
            }
            // i 之后的 pair 全部偏移了一位，重新入堆
            for (int j = i + 1; j + 1 < (int)s.size(); j++) {
                uint64_t np = pack_pair(s[j], s[j + 1]);
                auto it = merges_rank.find(np);
                if (it != merges_rank.end()) {
                    heap.push({it->second, j, np});
                }
            }
        }
        return s;
    }

    // ---------- 训练 ----------

    // 对单个序列统计 pair -> count
    static void count_seq_pairs(const std::vector<bpe_token_t>& seq,
                                std::unordered_map<uint64_t, long long>& counter) {
        for (size_t i = 0; i + 1 < seq.size(); i++) {
            counter[pack_pair(seq[i], seq[i + 1])]++;
        }
    }

    // 并行统计全局 pair 计数与位置
    void get_counter_location(
        const std::vector<std::vector<bpe_token_t>>& byte_seq,
        int max_workers,
        std::unordered_map<uint64_t, long long>& pair_counter,
        std::unordered_map<uint64_t, std::unordered_set<int>>& pair_location,
        BpeHeap& pair_heap) {

        int N = (int)byte_seq.size();
        std::vector<std::unordered_map<uint64_t, long long>> local_counters(max_workers);
        std::vector<std::unordered_map<uint64_t, std::unordered_set<int>>> local_locs(max_workers);

        // 带进度条的并行统计
        std::atomic<int> done{0};
        {
            ProgressBar bar(N, "统计pair", 40);
            std::vector<std::thread> pool;
            int chunk = (N + max_workers - 1) / max_workers;
            for (int t = 0; t < max_workers; t++) {
                int s = t * chunk;
                int e = std::min(s + chunk, N);
                if (s >= e) break;
                pool.emplace_back([&, t, s, e]() {
                    auto& cnt = local_counters[t];
                    auto& loc = local_locs[t];
                    for (int idx = s; idx < e; idx++) {
                        const auto& seq = byte_seq[idx];
                        for (size_t i = 0; i + 1 < seq.size(); i++) {
                            uint64_t key = pack_pair(seq[i], seq[i + 1]);
                            cnt[key]++;
                            loc[key].insert(idx);
                        }
                        done.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
            // 主线程轮询进度
            while (true) {
                int d = done.load(std::memory_order_relaxed);
                bar.set_progress(d);
                if (d >= N) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            for (auto& th : pool) th.join();
            bar.finish();
        }

        // 合并
        for (int b = 0; b < max_workers; b++) {
            for (auto& [key, cnt] : local_counters[b]) {
                pair_counter[key] += cnt;
            }
            for (auto& [key, locset] : local_locs[b]) {
                for (int idx : locset) pair_location[key].insert(idx);
            }
        }

        // 过滤 min_frequency，建堆
        std::vector<uint64_t> to_remove;
        for (auto& [key, cnt] : pair_counter) {
            if (cnt <= min_frequency) {
                to_remove.push_back(key);
            } else {
                pair_heap.push({-cnt, key});  // Python: (-cnt, pair)
            }
        }
        for (auto key : to_remove) {
            pair_counter.erase(key);
            pair_location.erase(key);
        }
    }

    // 合并单个序列中的 best_pair，返回新序列 + 增量计数
    struct MergeDelta {
        std::vector<bpe_token_t> new_seq;
        std::unordered_map<uint64_t, long long> new_pairs;
        std::unordered_map<uint64_t, long long> del_pairs;
        std::unordered_set<uint64_t> vanished_pairs;  // 新序列中完全消失的 pair（count==0）
    };

    MergeDelta merge_pair_update_seq(const std::vector<bpe_token_t>& seq,
                                     uint64_t best_pair, int new_id) const {
        int a = unpack_a(best_pair);
        int b = unpack_b(best_pair);

        // 构建新序列
        std::vector<bpe_token_t> result_seq;
        result_seq.reserve(seq.size());
        size_t i = 0, length = seq.size();
        while (i < length) {
            if (i + 1 < length && seq[i] == a && seq[i + 1] == b) {
                result_seq.push_back((bpe_token_t)new_id);
                i += 2;
            } else {
                result_seq.push_back(seq[i]);
                i += 1;
            }
        }

        // 旧 pair 计数
        std::unordered_map<uint64_t, long long> old_counter;
        for (size_t k = 0; k + 1 < seq.size(); k++) {
            old_counter[pack_pair(seq[k], seq[k + 1])]++;
        }
        // 新 pair 计数
        std::unordered_map<uint64_t, long long> new_counter;
        for (size_t k = 0; k + 1 < result_seq.size(); k++) {
            new_counter[pack_pair(result_seq[k], result_seq[k + 1])]++;
        }

        MergeDelta delta;
        delta.new_seq = std::move(result_seq);
        // new_pair_counter = new_counter - old_counter (只保留正差)
        for (auto& [key, cnt] : new_counter) {
            long long old_c = 0;
            auto it = old_counter.find(key);
            if (it != old_counter.end()) old_c = it->second;
            long long diff = cnt - old_c;
            if (diff > 0) delta.new_pairs[key] = diff;
        }
        // del_pair_counter = old_counter - new_counter (只保留正差)
        for (auto& [key, cnt] : old_counter) {
            long long new_c = 0;
            auto it = new_counter.find(key);
            if (it != new_counter.end()) new_c = it->second;
            long long diff = cnt - new_c;
            if (diff > 0) delta.del_pairs[key] = diff;
            // 新序列中 count==0 的 pair，标记为完全消失
            if (new_c == 0) delta.vanished_pairs.insert(key);
        }
        return delta;
    }

    void train(const std::string& data_file,
               const std::string& save_path = "ByteBpe.json",
               int max_workers = 4, int save_step = 1000, int min_seq_len = 10) {

        // ---- resume 自动检测 ----
        // 只要 vocab 里已经有超过 256 个基础字节（即已经做过合并），就是 resume。
        // 不能用 vocab.size() > vocab_size 判断——半成品 vocab（如 756/8192）永远不会触发。
        int loaded_size = (int)vocab.size();
        bool resume = false;
        if (loaded_size > base_vocab_size) {
            if (vocab[loaded_size - 1] == end) {
                std::cout << "已有规则，且最后一个为" << end << ", 继续合并会冲突，程序退出" << std::endl;
                return;
            }
            std::cout << "检测到vocab内部已经有合并规则，切换到resume模式，在已有规则上进行合并" << std::endl;
            resume = true;
        }

        // 读取文件
        std::ifstream ifs(data_file);
        if (!ifs) throw std::runtime_error("cannot open: " + data_file);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(ifs, line)) lines.push_back(std::move(line));
        ifs.close();

        // 多线程预处理（resume 时用当前 merges 全量编码，不加特殊 token/不截断）
        int N = (int)lines.size();
        std::vector<std::vector<bpe_token_t>> preprocessed(N);
        {
            ProgressBar bar(N, "预处理", 40);
            std::atomic<int> done{0};
            std::vector<std::thread> pool;
            int chunk = (N + max_workers - 1) / max_workers;
            for (int t = 0; t < max_workers; t++) {
                int s = t * chunk;
                int e = std::min(s + chunk, N);
                if (s >= e) break;
                pool.emplace_back([&, s, e]() {
                    for (int idx = s; idx < e; idx++) {
                        auto seq = preprocess(lines[idx]);
                        if (resume) seq = bpe_merge(seq);
                        preprocessed[idx] = std::move(seq);
                        done.fetch_add(1, std::memory_order_relaxed);
                    }
                });
            }
            while (true) {
                int d = done.load(std::memory_order_relaxed);
                bar.set_progress(d);
                if (d >= N) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            for (auto& th : pool) th.join();
            bar.finish();
        }

        // 过滤短序列
        std::vector<std::vector<bpe_token_t>> byte_seq;
        for (int i = 0; i < N; i++) {
            if ((int)preprocessed[i].size() > min_seq_len) {
                byte_seq.push_back(std::move(preprocessed[i]));
            }
        }
        std::cout << "================训练数据集共 " << byte_seq.size() << " 个句子=============================" << std::endl;

        int merge_max_num = vocab_size - loaded_size - 3;
        int next_id = loaded_size;

        std::cout << "================统计全局 pair 和全局位置=============================" << std::endl;
        std::unordered_map<uint64_t, long long> pair_counter;
        std::unordered_map<uint64_t, std::unordered_set<int>> pair_location;
        BpeHeap pair_heap;

        get_counter_location(byte_seq, max_workers, pair_counter, pair_location, pair_heap);

        // 初始化黑名单：重建 bytes -> id 映射，把每个已有 vocab 条目拆成子 token ID pair
        // 防止续训时通过不同合并路径重复创建相同字节序列
        std::unordered_map<std::string, int> bytes_to_id;
        for (int id = 0; id < (int)vocab.size(); id++) {
            bytes_to_id[vocab[id]] = id;
        }
        std::unordered_set<uint64_t> merged_pairs;
        for (int id = 256; id < (int)vocab.size(); id++) {
            const std::string& v = vocab[id];
            // 尝试所有分割点，找到左半和右半都存在于 vocab 的分割
            for (size_t k = 1; k < v.size(); k++) {
                std::string left = v.substr(0, k);
                std::string right = v.substr(k);
                auto it_l = bytes_to_id.find(left);
                auto it_r = bytes_to_id.find(right);
                if (it_l != bytes_to_id.end() && it_r != bytes_to_id.end()) {
                    merged_pairs.insert(pack_pair(it_l->second, it_r->second));
                    break;
                }
            }
        }
        int total = merge_max_num;
        ProgressBar bar(total, "BPE训练", 40);
        for (int step = 0; step < total; step++) {
            // 找最优 pair
            uint64_t best_pair = UINT64_MAX; // sentinel
            while (!pair_heap.empty()) {
                auto [neg_cnt, pair] = pair_heap.top();
                pair_heap.pop();
                if (merged_pairs.count(pair)) {
                    // 已经合过了，heap/counter/location 里这条残留一起清掉
                    pair_counter.erase(pair);
                    pair_location.erase(pair);
                    continue;
                }
                auto it_loc = pair_location.find(pair);
                if (it_loc == pair_location.end() || it_loc->second.empty()) {
                    pair_counter.erase(pair);
                    pair_location.erase(pair);
                    continue;
                }
                long long current_cnt = 0;
                auto it_cnt = pair_counter.find(pair);
                if (it_cnt != pair_counter.end()) current_cnt = it_cnt->second;
                if (current_cnt <= min_frequency) continue;
                if (current_cnt == -neg_cnt) {
                    pair_heap.push({neg_cnt, pair});  // 原样推回
                    best_pair = pair;
                    break;
                } else {
                    pair_heap.push({-current_cnt, pair});  // 用更新后的计数
                }
            }
            if (best_pair == UINT64_MAX) break;
            {
                auto it = pair_counter.find(best_pair);
                if (it == pair_counter.end() || it->second <= min_frequency) break;
            }
            merged_pairs.insert(best_pair);  // 登记：这个 pair 正式合并，后续不再重复选

            // 收集需要处理的序列 index
            auto& loc_set = pair_location[best_pair];
            std::vector<int> ind_arr(loc_set.begin(), loc_set.end());

            // 并行合并
            std::vector<MergeDelta> results(ind_arr.size());
            parallel_for(0, (int)ind_arr.size(), max_workers, [&](int /*bucket*/, int idx) {
                int ind = ind_arr[idx];
                results[idx] = merge_pair_update_seq(byte_seq[ind], best_pair, next_id);
            });

            std::unordered_set<uint64_t> new_pairs;
            // 汇总
            for (int idx = 0; idx < (int)ind_arr.size(); idx++) {
                int ind = ind_arr[idx];
                byte_seq[ind] = std::move(results[idx].new_seq);
                loc_set.erase(ind);

                // 新增 pair
                for (auto& [key, cnt] : results[idx].new_pairs) {
                    pair_counter[key] += cnt;
                    pair_location[key].insert(ind);
                    new_pairs.insert(key);
                }
                // 消失 pair：只扣计数，只有 <=0 且 location 空才整体删除
                for (auto& [key, cnt] : results[idx].del_pairs) {
                    auto it = pair_counter.find(key);
                    if (it != pair_counter.end()) {
                        it->second -= cnt;
                        auto it_loc = pair_location.find(key);
                        bool loc_empty = (it_loc == pair_location.end() || it_loc->second.empty());
                        if (it->second <= 0 && loc_empty) {
                            pair_counter.erase(it);
                            pair_location.erase(key);
                        }
                        // 注意：不再在这里 erase(ind)，减少但仍存在的 pair 要保留 location
                    }
                }
                // 只有完全消失（新序列 count==0）的 pair 才从 location 删 ind
                for (uint64_t key : results[idx].vanished_pairs) {
                    auto it_loc = pair_location.find(key);
                    if (it_loc != pair_location.end()) {
                        it_loc->second.erase(ind);
                        if (it_loc->second.empty()) pair_location.erase(it_loc);
                    }
                }
            }

            // 更新堆
            for (uint64_t pair : new_pairs) {
                auto it = pair_counter.find(pair);
                if (it != pair_counter.end()) {
                    pair_heap.push({-it->second, pair});
                }
            }

            // 写入 vocab 和 merges_rank
            if ((int)vocab.size() <= next_id) vocab.resize(next_id + 1);
            int a = unpack_a(best_pair);
            int b = unpack_b(best_pair);
            vocab[next_id] = vocab[a] + vocab[b];
            merges_rank[best_pair] = next_id;  // 训练时直接记录 pair -> id
            next_id++;

            // 清理冗余堆
            if ((long long)pair_heap.size() > (long long)pair_counter.size() * 2) {
                pair_heap = BpeHeap();
                for (auto& [key, cnt] : pair_counter) {
                    pair_heap.push({-cnt, key});
                }
            }

            bar.update(1);

            if (step % save_step == 0) {
                // 临时保存（特殊 token 还没加，与 Python 一致）
                save(save_path);
            }
        }
        bar.finish();

        // 添加特殊 token
        if ((int)vocab.size() <= next_id) vocab.resize(next_id + 3);
        vocab[next_id] = pad;
        pad_id = next_id;
        vocab[next_id + 1] = start;
        start_id = next_id + 1;
        vocab[next_id + 2] = end;
        end_id = next_id + 2;

        save(save_path);
        std::cout << "\n训练完成！词表大小: " << vocab.size() << std::endl;
    }

    // ---------- encode / decode ----------

    // encode：全量编码，不加 start/end/pad，不截断（与 Python encode 一致）
    std::vector<bpe_token_t> encode_single(const std::string& sentence) const {
        std::vector<bpe_token_t> seq = preprocess(sentence);
        return bpe_merge(seq);
    }

    std::vector<std::vector<bpe_token_t>> encode_batch(const std::vector<std::string>& sentences) const {
        std::vector<std::vector<bpe_token_t>> out;
        out.reserve(sentences.size());
        for (const auto& s : sentences) out.push_back(encode_single(s));
        return out;
    }

    std::string decode_single(const std::vector<int>& tokens) const {
        std::string bytes;
        for (int t : tokens) {
            if (t >= 0 && t < (int)vocab.size()) bytes += vocab[t];
        }
        std::string result;
        // UTF-8 解码（errors=replace 行为：遇到无效字节用 U+FFFD 替换）
        // 这里简单处理：直接用 bytes 作为 UTF-8 字符串，无效序列会保留原样
        result = bytes;
        // 去掉特殊 token
        auto remove_all = [](std::string& str, const std::string& pat) {
            size_t pos = 0;
            while ((pos = str.find(pat, pos)) != std::string::npos) {
                str.erase(pos, pat.size());
            }
        };
        remove_all(result, pad);
        remove_all(result, start);
        remove_all(result, end);
        return result;
    }

    std::vector<std::string> decode_batch(const std::vector<std::vector<int>>& batch) const {
        std::vector<std::string> out;
        out.reserve(batch.size());
        for (const auto& t : batch) out.push_back(decode_single(t));
        return out;
    }
};

// ============================================================
//  main
// ============================================================
int main() {
    //g++ -std=c++17 -pthread -O3 fast_bbpe.cpp -o fast_bbpe
    FastBBPE model(/*vocab_size=*/8192, /*min_frequency=*/5,
                   /*load=*/false, /*file_path=*/"./FastBBPE.json");

    model.train(/*data_file=*/"./data/data_100K.txt",
                /*save_path=*/"./FastBBPE.json",
                /*max_workers=*/6,
                /*save_step=*/1000,
                /*min_seq_len=*/10);

//     // 测试 encode/decode
//     FastBBPE loaded(/*vocab_size=*/65536, /*min_frequency=*/20,
//                     /*load=*/true, /*file_path=*/"./FastBBPE_doubao.json");
//     std::string test = "Hello, 世界! This is a test.";
//     auto ids = loaded.encode_single(test, 128);
//     std::cout << "encoded length: " << ids.size() << std::endl;
//     std::cout << "decoded: " << loaded.decode_single(ids) << std::endl;

    return 0;
}
