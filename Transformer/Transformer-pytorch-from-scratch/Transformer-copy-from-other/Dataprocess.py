import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence


class TokenizerTrainer:
    """
    训练并保存分词器的类。
    """
    def __init__(self, vocab_size, special_tokens):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def train_and_save(self, corpus_generator, output_path, language_name):
        """
        训练分词器并保存为 JSON 文件。
        :param corpus_generator: 语料库生成器，逐行生成句子。
        :param output_path: 输出路径。
        :param language_name: 语言名称，用于文件命名。
        """
        tokenizer = Tokenizer(models.BPE(unk_token=self.special_tokens[1]))  # 使用 BPE 模型
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # 添加预分词器
        trainer = trainers.BpeTrainer(special_tokens=self.special_tokens, vocab_size=self.vocab_size)
        tokenizer.train_from_iterator(corpus_generator, trainer=trainer)
        tokenizer.save(f"{output_path}/{language_name}_tokenizer.json")

    def create_pair_file(self, source_file, target_file, pair_file):
        """
        创建源语言和目标语言的配对文件。
        :param source_file: 源语言文件路径。
        :param target_file: 目标语言文件路径。
        :param pair_file: 输出的配对文件路径。
        """
        with open(source_file, 'r', encoding='utf-8') as src_f, \
            open(target_file, 'r', encoding='utf-8') as tgt_f, \
            open(pair_file, 'w', encoding='utf-8') as pair_f:

            for src_line, tgt_line in zip(src_f, tgt_f):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if src_line and tgt_line:  # 确保句子非空
                    pair_f.write(f"{src_line}\t{tgt_line}\n")


class TranslationDataset(Dataset):
    """
    翻译任务的数据集类。
    """
    def __init__(self, config):
        """
        初始化数据集。
        :param src_file: 源语言文件路径。
        :param tgt_file: 目标语言文件路径。
        :param src_tokenizer_file: 源语言分词器文件路径。
        :param tgt_tokenizer_file: 目标语言分词器文件路径。
        :param config: 配置字典，包含 min_length, max_length, max_length 等参数。
        """
        self.src_lines, self.tgt_lines = self.load_pairs(config['pair-file'], config['min-length'], config['max-length'])
        assert len(self.src_lines) == len(self.tgt_lines), "源语言和目标语言的句子数量不匹配！"

        # 加载分词器
        self.src_tokenizer = self.load_tokenizer(config['source-tokenizer-file'])
        self.tgt_tokenizer = self.load_tokenizer(config['target-tokenizer-file'])

        self.max_length = config['max-length']

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        tgt_line = self.tgt_lines[idx]

        # 编码源语言和目标语言
        src_encoding = self.src_tokenizer(
            src_line,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tgt_encoding = self.tgt_tokenizer(
            tgt_line,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": src_encoding['input_ids'].squeeze(0),
            "attention_mask": src_encoding['attention_mask'].squeeze(0),
            "labels": tgt_encoding['input_ids'].squeeze(0)
        }

    def load_corpus(self, file_path, min_length, max_length):
        """
        加载语料库，过滤掉不符合长度要求的句子。
        :param file_path: 文件路径。
        :param min_length: 最小句子长度。
        :param max_length: 最大句子长度。
        :return: 过滤后的句子生成器。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and min_length <= len(line.split()) <= max_length:
                    yield line

    def load_tokenizer(self, tokenizer_path):
        """
        加载分词器并封装为 PreTrainedTokenizerFast 对象。
        :param tokenizer_path: 分词器文件路径。
        :return: 加载好的分词器。
        """
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]"
        )
    
    def load_pairs(self, pair_file, min_length, max_length):
        """
        加载配对文件并过滤掉不符合长度要求的句子对。
        :param pair_file: 配对文件路径。
        :param min_length: 最小句子长度。
        :param max_length: 最大句子长度。
        :return: 过滤后的源语言和目标语言句子列表。
        """
        src_lines, tgt_lines = [], []
        with open(pair_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue  # 跳过格式错误的行

                src_line, tgt_line = parts[0], parts[1]
                if (min_length <= len(src_line.split()) <= max_length and
                        min_length <= len(tgt_line.split()) <= max_length):
                    src_lines.append(src_line)
                    tgt_lines.append(tgt_line)

        return src_lines, tgt_lines

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)  # -100 是常用的忽略索引值
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 配置字典
config = {
    'pair-file': "/harddisk1/SZC-Project/NLP-learning/Transformer/Transformer-pytorch-from-scratch/en-de-pair.txt",
    'source-file': "/harddisk1/SZC-Project/NLP-learning/Transformer/Transformer-pytorch-from-scratch/europarl-v7.de-en.en",
    'target-file': "/harddisk1/SZC-Project/NLP-learning/Transformer/Transformer-pytorch-from-scratch/europarl-v7.de-en.de",
    'source-tokenizer-file': "/harddisk1/SZC-Project/NLP-learning/Transformer/Transformer-pytorch-from-scratch/en_tokenizer.json",
    'target-tokenizer-file': "/harddisk1/SZC-Project/NLP-learning/Transformer/Transformer-pytorch-from-scratch/de_tokenizer.json",
    'special-tokens': ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    'vocab-size': 30000,
    'min-length': 5,
    'max-length': 128,
    'batch-size': 64,
    'sample-ratio': 0.1,
    'num-workers': 4
}

# 测试用例
if __name__ == "__main__":
    # 创建数据集
    tokenizerTrainer = TokenizerTrainer(config['vocab-size'], config['special-tokens'])
    tokenizerTrainer.create_pair_file(source_file=config['source-file'], target_file=config['target-file'], pair_file='en-de-pair.txt')

    translation_dataset = TranslationDataset(config)

    # 创建采样器
    indices = np.random.choice(len(translation_dataset), int(len(translation_dataset) * config["sample-ratio"]), replace=False)
    sampler = SubsetRandomSampler(indices)

    # 创建数据加载器
    sampled_loader = DataLoader(
        translation_dataset,
        batch_size=config['batch-size'],
        sampler=sampler,
        num_workers=config['num-workers'],
        collate_fn=collate_fn
    )