import sentencepiece as spm
import os
import re
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as model


def filter_vocab(vocab, llama_spm_tokens_set):
    filtered_vocab = []
    special_tokens = ['。', '，', '?', ':', '.', '(', ')', '、', '【', '】', ',']
    for piece in vocab:
        for special_token in special_tokens:
            if special_token in piece:
                continue
        # 过滤特殊词汇
        if piece not in llama_spm_tokens_set:
            filtered_vocab.append(piece)
    return filtered_vocab


def generate_custom_vocab(input_files, vocab_path, model_path, vocab_size=32000):
    # 加载 llama 词表
    llama_spm = model.ModelProto()
    llama_spm.ParseFromString(open(model_path, "rb").read())
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    # 创建模型前缀和输入文件列表字符串
    model_prefix = "medical"
    input_files_str = ','.join(input_files)

    # 训练 SentencePiece 模型
    spm.SentencePieceTrainer.train(
        f'--input={input_files_str} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=0.9995 --shuffle_input_sentence=true')

    # 载入训练好的模型
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')

    # 创建空词汇表列表
    vocab_list = []

    # 将词汇添加到词汇表列表中
    for id in range(sp.get_piece_size()):
        piece = sp.id_to_piece(id)
        vocab_list.append(piece)

    # 过滤词汇表
    filtered_vocab_list = filter_vocab(vocab_list, llama_spm_tokens_set)

    # 将词汇表写入到文件中
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in filtered_vocab_list:
            f.write(word + "\n")

    # 删除临时生成的模型文件
    if os.path.exists(f'{model_prefix}.model'):
        os.remove(f'{model_prefix}.model')
    if os.path.exists(f'{model_prefix}.vocab'):
        os.remove(f'{model_prefix}.vocab')

    print(f"Vocabulary file saved to {vocab_path}")


def merge_vocab_and_tokenizer(load_path, save_dir, voc_path, test_text):
    # Load pre-trained llama tokenizer and sentencepiece model
    llama_spm = model.ModelProto()
    llama_spm.ParseFromString(open(load_path, "rb").read())

    # Load custom vocabulary
    new_tokens = open(voc_path, "r").read().split("\n")
    for token in new_tokens:
        # Skip empty tokens
        if not token:
            continue
        if token not in [p.piece for p in llama_spm.pieces]:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            llama_spm.pieces.append(new_token)

    # save
    os.makedirs(save_dir, exist_ok=True)
    save_model_path = os.path.join(save_dir, 'tokenizer.model')
    save_vocab_path = os.path.join(save_dir, 'tokenizer.vocab')
    with open(save_model_path, 'wb') as f:
        f.write(llama_spm.SerializeToString())
    with open(save_vocab_path, 'w') as f:
        f.writelines([f'{token.piece} {token.score}\n' for token in llama_spm.pieces])
    tokenizer = LlamaTokenizer(save_model_path)
    tokenizer.save_pretrained(save_dir)
    print(f'New llama tokenizer and spm has been saved to {save_dir}')

    # test
    llama_tokenizer_old = LlamaTokenizer.from_pretrained(load_path)
    llama_tokenizer_new = LlamaTokenizer.from_pretrained(save_dir)

    print(f'Size of old vocabulary: {llama_tokenizer_old.vocab_size}')
    print(f'Size of new vocabulary: {llama_tokenizer_new.vocab_size}')
    print('All special tokens and ids in new llama:')
    print(llama_tokenizer_new.all_special_tokens)
    print(llama_tokenizer_new.all_special_ids)
    print(llama_tokenizer_new.special_tokens_map)

    print(f'Text:\n{test_text}')
    print(f'Tokenized by LLaMA tokenizer:\n {llama_tokenizer_old.tokenize(test_text)}')
    print(f'Tokenized by NEW LLaMA tokenizer:\n {llama_tokenizer_new.tokenize(test_text)}')


if __name__ == "__main__":
    input_file = ['../../data/pretrain_data/CP.txt',
                  '../../data/pretrain_data/cmekg.txt',
                  '../../data/pretrain_data/huatuo_encyclopedia_qa.txt',
                  '../../data/pretrain_data/huatuo_knowledge_graph_qa.txt',
                  '../../data/pretrain_data/medical_book.txt',
                  '../../data/pretrain_data/PromptCBLUE.txt',
                  '../../data/pretrain_data/gpt_电子病历.txt',
                  '../../data/pretrain_data/gpt_临床报告.txt',
                  ]
    output_vocab_path = '../../data/vocab/medical_vocab.txt'
    # tokenizer_path = '/home2/mrwu/models/chinese-llama-plus-7b/tokenizer.model'
    tokenizer_path = '/root/autodl-tmp/chinese-llama-plus-7b/tokenizer.model'
    generate_custom_vocab(input_file, output_vocab_path, tokenizer_path)

    save_dir = '/root/autodl-tmp/chinese-llama-plus-7b-med_tokenizer'
    test_text = '建议患者休息，多饮水，避免受凉。如持续发热，建议服用退烧药。如咳嗽症状加重，建议咨询医师并配合抗生素治疗。定期观察病情变化，如有需要请及时就医。'
    merge_vocab_and_tokenizer(tokenizer_path, save_dir, output_vocab_path, test_text)
