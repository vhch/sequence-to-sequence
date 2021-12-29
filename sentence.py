import pandas as pd
import sentencepiece as spm
import csv

df = pd.read_excel('conversation.xlsx', engine='openpyxl')
print(df)
df = df.loc[:, '원문':'번역문']

with open('translate_english.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(df['번역문']))

spm.SentencePieceTrainer.Train(
input='translate_english.txt', model_prefix='translate_english', vocab_size=30000, model_type='bpe', unk_id=0, pad_id=1, bos_id=2, eos_id=3)

with open('translate_korea.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(df['원문']))

spm.SentencePieceTrainer.Train(
input='translate_korea.txt', model_prefix='translate_korea', vocab_size=30000, model_type='bpe', unk_id=0, pad_id=1, bos_id=2, eos_id=3)

# sp = spm.SentencePieceProcessor()
# vocab_file = "translate_english.model"
# sp.load(vocab_file)

# sp_e = spm.SentencePieceProcessor()
# vocab_file = "translate_korea.model"
# sp_e.load(vocab_file)

# print(sp.IdToPiece(397))
# print(sp.IdToPiece(31))
# print(sp.IdToPiece(223))
# print(sp.IdToPiece(121))
# print(sp.IdToPiece(5))
# print(sp.IdToPiece(4574))



# sequence = "씨티은행에서 일하세요?"

# sp.SetEncodeExtraOptions('bos')
# sequence2 = "Do you work at a City bank?"
# print(sp.encode("Do you work at a City bank?", out_type=str))
# print(sp.DecodeIds([1079, 29046]))


# sp.SetEncodeExtraOptions('eos')
# sequence2 = "Do you work at a City bank?"
# print(sp.encode('sequence2', out_type=int))
