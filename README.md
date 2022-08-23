# mDeBERTa-v3-base-kor-further
Further pre-trained mDeBERTa-v3-base Model with Korean dataset

> 💡 아래 프로젝트는 KPMG Lighthouse Korea에서 진행하였습니다.   
> KPMG Lighthouse Korea에서는, Financial area의 다양한 문제들을 해결하기 위해 Edge Technology의 NLP/Vision AI를 모델링하고 있습니다.
  

## What is DeBERTa?
- [DeBERTa](https://arxiv.org/abs/2006.03654)는 `Disentangled Attention` + `Enhanced Mask Decoder` 를 적용하여 단어의 positional information을 효과적으로 학습합니다. 이와 같은 아이디어를 통해, 기존의 BERT, RoBERTa에서 사용했던 absolute position embedding과는 달리 DeBERTa는 단어의 상대적인 위치 정보를 학습 가능한 벡터로 표현하여 모델을 학습하게 됩니다. 결과적으로, BERT, RoBERTA 와 비교했을 때 더 준수한 성능을 보여주었습니다.
- [DeBERTa-v3](https://arxiv.org/abs/2111.09543)에서는, 이전 버전에서 사용했던 MLM (Masked Language Model) 을 RTD (Replaced Token Detection) Task 로 대체한 ELECTRA 스타일의 사전학습 방법과, Gradient-Disentangled Embedding Sharing 을 적용하여 모델 학습의 효율성을 개선하였습니다.
- DeBERTa의 아키텍처로 풍부한 한국어 데이터를 학습하기 위해서,  `mDeBERTa-v3-base-kor-further` 는 microsoft 가 발표한 `mDeBERTa-v3-base` 를 약 40GB의 한국어 데이터에 대해서 **추가적인 사전학습**을 진행한 언어 모델입니다.
  
## How to Use
- Requirements
    ```
    pip install transformers
    pip install sentencepiece
    ```   
- [Huggingface Hub](https://huggingface.co/lighthouse/mdeberta-v3-base-kor-further)
    ```python
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("mdeberta-v3-base-kor-further")  # DebertaV2ForModel
    tokenizer = AutoTokenizer.from_pretrained("mdeberta-v3-base-kor-further")  # DebertaV2Tokenizer (SentencePiece)
    ```

## Pre-trained Models
- 모델의 아키텍처는 기존 microsoft에서 발표한 `mdeberta-v3-base`와 동일한 구조입니다.
    
    |  | Vocabulary(K) | Backbone Parameters(M) | Hidden Size | Layers | Note |
    | --- | --- | --- | --- | --- | --- |
    | mdeberta-v3-base-kor-further (mdeberta-v3-base와 동일) | 250 | 86 | 768 | 12 | 250K new SPM vocab |

## Further Pretraing Details (MLM Task)
- `mDeBERTa-v3-base-kor-further` 는 `microsoft/mDeBERTa-v3-base` 를 약 40GB의 한국어 데이터에 대해서 MLM Task를 적용하여 추가적인 사전 학습을 진행하였습니다.
    
    |  | Max length | Learning Rate | Batch Size | Train Steps | Warm-up Steps |
    | --- | --- | --- | --- | --- | --- |
    | mdeberta-v3-base-kor-further | 512 | 2e-5 | 8 | 5M | 50k |
    

## Datasets
- 모두의 말뭉치(신문, 구어, 문어), 한국어 Wiki, 국민청원 등 약 40 GB 의 한국어 데이터셋이 추가적인 사전학습에 사용되었습니다.
    - Train: 10M lines, 5B tokens
    - Valid: 2M lines, 1B tokens
    - cf) 기존 mDeBERTa-v3은 XLM-R 과 같이 [cc-100 데이터셋](https://data.statmt.org/cc-100/)으로 학습되었으며, 그 중 한국어 데이터셋의 크기는 54GB입니다.
    

## Fine-tuning on NLU Tasks - Base Model
| Model | Size | NSMC(acc) | Naver NER(F1) | PAWS (acc) | KorNLI (acc) | KorSTS (spearman) | Question Pair (acc) | KorQuaD (Dev) (EM/F1) | Korean-Hate-Speech (Dev) (F1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XLM-Roberta-Base | 1.03G | 89.03 | 86.65 | 82.80 | 80.23 | 78.45 | 93.80 | 64.70 / 88.94 | 64.06 |
| mdeberta-base | 534M | 90.01 | 87.43 | 85.55 | 80.41 | **82.65** | 94.06 | 65.48 / 89.74 | 62.91 |
| mdeberta-base-kor-further (Ours) | 534M | **90.52** | **87.87** | **85.85** | **80.65** | 81.90 | **94.98** | **66.07 / 90.35** | **68.16** |

## Citation
```
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```

## Reference
- [DeBERTa](https://github.com/microsoft/DeBERTa)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [모두의 말뭉치](https://corpus.korean.go.kr/)
- [Korpora: Korean Corpora Archives](https://github.com/ko-nlp/Korpora)
- [sooftware/Korean PLM](https://github.com/sooftware/Korean-PLM)
