# WavTokenizer
SOTA Discrete Codec Models With Forty Tokens Per Second for Audio Language Modeling 



[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.16532)
[![demo](https://img.shields.io/badge/WanTokenizer-Demo-red)](https://wavtokenizer.github.io/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20WavTokenizer-Models-blue)](https://huggingface.co/novateur/WavTokenizer)



### 🎉🎉 with WavTokenizer, you can represent speech, music, and audio with only 40 tokens per second!
### 🎉🎉 with WavTokenizer, You can get strong reconstruction results.
### 🎉🎉 WavTokenizer owns rich semantic information and is build for audio language models such as GPT-4o.

<!--
# Tips
We have noticed that several works (approximately exceed ten recent months) have incorrectly cited WavTokenizer. Below is the correct citation format. We sincerely appreciate the community's attention and interest.
```
@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}
```
-->

# 🔥 News
- *2025.02.25*: We update WavTokenizer camera ready version for ICLR 2025 and update WavTokenizer-large-v2 checkpoint on [huggingface](https://huggingface.co/novateur/WavTokenizer-large-speech-75token). 
- *2024.10.22*: We update WavTokenizer on arxiv and release WavTokenizer-Large checkpoint.
- *2024.09.09*: We release WavTokenizer-medium checkpoint on [huggingface](https://huggingface.co/collections/novateur/wavtokenizer-medium-large-66de94b6fd7d68a2933e4fc0).
- *2024.08.31*: We release WavTokenizer on arxiv.

![result](result.png)


## Installation

```bash
pip3 install git+https://github.com/mesolitica/WavTokenizer-package
```

## Encode decode

```python
from wavtokenizer.encoder.utils import convert_audio
import torchaudio
import torch
from wavtokenizer.decoder.pretrained import WavTokenizer

config_path = "configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

# !wget https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_v2.ckpt
model_path = "wavtokenizer_large_speech_320_v2.ckpt"

model = WavTokenizer.from_pretrained0802(config_path, model_path)

wav, sr = torchaudio.load('husein-assistant-trim.mp3')
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])

_, discrete_code = model.encode_infer(wav, bandwidth_id=bandwidth_id)
features = model.codes_to_features(discrete_code)

audio_out = model.decode(features, bandwidth_id=bandwidth_id)
```

Config and model also already mirrored at https://huggingface.co/huseinzol05/WavTokenizer-mirror

## Available models
🤗 links to the Huggingface model hub.

| Model name                                                          |                                                                                                            HuggingFace                                                                                                             |  Corpus  |  Token/s  | Domain | Open-Source |
|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------:|:----------:|:------:|
| WavTokenizer-small-600-24k-4096             |             [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_600_24k_4096.ckpt)    | LibriTTS  | 40  |  Speech  | √ |
| WavTokenizer-small-320-24k-4096             |             [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_320_24k_4096.ckpt)     | LibriTTS  | 75 |  Speech  | √|
| WavTokenizer-medium-320-24k-4096                 |               [🤗](https://huggingface.co/collections/novateur/wavtokenizer-medium-large-66de94b6fd7d68a2933e4fc0)         | 10000 Hours | 75 |  Speech, Audio, Music  | √ |
| WavTokenizer-large-600-24k-4096 | [🤗](https://huggingface.co/novateur/WavTokenizer-large-unify-40token) | 80000 Hours | 40 |   Speech, Audio, Music   | √|
| WavTokenizer-large-320-24k-4096   | [🤗](https://huggingface.co/novateur/WavTokenizer-large-speech-75token) | 80000 Hours | 75 |   Speech, Audio, Music   | √ |

      

## Training

### Step1: Prepare train dataset
```python
# Process the data into a form similar to ./data/demo.txt
```

### Step2: Modifying configuration files
```python
# ./configs/xxx.yaml
# Modify the values of parameters such as batch_size, filelist_path, save_dir, device
```

### Step3: Start training process
Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

```bash
cd ./WavTokenizer
python train.py fit --config ./configs/xxx.yaml
```


## Citation

If this code contributes to your research, please cite our work, Language-Codec and WavTokenizer:

```
@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}

@article{ji2024language,
  title={Language-codec: Reducing the gaps between discrete codec representation and speech language models},
  author={Ji, Shengpeng and Fang, Minghui and Jiang, Ziyue and Huang, Rongjie and Zuo, Jialung and Wang, Shulei and Zhao, Zhou},
  journal={arXiv preprint arXiv:2402.12208},
  year={2024}
}
```

