# Moss Audio Tokenizer ONNX

![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX-1.20.1-green.svg)
![Onnx Runtime](https://img.shields.io/badge/Onnx%20Runtime-1.23.2-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg)

## 模型简介

MOSSAudioTokenizer 是一种基于 Cat（Causal Audio Tokenizer with Transformer）架构的统一离散音频分词器。该模型参数量达到 16 亿，可作为统一的离散接口，同时实现无损质量重建和高层语义对齐。

主要特性：

    极致压缩与可变比特率：将 24kHz 原始音频压缩至极低的 12.5Hz 帧率。通过 32 层残差矢量量化器（RVQ），支持从 0.125kbps 到 4kbps 的宽范围比特率下的高保真重建。
    纯 Transformer 架构：模型采用“无 CNN”的同质架构，完全由因果 Transformer 模块构建。编码器与解码器合计 16 亿参数，确保卓越的可扩展性，并支持低延迟流式推理。
    大规模通用音频训练：在 300 万小时多样化音频数据上进行训练，能够出色地编码和重建所有音频领域内容，包括语音、音效和音乐。
    统一的语义-声学表征：在实现当前最优重建质量的同时，Cat 生成的离散 token 具有“丰富的语义”，非常适合下游任务，如语音理解（ASR）和语音合成（TTS）。
    完全从零训练：Cat 不依赖任何预训练编码器（如 HuBERT 或 Whisper），也不使用教师模型蒸馏。所有表征均从原始数据中自主学习。
    端到端联合优化：所有组件——包括编码器、量化器、解码器、判别器，以及用于语义对齐的仅解码器 LLM——均在单一统一训练流程中联合优化。

总结： 通过结合简洁可扩展的架构与大规模数据，Cat 架构突破了传统音频分词器的瓶颈，为下一代原生音频基础模型提供了鲁棒、高保真且语义扎实的接口。

本仓库包含了 MOSSAudioTokenizer Decoder 的 ONNX 模型文件，以及使用该模型进行音频解码的 Py 代码。

![架构图](./rep_sources/arch.png)

## 开源仓库

 - ModelScope: [https://www.modelscope.cn/models/KeanuX/MossAudioTokenizer-ONNX](https://www.modelscope.cn/models/KeanuX/MossAudioTokenizer-ONNX)

 - GitHub: [https://github.com/201831771214/MossAudioTokenizer-ONNX.git](https://github.com/201831771214/MossAudioTokenizer-ONNX.git)

## Clone 仓库

```shell
# 获取仓库源码
git clone https://github.com/201831771214/MossAudioTokenizer-ONNX.git

# 获取模型
modelscope download --model KeanuX/MossAudioTokenizer-ONNX --local_dir ./
```

### Git仓库结构

```shell
../../VCProjects/MossAudioTokenizerDecoder-ONNX/
├── audio_tokens.npy
├── check_onnx.py
├── export_audio_tokenizer.py
├── export_model_info.py
├── generated_audio.wav
├── infos
│   └── audio_tokenizer_decoder.info
├── logs
│   ├── check_onnx.log
│   └── run_onnx.log
├── models
│   └── moss_audio_tokenizer_decoder_onnx
│       ├── 2005f62a-1458-11f1-80d4-cc28aa3bf0f5.data
│       ├── 2799b6f6-1458-11f1-80d4-cc28aa3bf0f5.data
│       ├── audio_tokenizer_decoder.onnx
│       └── audio_tokenizer_encoder.onnx
├── moss_audio_tokenizer
│   ├── config.json
│   ├── configuration_moss_audio_tokenizer.py
│   ├── demo
│   │   ├── demo_gt.wav
│   │   └── test_reconstruction.py
│   ├── images
│   │   ├── arch.png
│   │   ├── metrics_on_librispeech_test_clean.png
│   │   ├── mosi-logo.png
│   │   ├── OpenMOSS_Logo.png
│   │   └── reconstruct_comparison_table.png
│   ├── __init__.py
│   ├── LICENSE
│   ├── modeling_moss_audio_tokenizer.py
│   ├── __pycache__
│   │   ├── configuration_moss_audio_tokenizer.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   └── modeling_moss_audio_tokenizer.cpython-310.pyc
│   ├── README.md
│   └── requirements.txt
├── moss_extra
│   ├── audio_tokenizer_decoder.py
│   └── audio_tokenizer_encoder.py
├── README_Git.md
├── README_ModelScope.md
├── rep_sources
│   ├── arch.png
│   ├── License-MIT-yellow.png
│   ├── NumPy-1.26.4-blue.png
│   ├── ONNX-1.20.1-green.png
│   ├── Onnx Runtime-1.23.2-blue.png
│   └── Python-3.10+-blue.png
├── requirements.txt
├── run_decoder.py
└── run_encoder.py

11 directories, 42 files
```

## 使用方法

### 模型信息

```txt
#### Audio Decoder Infos ####
============================================================
ONNX模型基本信息
============================================================
模型文件路径: ./models/moss_audio_tokenizer_decoder_onnx/audio_tokenizer_decoder.onnx
ONNX版本: 7
生产者信息: pytorch 2.8.0
模型版本: 0
描述: 

============================================================
模型输入信息 (共 1 个输入)
============================================================
Input 1: audio_codes
  数据类型: int32
  形状: [0, 0, 0]

============================================================
模型输出信息 (共 1 个输出)
============================================================
Output 1: audio
  数据类型: float32
  形状: [0, 0]

#### Audio Encoder Infos ####
============================================================
ONNX模型基本信息
============================================================
模型文件路径: ./models/moss_audio_tokenizer_decoder_onnx/audio_tokenizer_encoder.onnx
ONNX版本: 7
生产者信息: pytorch 2.8.0
模型版本: 0
描述: 

============================================================
模型输入信息 (共 1 个输入)
============================================================
Input 1: input_values
  数据类型: float32
  形状: [0, 0, 0]

============================================================
模型输出信息 (共 3 个输出)
============================================================
Output 1: audio_codes
  数据类型: int64
  形状: [32, 0, 0]

Output 2: audio_codes_lengths
  数据类型: int64
  形状: [0]

Output 3: encoder_hidden_states
  数据类型: float32
  形状: [0, 768, 0]
```

详细信息请参考: [ONNX模型信息](https://github.com/201831771214/MossAudioTokenizer-ONNX/infos/)

### 安装依赖

```shell
# 安装依赖
pip install -r requirements.txt
```

### 快速入门

#### Run Audio Tokenizer Decoder

```python
import onnxruntime as ort
import numpy as np
from numpy.typing import NDArray
import soundfile as sf
from typing import Tuple
import os
import sys

import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/run_onnx.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class AudioTokenizerDecoder:
    def __init__(self, model_path:str, device:str="cuda"):
        self.model_path = model_path
        self.device = device.lower()
        
        logger.info(f"All available providers: {ort.get_available_providers()}")
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider"]
        elif device == "cpu" and "CPUExecutionProvider" in ort.get_available_providers():
            self.providers = ["CPUExecutionProvider"]
        else:
            logger.warning(f"Device {device} is not supported. Fall back to CPU.")
            self.providers = ["CPUExecutionProvider"]
        
        # Configure session options for memory optimization
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        self.session = ort.InferenceSession(model_path, providers=self.providers, sess_options=sess_options)
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info(f"Session Input names: {self.input_names}")
        logger.info(f"Session Output names: {self.output_names}")
        
    def __decode(self, audio_tokens:np.ndarray) -> Tuple[np.ndarray, int]:
        """Decode audio tokens into audio waveform.

        Args:
            audio_tokens (np.ndarray): Audio tokens of shape (batch_size, channels, audio_tokens_len).

        Returns:
            audio (np.ndarray): Audio waveform of shape (batch_size, audio_len).
            sample_rate (int): Sample rate of the audio waveform.
            
        """
        input_spec = {
            self.input_names[0]: audio_tokens
        }
        
        audio = self.session.run(self.output_names, input_spec)[0]
        return audio, 24000

    # TODO: Both support dynamic input shape and static input shape, when the input shape is static, the chunk_size should be the same as the audio_tokens_len.
    def decode_chunked(self, audio_tokens:np.ndarray, chunk_size:int=200) -> Tuple[np.ndarray, int]:
        """Decode audio tokens in chunks to manage memory."""
        batch_size, channels, seq_len = audio_tokens.shape
        
        if seq_len <= chunk_size:
            return self.__decode(audio_tokens)
        
        # Process in chunks
        chunks = []
        for i in range(0, seq_len, chunk_size):
            chunk = audio_tokens[:, :, i:i+chunk_size]
            chunk_audio, _ = self.__decode(chunk)
            chunks.append(chunk_audio)
        
        # Concatenate the results (this might need adjustment based on model behavior)
        full_audio = np.concatenate(chunks, axis=-1)  # This depends on the output shape
        return full_audio, 24000
    
test_audio_tokens = "./audio_tokens.npy"
model_path = "./models/moss_audio_tokenizer_decoder_onnx/audio_tokenizer_decoder.onnx"
output_path = "generated_audio.wav"

if __name__ == "__main__":
    audio_decoder = AudioTokenizerDecoder(model_path)
    audio_tokens:NDArray = np.load(test_audio_tokens)
    if audio_tokens.ndim != 3:
        audio_tokens = np.expand_dims(audio_tokens, axis=0).astype(np.int32)
    logger.info(f"Audio tokens shape: {audio_tokens.shape}")
    
    audio, sample_rate = audio_decoder.decode_chunked(audio_tokens)
    logger.info(f"Generated Audio shape: {audio.shape}")
    
    if audio.ndim == 2:
        audio = audio[0]
    
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Audio waveform saved to {output_path} with sample rate {sample_rate}")
```

##### Decode Audio Tokens Result

 - 生成的音频文件: [generated_audio.wav](https://github.com/201831771214/MossAudioTokenizer-ONNX/blob/main/generated_audio.wav)

 - Audio Text: "MOSS-TTS-Realtime 是一个上下文感知、多轮次流式 TTS 模型，专为实时语音智能体设计。通过结合文本对话历史和用户先前的声学特征，它能在多轮交互中提供低延迟、连贯一致的语音响应。"

#### Run Audio Tokenizer Encoder

```python
import onnxruntime as ort
import numpy as np
from numpy.typing import NDArray
import librosa
from typing import Tuple
import os
import sys

import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/run_encoder.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

class AudioTokenizerEncoder:
    def __init__(self, model_path:str, device:str="cuda"):
        self.model_path = model_path
        self.device = device.lower()
        
        logger.info(f"All available providers: {ort.get_available_providers()}")
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider"]
        elif device == "cpu" and "CPUExecutionProvider" in ort.get_available_providers():
            self.providers = ["CPUExecutionProvider"]
        else:
            logger.warning(f"Device {device} is not supported. Fall back to CPU.")
            self.providers = ["CPUExecutionProvider"]
        
        # Configure session options for memory optimization
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        self.session = ort.InferenceSession(model_path, providers=self.providers, sess_options=sess_options)
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info(f"Session Input names: {self.input_names}")
        logger.info(f"Session Output names: {self.output_names}")
        
    def encode(self, audio_data:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode audio waveform into audio tokens.

        Args:
            audio_data (np.ndarray): Audio waveform of shape (batch_size, channels, audio_len).

        Returns:
            audio_codes (np.ndarray): Audio tokens of shape (num_layers, batch_size, seq_len).
            audio_codes_lengths (np.ndarray): Audio tokens lengths of shape (seq_len,).
            encoder_hidden_states (np.ndarray): Encoder hidden states of shape (batch_size, d_model, seq_len).
            
        """
        input_spec = {
            self.input_names[0]: audio_data
        }
        
        audio_codes, audio_codes_lengths, encoder_hidden_states = self.session.run(self.output_names, input_spec)
        
        
        return audio_codes, audio_codes_lengths, encoder_hidden_states

test_audio = "./sources/audios/dubowen.wav"
model_path = "./models/moss_audio_tokenizer_decoder_onnx/audio_tokenizer_encoder.onnx"

if __name__ == "__main__":
    audio_encoder = AudioTokenizerEncoder(model_path)
    
    audio_data, sample_rate = librosa.load(test_audio, sr=24000, dtype=np.float32)
    logger.info(f"Audio data shape: {audio_data.shape}")
    
    # Add batch size and channel dimension if not present
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=0)
        audio_data = np.expand_dims(audio_data, axis=1)
    
    audio_codes, audio_codes_lengths, encoder_hidden_states = audio_encoder.encode(audio_data)
    logger.info(f"Audio codes shape: {audio_codes.shape} ---- {audio_codes}")
    logger.info(f"Audio codes lengths shape: {audio_codes_lengths.shape} ---- {audio_codes_lengths}")
    logger.info(f"Encoder hidden states shape: {encoder_hidden_states.shape} ---- {encoder_hidden_states}")
```

##### Audio Tokenizer Encoder Result

```shell
2026-02-28 14:30:43,491 - INFO - Audio codes shape: (32, 1, 69) ---- [[[ 817   57  132 ...  335  335  950]]

 [[ 470   18  632 ...  175  667  370]]

 [[ 224  363  968 ...  516  282  126]]

 ...

 [[ 125  958  275 ...  794  610  522]]

 [[ 963  217  460 ...  468  575  452]]

 [[1014  367  950 ...  968  688  369]]]
2026-02-28 14:30:43,491 - INFO - Audio codes lengths shape: (1,) ---- [68]
2026-02-28 14:30:43,492 - INFO - Encoder hidden states shape: (1, 768, 69) ---- [[[ 9.686964   -5.875668   -4.5082426  ...  4.122473    4.668351
    4.7193575 ]
  [-3.454163   -1.0035998   2.433176   ... -0.72330576 -1.1036214
   -1.0855117 ]
  [-2.9715493   2.7055      2.9065142  ... -1.2198896   3.042768
    2.9706757 ]
  ...
  [-6.3878407  -8.618917    8.772977   ...  4.81421     4.5244756
    4.5736403 ]
  [-4.635868    9.056893    6.7735367  ... -3.4137735  -4.1788063
   -4.2336264 ]
  [ 4.5891705  12.405132   -1.1379069  ... -2.1964636  -2.6125443
   -2.6521518 ]]]
```

### 加入我们

 - 公众号："CrazyNET"

### 关注CrazyNET公众号，获取更多关于MOSS-TTS-Realtime以及MossAudioTokenizer的信息和更新。
