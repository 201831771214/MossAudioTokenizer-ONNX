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
model_path = "./models/moss_tts/audio_tokenizer_decoder.onnx"
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