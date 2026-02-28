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
    logger.info(f"Audio codes shape: {audio_codes.shape}")
    logger.info(f"Audio codes lengths shape: {audio_codes_lengths.shape}")
    logger.info(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
