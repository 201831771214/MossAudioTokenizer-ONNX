from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
import torch
import torch.nn as nn
from typing import Tuple

class AudioTokenizerEncoderWrapper(nn.Module):
    def __init__(self, audio_tokenizer:MossAudioTokenizerModel) -> None:
        super().__init__()
        self.audio_tokenizer = audio_tokenizer
    
    def forward(self, 
                input_values:torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
         - input_values: torch.Tensor of shape (batch_size, channels, sequence_length)
        
        Returns:
         - audio_codes: torch.Tensor of shape (num_layers, batch_size, sequence_length) -> shape: torch.Size([32, 1, 68])
         - audio_codes_lengths: torch.Tensor of shape (sequence_length,) -> shape: torch.Size([1])
         - encoder_hidden_states: torch.Tensor of shape (batch_size, d_model, sequence_length)
        
        """
        
        encoder_res = self.audio_tokenizer.encode(
            input_values=input_values, 
            padding_mask=None,
            num_quantizers=None,
            return_dict=None,
            chunk_duration=None)
        
        audio_codes = encoder_res.audio_codes
        audio_codes_lengths = encoder_res.audio_codes_lengths
        encoder_hidden_states = encoder_res.encoder_hidden_states
        
        return audio_codes, audio_codes_lengths, encoder_hidden_states