from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
import torch
import torch.nn as nn

class AudioTokenizerDecoderWrapper(nn.Module):
    def __init__(self, audio_tokenizer:MossAudioTokenizerModel) -> None:
        super().__init__()
        self.audio_tokenizer = audio_tokenizer
    
    def forward(self, 
                audio_codes:torch.Tensor,
                ) -> torch.Tensor:
        """
        Args:
         - audio_codes: torch.Tensor of shape (num_quantizers, batch_size, sequence_length)
         - chunk_duration: float, duration of each chunk in seconds
        
        """
        
        decode_res = self.audio_tokenizer.decode(
            audio_codes=audio_codes, 
            padding_mask=None,
            return_dict=None,
            chunk_duration=None,
            num_quantizers=None)
        
        audio = decode_res["audio"][0].cpu().detach()
        
        return audio