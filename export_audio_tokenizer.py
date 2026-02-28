from moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from moss_extra.audio_tokenizer_decoder import AudioTokenizerDecoderWrapper
import torch
from typing import Tuple
import argparse
import os
import onnx
from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
import sys
import traceback

EXTERNAL_DATA_THRESHOLD = 4 * 1024 * 1024 * 1024 # 4GB

def build_audio_tokenizer_decoder_dummy_input(batch_size:int=1, channels:int=16, sequence_length:int=237) -> Tuple[torch.Tensor]:
    audio_codes = torch.randint(0, 1000, (batch_size, channels, sequence_length), dtype=torch.int32)
    return (audio_codes, )

def export_audio_tokenizer_decoder(
    audio_tokenizer:MossAudioTokenizerModel,
    output_path:str,
    opset_version:int=14,
    dummy_input:Tuple[torch.Tensor]=build_audio_tokenizer_decoder_dummy_input(),
    is_dynamic:bool=True
):
    audio_tokenizer_decoder = AudioTokenizerDecoderWrapper(audio_tokenizer).eval()
    
    input_names = ["audio_codes"]
    output_names = ["audio"]

    if is_dynamic:
        dynamic_axes = {
            "audio_codes": {0: "batch_size", 1: "channels", 2: "sequence_length"},
        }
    else:
        dynamic_axes = {}
    
    try:
        os.makedirs(os.path.join(output_path, "tokenizer"), exist_ok=True)
        save_root = os.path.join(output_path, "tokenizer")
        model_save_path = os.path.join(output_path, "tokenizer", "audio_tokenizer_decoder.onnx")
        torch.onnx.export(
            model=audio_tokenizer_decoder,
            args=dummy_input,
            f=model_save_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            external_data=True,
            do_constant_folding=True,
            export_modules_as_functions=False
        )
        
        # combine external data into main onnx file for easier loading in some runtimes (optional, can keep as external if preferred)
        model_proto = onnx.load(model_save_path)
        
        load_external_data_for_model(model_proto, save_root)
        convert_model_to_external_data(
            model_proto,
            all_tensors_to_one_file=True,
            convert_attribute=True
        )
        onnx.save(model_proto, os.path.join(output_path, "audio_tokenizer_decoder.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=EXTERNAL_DATA_THRESHOLD)
        
        print(f"Audio Tokenizer Decoder exported successfully to {os.path.join(output_path, 'audio_tokenizer_decoder.onnx')}")
        
    except Exception as e:
        print(f"Export audio tokenizer decoder failed: {e}")
        raise e

default_model_path = "./models/Moss-Audio-Tokenizer/"
default_output_path = "./models/moss_tts/"

if __name__ == "__main__":
    msg_info = f"Export audio tokenizer decoder to {default_output_path} by default."
    usg_info = """
    Usage:
        python export_audio_tokenizer.py [-m MODEL_PATH] [-o OUTPUT_PATH] [--opset OPSET] [-id] [-b BATCH_SIZE] [-c CHANNELS] [-s SEQUENCE_LENGTH] [-d CHUNK_DURATION]
    """
    
    parser = argparse.ArgumentParser(usage=usg_info, description=msg_info)
    parser.add_argument("-m", "--model_path", type=str, default=default_model_path, help="Path to the audio tokenizer model.")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path, help="Path to save the exported ONNX model.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version.")
    parser.add_argument("-id", "--is_dynamic", action="store_true", help="Whether to export the model with dynamic axes.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for dummy input.")
    parser.add_argument("-c", "--channels", type=int, default=16, help="Channels for dummy input.")
    parser.add_argument("-s", "--sequence_length", type=int, default=237, help="Sequence length for dummy input.")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    audio_tokenizer = MossAudioTokenizerModel.from_pretrained(args.model_path, trust_remote_code=True)
    
    audio_tokenizer.eval().cpu()
    
    dummy_inputs = build_audio_tokenizer_decoder_dummy_input(
        batch_size=args.batch_size,
        channels=args.channels,
        sequence_length=args.sequence_length,
    )
    
    export_audio_tokenizer_decoder(
        audio_tokenizer=audio_tokenizer,
        output_path=args.output_path,
        opset_version=args.opset,
        dummy_input=dummy_inputs,
        is_dynamic=args.is_dynamic
    )