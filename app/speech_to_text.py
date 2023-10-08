from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 

    WhisperProcessor,
    WhisperForConditionalGeneration,
)

import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO, format='[STT_MODULE]%(asctime)s %(levelname)s %(message)s')

def __feature_extractor(
        model: torch.nn.Module = None,
        processor: torch.nn.Module = None,
        audio: np.ndarray = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
    ):
    extracted_features = None

    if verbose: logging.info("% [PROCESSOR] Converting audio...")
    input_features = processor(
        audio,
        sampling_rate=audio_sample_rate,
        return_tensors="pt"
    ).input_features.to(device)
    if verbose: logging.info("% [PROCESSOR] Done.")

    if verbose: logging.info("Inference...")
    for param in model.parameters():    param.requires_grad = False

    for batch in input_features.split(32):
        with torch.no_grad():
            predicted_ids = model.generate(batch)
        if extracted_features is None: extracted_features = predicted_ids
        else: extracted_features = torch.cat((extracted_features, predicted_ids), dim=0)

    if verbose: logging.info("Decoding...")
    transcription = processor.batch_decode(extracted_features, skip_special_tokens=True)

    return transcription

def speech_to_text_whisper(
        pretrained_model_name_or_path: str = None,
        audio: np.ndarray = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
        batchsize: int = None,
    ):
    if pretrained_model_name_or_path is None: raise ValueError(f"pretrained_model_name_or_path argument is required. Excepted: str, but got {pretrained_model_name_or_path}")
    if audio is None: raise ValueError(f"audio argument is required. Excepted: np.ndarray, but got {audio}")
    if audio_sample_rate is None:
        audio_sample_rate = 16_000
        logging.info(f"audio_sample_rate argument is not provided. Using default value: {audio_sample_rate}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device argument is not provided. Using default value: {device}")
    if batchsize is None:
        batchsize = 32
        logging.info(f"batchsize argument is not provided. Using default value: {batchsize}")
    
    if verbose: logging.info("Loading model...")
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path).to(device)

    return __feature_extractor(
        model=model,
        processor=processor,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
        device=device,
        verbose=verbose,
    )

def speech_to_text(
        processor_pretrained_argument: str = None,
        audio: torch.Tensor = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
    ):
    if processor_pretrained_argument is None: raise ValueError(f"processor_pretrained_argument argument is required. Excepted: str, but got {processor_pretrained_argument}")
    if audio is None: raise ValueError(f"audio argument is required. Excepted: torch.Tensor, but got {audio}")
    if audio_sample_rate is None:
        audio_sample_rate = 16_000
        logging.info(f"audio_sample_rate argument is not provided. Using default value: {audio_sample_rate}")
    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device argument is not provided. Using default value: {device}")

    if verbose: logging.info("Loading model...")
    processor = Wav2Vec2Processor.from_pretrained(processor_pretrained_argument)
    model = Wav2Vec2ForCTC.from_pretrained(processor_pretrained_argument).to(device)

    if verbose: logging.info("Converting audio...")
    input_values = processor(
        audio,
        sampling_rate=audio_sample_rate,
        return_tensors="pt"
    ).input_values.to(device)
    input_values = input_values.reshape(1, -1)

    if verbose: logging.info("Inference...")
    with torch.no_grad():
        logits = model(input_values).logits

    if verbose: logging.info("Decoding...")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription
    