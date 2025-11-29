import os
from typing import Optional

BACKEND_HF = "hf"
BACKEND_LLAMA_CPP = "llama_cpp"


def get_llm(backend: str,
            model_name: Optional[str] = None,
            gguf_path: Optional[str] = None,
            quantization: Optional[str] = None):
    """Return a callable generate(prompt: str, max_tokens: int, temperature: float)->str"""
    if backend == BACKEND_HF:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        load_kwargs = {}
        if quantization == '4bit' and device == 'cuda':
            load_kwargs.update(dict(
                device_map='auto',
                load_in_4bit=True,
            ))
        else:
            load_kwargs.update(dict(device_map='auto'))
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        def _generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
            import torch
            inputs = tok(prompt, return_tensors='pt').to(model.device)
            gen = model.generate(**inputs, do_sample=temperature > 0,
                                 temperature=temperature,
                                 max_new_tokens=max_tokens)
            text = tok.decode(gen[0], skip_special_tokens=True)
            return text
        return _generate

    elif backend == BACKEND_LLAMA_CPP:
        from llama_cpp import Llama
        if not gguf_path or not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
        llm = Llama(model_path=gguf_path, n_ctx=4096)

        def _generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
            out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
            return out["choices"][0]["text"]
        return _generate

    else:
        raise ValueError(f"Unknown backend: {backend}")
