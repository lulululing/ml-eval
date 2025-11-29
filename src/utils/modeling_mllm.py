import os
from typing import Optional, Tuple

BACKEND_HF = "hf"
BACKEND_LLAVA_CPP = "llava_cpp"


def get_mllm(backend: str,
             model_name: Optional[str] = None,
             gguf_path: Optional[str] = None):
    """Return a callable generate(image, question: str, options: Tuple[str,str,str,str])->str (full text)"""
    if backend == BACKEND_HF:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name, device_map='auto')

        def _generate(image, question: str, options: Tuple[str, str, str, str], max_tokens: int = 256, temperature: float = 0.2) -> str:
            prompt = f"Question: {question}\nOptions: A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]}\nAnswer:"
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
            gen = model.generate(**inputs, do_sample=temperature > 0, temperature=temperature, max_new_tokens=max_tokens)
            out = processor.batch_decode(gen, skip_special_tokens=True)[0]
            return out
        return _generate

    elif backend == BACKEND_LLAVA_CPP:
        # 占位：llava.cpp的Python封装不稳定，建议使用独立可执行并通过管道/HTTP调用。
        # 这里先抛出异常并在README提供使用说明。
        raise NotImplementedError("请使用llava.cpp可执行程序进行推理，或切换到HF后端（需GPU）。")
    else:
        raise ValueError(f"Unknown backend: {backend}")
