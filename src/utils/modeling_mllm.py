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
        import torch

        # 检测是否是Qwen2-VL模型
        is_qwen2_vl = model_name and (('qwen2-vl' in model_name.lower()) or ('qwen2vl' in model_name.lower()))

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        if is_qwen2_vl:
            # 使用 Qwen2-VL 优化的生成，确保图像 tokens 与特征对齐
            def _generate_qwen2vl(image, question: str, options: Tuple[str, str, str, str], max_tokens: int = 256, temperature: float = 0.2) -> str:
                try:
                    from qwen_vl_utils import process_vision_info

                    prompt = (
                        f"Question: {question}\n"
                        f"Options:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\n"
                        f"Analyze the image and choose the correct option. Output only the letter (A/B/C/D)."
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(model.device)
                except Exception:
                    # fallback：无 qwen_vl_utils 时，使用简化 prompt 并显式 <image>
                    prompt = (
                        f"<|im_start|>user\n<image>\n"
                        f"Question: {question}\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\n"
                        f"Choose the correct option (A/B/C/D).<|im_end|>\n<|im_start|>assistant\n"
                    )
                    inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)

                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else 1.0,
                        pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else 0
                    )

                # 解码（尝试只取新生成部分）
                try:
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
                    out = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                except Exception:
                    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
                return out

            return _generate_qwen2vl
        else:
            # 通用 Vision2Seq 路径（如 LLaVA 等）
            def _generate(image, question: str, options: Tuple[str, str, str, str], max_tokens: int = 256, temperature: float = 0.2) -> str:
                prompt = (
                    f"Question: {question}\n"
                    f"Options:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n\n"
                    f"Answer with the correct option letter (A/B/C/D):"
                )
                inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else 1.0,
                        max_new_tokens=max_tokens
                    )
                out = processor.batch_decode(gen, skip_special_tokens=True)[0]
                return out
            return _generate

    elif backend == BACKEND_LLAVA_CPP:
        # 占位：llava.cpp的Python封装不稳定，建议使用独立可执行并通过管道/HTTP调用。
        # 这里先抛出异常并在README提供使用说明。
        raise NotImplementedError("请使用llava.cpp可执行程序进行推理，或切换到HF后端（需GPU）。")
    else:
        raise ValueError(f"Unknown backend: {backend}")
