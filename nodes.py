import base64
import gc
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import folder_paths
import numpy as np
from PIL import Image

from .i18n import t
from .rag_core import (
    build_faiss_index,
    extract_answer_between_newlines,
    list_lmstudio_models,
    lmstudio_chat,
    load_single_document,
    search_index,
    unload_embedding_model,
    unload_lmstudio_model,
)


SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def _is_supported_doc_file(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_DOC_EXTENSIONS


def _list_input_docs_for_combo() -> List[str]:
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return [""]
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    docs = [f for f in files if _is_supported_doc_file(f)]
    docs = sorted(docs)
    return docs if docs else [""]


def _list_local_embedding_models() -> List[str]:
    # 仅从 ComfyUI/models/embeddings 读取本地向量模型
    model_paths: List[str] = []
    for emb_root in folder_paths.get_folder_paths("embeddings"):
        root = Path(emb_root)
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            # sentence-transformers 常见模型目录特征文件
            if (p / "config.json").exists() or (p / "modules.json").exists():
                model_paths.append(str(p.resolve()))

    model_paths = sorted(set(model_paths))
    return model_paths if model_paths else [""]


def _list_lmstudio_models_for_ui() -> List[str]:
    models = list_lmstudio_models("http://127.0.0.1:1234", timeout=2)
    return models if models else [""]


def _image_tensor_to_data_url(image) -> str:
    if image is None:
        return ""
    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


_LAST_MODEL_BY_BASE_URL: Dict[str, str] = {}


def _clear_vram_before_run(enabled: bool) -> Dict:
    if not enabled:
        return {"requested": False, "ok": True, "steps": []}

    steps: List[str] = []
    errors: List[str] = []

    try:
        gc.collect()
        steps.append(t("gc.collect"))
    except Exception as e:
        errors.append(t("gc.collect failed: {error}", error=e))

    try:
        import comfy.model_management as model_management  # type: ignore

        # 优先使用 ComfyUI 官方卸载路径，避免仅清 cache 但模型仍常驻显存。
        if hasattr(model_management, "unload_all_models"):
            model_management.unload_all_models()
            steps.append(t("comfy.model_management.unload_all_models"))

        if hasattr(model_management, "cleanup_models"):
            try:
                model_management.cleanup_models()
                steps.append(t("comfy.model_management.cleanup_models"))
            except TypeError:
                # 兼容部分版本 cleanup_models 需要参数。
                model_management.cleanup_models(True)
                steps.append(t("comfy.model_management.cleanup_models(True)"))

        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
            steps.append(t("comfy.model_management.soft_empty_cache"))
        elif hasattr(model_management, "empty_cache"):
            model_management.empty_cache()
            steps.append(t("comfy.model_management.empty_cache"))
    except Exception as e:
        errors.append(t("comfy model_management clear failed: {error}", error=e))

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            steps.append(t("torch.cuda.empty_cache"))
    except Exception as e:
        errors.append(t("torch cuda clear failed: {error}", error=e))

    return {
        "requested": True,
        "ok": len(errors) == 0,
        "steps": steps,
        "errors": errors,
    }


class DocumentLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "document": (
                    _list_input_docs_for_combo(),
                    {
                        "tooltip": t(
                            "选择文档（txt/json/md/pdf）。可用下方“上传文档”按钮上传到 input 目录后再选择。"
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("RAG_DOCUMENTS", "STRING")
    RETURN_NAMES = ("documents", "summary")
    FUNCTION = "load_documents"
    CATEGORY = "EasyRAG"

    @classmethod
    def VALIDATE_INPUTS(cls, document):
        if document:
            if not folder_paths.exists_annotated_filepath(document):
                return t("无效文档: {document}", document=document)
            if not _is_supported_doc_file(document):
                return t("不支持的文档类型: {document}", document=document)
        return True

    def load_documents(self, document: str):
        if not document:
            return ([], t("请在 document 中选择或上传一个文档（txt/json/md/pdf）。"))

        file_path = Path(folder_paths.get_annotated_filepath(document)).resolve()
        if not _is_supported_doc_file(str(file_path)):
            return (
                [],
                t(
                    "不支持的文档类型: {suffix}，仅支持 txt/json/md/pdf。",
                    suffix=file_path.suffix,
                ),
            )

        documents: List[Dict] = []
        errors: List[str] = []
        try:
            doc = load_single_document(file_path, encoding="utf-8")
            if doc.get("text"):
                documents.append(doc)
        except Exception as e:
            errors.append(f"{file_path}: {e}")

        summary = t(
            "文档加载完成。总文件: {total}, 成功: {success}, 失败: {failed}",
            total=1,
            success=len(documents),
            failed=len(errors),
        )
        if errors:
            summary += "\n" + "\n".join(errors[:5])
        return (documents, summary)


class VectorStoreBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "documents": ("RAG_DOCUMENTS",),
                "index_name": ("STRING", {"default": "default_index"}),
                "embedding_model": (
                    _list_local_embedding_models(),
                    {"tooltip": t("仅允许使用 ComfyUI/models/embeddings 下的本地模型")},
                ),
                "chunk_size": ("INT", {"default": 400, "min": 100, "max": 4000, "step": 10}),
                "chunk_overlap": ("INT", {"default": 80, "min": 0, "max": 2000, "step": 10}),
                "show_retrieval_log": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("日志开"), "label_off": t("日志关")},
                ),
            }
        }

    RETURN_TYPES = ("RAG_INDEX", "STRING")
    RETURN_NAMES = ("rag_index", "summary")
    FUNCTION = "build_vector_store"
    CATEGORY = "EasyRAG"

    def build_vector_store(
        self,
        documents: List[Dict],
        index_name: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        show_retrieval_log: bool,
    ):
        selected_model = str(embedding_model or "").strip()
        if not selected_model:
            raise ValueError(
                t(
                    "未检测到可用 embedding 模型。请先把 sentence-transformers 模型放到 ComfyUI/models/embeddings。"
                )
            )
        info = build_faiss_index(
            documents=documents,
            embedding_model=selected_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            index_name=index_name.strip(),
        )
        unload_info = unload_embedding_model(selected_model)
        info["show_retrieval_log"] = bool(show_retrieval_log)
        info["embedding_unload_info"] = unload_info
        summary = (
            t(
                "向量库构建完成: {index_name}, 文档数: {documents_count}, chunk数: {chunks_count}, 模型: {selected_model}, 目录: {index_dir}",
                index_name=info["index_name"],
                documents_count=info["documents_count"],
                chunks_count=info["chunks_count"],
                selected_model=selected_model,
                index_dir=info["index_dir"],
            )
        )
        return (info, summary)


class LMStudioRAGChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": t("模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）")},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": t("你是一个严谨的本地RAG助手，优先根据给定上下文回答。"),
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 16}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True},
                ),
                "stream": ("BOOLEAN", {"default": False, "label_on": t("流式开"), "label_off": t("流式关")}),
                "clear_vram_before_run": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("运行前清理"), "label_off": t("直接运行")},
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("卸载模型"), "label_off": t("保留模型")},
                ),
            },
            "optional": {
                "rag_index": ("RAG_INDEX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "context_used", "raw_response")
    FUNCTION = "chat_with_rag"
    CATEGORY = "EasyRAG"

    def chat_with_rag(
        self,
        question: str,
        base_url: str,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_k: int,
        seed: int,
        stream: bool,
        clear_vram_before_run: bool,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        vram_cleanup = _clear_vram_before_run(bool(clear_vram_before_run))
        if not vram_cleanup.get("ok", True):
            print(t("[EasyRAG][显存清理] 运行前清理失败: {errors}", errors=vram_cleanup.get("errors", [])))

        available_models = list_lmstudio_models(base, timeout=4)
        chosen_model = (model or "").strip()
        if (not chosen_model) and available_models:
            chosen_model = available_models[0]
        elif chosen_model and available_models and chosen_model not in available_models:
            # 若节点缓存了旧模型名，自动回退到当前可用模型
            chosen_model = available_models[0]

        # 自动模型切换卸载：同一 LM Studio 地址下，模型变化时先卸载上一个
        auto_unload_before_switch = None
        prev_model = _LAST_MODEL_BY_BASE_URL.get(base, "")
        if prev_model and chosen_model and prev_model != chosen_model:
            try:
                auto_unload_before_switch = unload_lmstudio_model(base, prev_model)
            except Exception as e:
                auto_unload_before_switch = {"ok": False, "error": str(e), "instance_id": prev_model}

        context_used = ""
        show_retrieval_log = bool((rag_index or {}).get("show_retrieval_log", False))
        embedding_unload_after_retrieval = None
        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
            result = search_index(index_ref, query=question, top_k=int(top_k), device="cpu")
            context_used = result["context"]
            if show_retrieval_log:
                print(t("[EasyRAG][问答检索] question={question!r} top_k={top_k}", question=question, top_k=top_k))
                print(
                    t(
                        "[EasyRAG][问答检索] rag_hit={rag_hit} best_score={best_score:.4f}",
                        rag_hit=result.get("rag_hit"),
                        best_score=result.get("best_score", 0.0),
                    )
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        t(
                            "[EasyRAG][问答检索#{i}] score={score:.4f} source={source} chunk_text={chunk_text}",
                            i=i,
                            score=item.get("score", 0.0),
                            source=item.get("source", ""),
                            chunk_text=snippet,
                        )
                    )

            retrieval_embedding_model = str((rag_index or {}).get("embedding_model", "")).strip()
            embedding_unload_after_retrieval = unload_embedding_model(
                retrieval_embedding_model if retrieval_embedding_model else None
            )
            if show_retrieval_log:
                print(
                    t(
                        "[EasyRAG][问答检索] embedding检索后卸载: count={count} ok={ok}",
                        count=int(embedding_unload_after_retrieval.get("count", 0)),
                        ok=bool(embedding_unload_after_retrieval.get("ok", True)),
                    )
                )
                unload_errors = embedding_unload_after_retrieval.get("errors") or []
                if unload_errors:
                    print(
                        t(
                            "[EasyRAG][问答检索] embedding检索后卸载错误: {errors}",
                            errors=unload_errors,
                        )
                    )

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=question,
            context=context_used,
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            seed=int(seed),
            api_mode="responses",
            stream=bool(stream),
            emit_stream_log=bool(stream or show_retrieval_log),
        )
        _LAST_MODEL_BY_BASE_URL[base] = chosen_model or response.get("model", "")
        unload_status = None
        selected_model = response.get("model") or chosen_model
        if unload_model_after_response and selected_model:
            try:
                unload_status = unload_lmstudio_model(base, selected_model)
                if _LAST_MODEL_BY_BASE_URL.get(base) == selected_model:
                    _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except Exception as e:
                unload_status = {"ok": False, "error": str(e)}
        extracted_answer = extract_answer_between_newlines(response["answer"])
        raw_out = {
            "chat": response["raw"],
            "selected_model": selected_model,
            "available_models": available_models,
            "vram_cleanup_before_run": vram_cleanup,
            "embedding_unload_after_retrieval": embedding_unload_after_retrieval,
            "auto_unload_before_switch": auto_unload_before_switch,
            "seed": int(seed),
            "unload_requested": bool(unload_model_after_response),
            "unload_status": unload_status,
        }
        return (
            extracted_answer,
            context_used,
            json.dumps(raw_out, ensure_ascii=False, indent=2),
        )


class LMStudioRAGChatSimpleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True, "default": ""}),
                "base_url": ("STRING", {"default": "http://127.0.0.1:1234"}),
                "model": (
                    _list_lmstudio_models_for_ui(),
                    {"tooltip": t("模型选项来自 LM Studio API（默认地址: 127.0.0.1:1234）")},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": t("你是一个严谨的本地RAG助手，优先根据给定上下文回答。"),
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True},
                ),
                "clear_vram_before_run": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("运行前清理"), "label_off": t("直接运行")},
                ),
                "unload_model_after_response": (
                    "BOOLEAN",
                    {"default": True, "label_on": t("卸载模型"), "label_off": t("保留模型")},
                ),
            },
            "optional": {
                "rag_index": ("RAG_INDEX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)
    FUNCTION = "chat_simple"
    CATEGORY = "EasyRAG"

    def chat_simple(
        self,
        question: str,
        base_url: str,
        model: str,
        system_prompt: str,
        seed: int,
        clear_vram_before_run: bool,
        unload_model_after_response: bool,
        rag_index=None,
        image=None,
    ):
        base = base_url.strip()
        vram_cleanup = _clear_vram_before_run(bool(clear_vram_before_run))
        if not vram_cleanup.get("ok", True):
            print(t("[EasyRAG][显存清理] 运行前清理失败: {errors}", errors=vram_cleanup.get("errors", [])))

        available_models = list_lmstudio_models(base, timeout=4)
        chosen_model = (model or "").strip()
        if (not chosen_model) and available_models:
            chosen_model = available_models[0]
        elif chosen_model and available_models and chosen_model not in available_models:
            chosen_model = available_models[0]

        prev_model = _LAST_MODEL_BY_BASE_URL.get(base, "")
        if prev_model and chosen_model and prev_model != chosen_model:
            try:
                unload_lmstudio_model(base, prev_model)
            except Exception:
                pass

        context_used = ""
        show_retrieval_log = bool((rag_index or {}).get("show_retrieval_log", False))
        if rag_index is not None:
            index_ref = rag_index.get("index_dir") or rag_index.get("index_name")
            result = search_index(index_ref, query=question, device="cpu")
            context_used = result["context"]
            if show_retrieval_log:
                print(t("[EasyRAG][简约检索] question={question!r} top_k=5(default)", question=question))
                print(
                    t(
                        "[EasyRAG][简约检索] rag_hit={rag_hit} best_score={best_score:.4f}",
                        rag_hit=result.get("rag_hit"),
                        best_score=result.get("best_score", 0.0),
                    )
                )
                for i, item in enumerate(result.get("items", []), start=1):
                    snippet = (item.get("text", "") or "").replace("\n", " ")[:120]
                    print(
                        t(
                            "[EasyRAG][简约检索#{i}] score={score:.4f} source={source} chunk_text={chunk_text}",
                            i=i,
                            score=item.get("score", 0.0),
                            source=item.get("source", ""),
                            chunk_text=snippet,
                        )
                    )
            retrieval_embedding_model = str((rag_index or {}).get("embedding_model", "")).strip()
            embedding_unload_after_retrieval = unload_embedding_model(
                retrieval_embedding_model if retrieval_embedding_model else None
            )
            if show_retrieval_log:
                print(
                    t(
                        "[EasyRAG][简约检索] embedding检索后卸载: count={count} ok={ok}",
                        count=int(embedding_unload_after_retrieval.get("count", 0)),
                        ok=bool(embedding_unload_after_retrieval.get("ok", True)),
                    )
                )
                unload_errors = embedding_unload_after_retrieval.get("errors") or []
                if unload_errors:
                    print(
                        t(
                            "[EasyRAG][简约检索] embedding检索后卸载错误: {errors}",
                            errors=unload_errors,
                        )
                    )
        elif show_retrieval_log:
            print(t("[EasyRAG][简约检索] 未连接 rag_index，跳过检索。"))

        image_data_url = _image_tensor_to_data_url(image) if image is not None else ""
        response = lmstudio_chat(
            base_url=base,
            model=chosen_model,
            question=question,
            context=context_used,
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            seed=int(seed),
            api_mode="responses",
            stream=True,
            emit_stream_log=True,
        )
        _LAST_MODEL_BY_BASE_URL[base] = chosen_model or response.get("model", "")
        selected_model = response.get("model") or chosen_model
        if unload_model_after_response and selected_model:
            try:
                unload_lmstudio_model(base, selected_model)
                if _LAST_MODEL_BY_BASE_URL.get(base) == selected_model:
                    _LAST_MODEL_BY_BASE_URL.pop(base, None)
            except Exception:
                pass

        extracted_answer = extract_answer_between_newlines(response["answer"])
        return (extracted_answer,)

NODE_CLASS_MAPPINGS = {
    "EasyRAGDocumentLoader": DocumentLoaderNode,
    "EasyRAGVectorStoreBuilder": VectorStoreBuilderNode,
    "EasyRAGLMStudioChatAdvanced": LMStudioRAGChatNode,
    "EasyRAGLMStudioChatSimple": LMStudioRAGChatSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyRAGDocumentLoader": "EasyRAG - Document Loader",
    "EasyRAGVectorStoreBuilder": "EasyRAG - Vector Store Builder (FAISS)",
    "EasyRAGLMStudioChatAdvanced": "EasyRAG - LM Studio API (Advanced)",
    "EasyRAGLMStudioChatSimple": "EasyRAG - LM Studio API (Simple)",
}
