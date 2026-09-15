from typing import List
from openai import OpenAI


class TextCompressor:
    """Compresses long text by chunking and summarizing via LLM."""

    def __init__(self, model_name: str = None, api_base: str = None, api_key: str = None,
                 chunk_size: int = 20000, chunk_overlap: int = 500):
        self.model_name = model_name or "llama2"
        self.api_key = api_key or "ollama"
        self.api_base = api_base or "http://localhost:11434/v1"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text) // 3

    def split_text(self, text: str) -> List[str]:
        """Simple paragraph-based text splitting."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if self.count_tokens(current_chunk + para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks or [text]

    def compress_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        prompt = (
            f"请将以下文本（第{chunk_index+1}/{total_chunks}块）压缩成详细摘要，"
            f"保留所有关键信息，忽略冗余内容：\n\n---\n{chunk}\n---"
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=16384,
        )
        return response.choices[0].message.content or ""

    def aggregate(self, compressed_chunks: List[str], original_query: str) -> str:
        combined = "\n\n".join(f"--- 块{i+1} ---\n{c}" for i, c in enumerate(compressed_chunks))
        prompt = f"基于以下摘要回答用户问题：\n\n问题：{original_query}\n\n摘要：{combined}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""

    def process_long_text(self, long_text: str, user_query: str,
                          max_context: int = 262144) -> tuple:
        total = self.count_tokens(long_text) + self.count_tokens(user_query) + 500
        if total <= max_context:
            return long_text, False
        chunks = self.split_text(long_text)
        compressed = [self.compress_chunk(c, i, len(chunks)) for i, c in enumerate(chunks)]
        return self.aggregate(compressed, user_query), True
