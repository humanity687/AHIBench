from sdk.base_agent import BaseAHIAgent


class LinShenAgent(BaseAHIAgent):
    """林深 - 自主数字生命，按自己的节奏思考和行动。

    基线对照：上下文策略 = 原 AHI 行为（LLMClient 消息累积 + 超 120 条时
    中间段摘要压缩）。压缩责任由 agent 类自己接管（决策 A5），每次调用前
    检查长度并压缩（compress_middle 幂等：未超阈值直接返回）。
    """

    def _call_llm(self, prompt: str) -> str | None:
        if self.llm_client is not None:
            try:
                self.llm_client.compress_middle()
            except Exception:
                pass
        return super()._call_llm(prompt)
