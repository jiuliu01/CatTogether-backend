"""Dependency-free conservative token estimation for prompt budgeting."""
from __future__ import annotations


class ApproximateTokenCounter:
    def count(self, text: str) -> int:
        if not text:
            return 0
        ascii_count = sum(1 for char in text if ord(char) < 128)
        other_count = len(text) - ascii_count
        return max(1, (ascii_count + 3) // 4 + other_count)

    def truncate(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        if self.count(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid]) <= budget:
                low = mid
            else:
                high = mid - 1
        return text[:low].rstrip() + ("…" if low < len(text) else "")

    def truncate_tail(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        if self.count(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[len(text) - mid:]) <= budget:
                low = mid
            else:
                high = mid - 1
        return ("…" if low < len(text) else "") + text[len(text) - low:].lstrip()


token_counter = ApproximateTokenCounter()
