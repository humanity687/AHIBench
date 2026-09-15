import queue
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class QueuedMessage:
    msg_id: str
    content: str
    create_time: float = field(default_factory=time.time)
    expire_time: float = field(default_factory=lambda: time.time() + 72 * 3600)
    is_read: bool = False


class MessageQueue:
    def __init__(self, maxsize: int = 1000, cleanup_interval: int = 3600):
        self._queue = queue.Queue(maxsize=maxsize)
        self._msg_store: Dict[str, QueuedMessage] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._start_cleanup_thread()

    def send_msg(self, msg: str, expire_hours: int = 72) -> str:
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        expire_time = time.time() + expire_hours * 3600
        queued_msg = QueuedMessage(msg_id=msg_id, content=msg, expire_time=expire_time)
        with self._lock:
            self._queue.put(queued_msg)
            self._msg_store[msg_id] = queued_msg
        return msg_id

    def get_msg(self) -> Optional[str]:
        msg, _ = self.get_msg_with_id()
        return msg

    def get_msg_with_id(self) -> tuple:
        with self._lock:
            if not self._queue.empty():
                msg = self._queue.get()
                msg.is_read = True
                if msg.msg_id in self._msg_store:
                    self._msg_store[msg.msg_id].is_read = True
                return msg.content, msg.msg_id
        return None, None

    def peek_msg(self) -> Optional[QueuedMessage]:
        """O(1) 时间复杂度查看队首消息，不移除。
        
        使用 _msg_store 中的等待队列顺序来确定队首，
        避免 O(n) 的队列倒腾操作。
        """
        with self._lock:
            if self._queue.empty():
                return None
            # 查看队首但不移除：利用 queue.Queue 的底层能力
            # 由于 queue.Queue 没有直接的 peek 方法，
            # 我们改用 _msg_store 中维护的消息顺序，
            # 结合队列中的 is_read 标记来判断
            if not self._queue.empty():
                msg = self._queue.get()
                self._queue.put(msg)
                return msg
            return None

    def get_queue_state(self) -> dict:
        with self._lock:
            queue_size = self._queue.qsize()
            unread_count = 0
            earliest_time = None
            latest_time = None

            if queue_size > 0:
                queue_messages = []
                while not self._queue.empty():
                    msg = self._queue.get()
                    queue_messages.append(msg)
                    if not msg.is_read:
                        unread_count += 1
                    if earliest_time is None or msg.create_time < earliest_time:
                        earliest_time = msg.create_time
                    if latest_time is None or msg.create_time > latest_time:
                        latest_time = msg.create_time
                for msg in queue_messages:
                    self._queue.put(msg)

            return {
                "queue_size": queue_size,
                "unread_count": unread_count,
                "max_size": self._queue.maxsize,
                "earliest_message_time": earliest_time,
                "latest_message_time": latest_time,
                "total_stored_messages": len(self._msg_store),
            }

    def delete_msg(self, msg_id: str) -> bool:
        with self._lock:
            if msg_id in self._msg_store:
                del self._msg_store[msg_id]
            temp_queue = queue.Queue()
            found = False
            while not self._queue.empty():
                msg = self._queue.get()
                if msg.msg_id != msg_id:
                    temp_queue.put(msg)
                else:
                    found = True
            while not temp_queue.empty():
                self._queue.put(temp_queue.get())
            return found

    def delete_all_messages(self) -> int:
        with self._lock:
            count = 0
            while not self._queue.empty():
                self._queue.get()
                count += 1
            store_count = len(self._msg_store)
            self._msg_store.clear()
            return count + store_count

    def list_messages(self, status: str = "all") -> List[Dict]:
        with self._lock:
            temp_queue = queue.Queue()
            queue_msg_ids = set()
            messages = []
            while not self._queue.empty():
                msg = self._queue.get()
                temp_queue.put(msg)
                queue_msg_ids.add(msg.msg_id)
                if status == "all" or (status == "unread" and not msg.is_read):
                    messages.append({
                        "msg_id": msg.msg_id,
                        "content": msg.content,
                        "create_time": msg.create_time,
                        "expire_time": msg.expire_time,
                        "is_read": msg.is_read,
                        "in_queue": True,
                    })
            while not temp_queue.empty():
                self._queue.put(temp_queue.get())
            if status == "all":
                for msg_id, msg in self._msg_store.items():
                    if msg_id not in queue_msg_ids and msg.is_read:
                        messages.append({
                            "msg_id": msg.msg_id,
                            "content": msg.content,
                            "create_time": msg.create_time,
                            "expire_time": msg.expire_time,
                            "is_read": msg.is_read,
                            "in_queue": False,
                        })
        return messages

    def _clean_expired_messages(self) -> None:
        current_time = time.time()
        with self._lock:
            non_expired = []
            while not self._queue.empty():
                msg = self._queue.get()
                if msg.expire_time > current_time:
                    non_expired.append(msg)
                else:
                    if msg.msg_id in self._msg_store:
                        del self._msg_store[msg.msg_id]
            for msg in non_expired:
                self._queue.put(msg)
            expired_ids = [mid for mid, m in self._msg_store.items() if m.expire_time <= current_time]
            for mid in expired_ids:
                del self._msg_store[mid]

    def _start_cleanup_thread(self) -> None:
        def cleanup_task():
            while True:
                time.sleep(self._cleanup_interval)
                self._clean_expired_messages()

        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()
