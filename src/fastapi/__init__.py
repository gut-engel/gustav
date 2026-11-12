class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = "", headers: dict | None = None):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}
        super().__init__(f"HTTP {status_code}: {detail}")
