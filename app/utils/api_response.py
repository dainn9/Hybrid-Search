import time

def now_ts():
    return int(time.time())

def make_api_response(status: int,
                      message: str,
                      query: str = None,
                      results=None,
                      extra: dict = None):
    """
    Chuẩn hóa format JSON trả về cho API.
    Dùng chung cho success và error.
    """
    response = {
        "status": status,
        "message": message,
        "query": query,
        "timestamp": int(time.time()),
        "results": results or []
    }

    if extra:
        response.update(extra)

    return response