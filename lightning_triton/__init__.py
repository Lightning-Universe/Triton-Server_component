import functools
import signal
import threading


class SignalHandlerComposer:

    def __init__(self, signum):
        self._signum = signum
        self._handlers = []

    def __repr__(self):
        return f"<SignalHandlerComposer {self._signum}>"

    def add_handler(self, handler):
        self._handlers.append(handler)

    def __call__(self, signum, frame):
        call_hashes = set()
        for handler in reversed(self._handlers):

            if isinstance(handler, functools.partial):
                call_hash = handler.keywords.get("call_hash")
                if call_hash in call_hashes:
                    continue

            handler(signum, frame)


@functools.wraps(signal.signal)
def custom_signal_handler(_original_signal_function):
    composer_dict = {
        signal.SIGTERM: SignalHandlerComposer(signal.SIGTERM),
    }

    for signum, handler in composer_dict.items():
        _original_signal_function(signum, handler)

    def wrapper(signum, handler):
        # TODO - multiple handlers are registered by the uvloop/asyncio
        if signum in composer_dict:
            composer_dict[signum].add_handler(handler)
        else:
            _original_signal_function(signum, handler)

    return wrapper


if threading.currentThread() == threading.main_thread():
    signal.signal = custom_signal_handler(signal.signal)



from lightning_triton.lightning_triton import TritonServer

__all__ = ["TritonServer"]
