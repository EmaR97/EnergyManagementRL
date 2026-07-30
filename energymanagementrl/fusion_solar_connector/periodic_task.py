import sys
import threading
import time
from logging import Logger

from utility import get_logger, logging


class PeriodicTask:
    def __init__(self, task_function, interval=60, logger: Logger=None):
        """
        Initializes the PeriodicTask object.

        :param task_function: The function to call periodically.
        :param interval: Time interval between calls (in seconds).
        """
        self.logger = logger or get_logger(self.__class__.__name__, logging.WARNING)
        self.task_function = task_function
        self.interval = interval
        self._stop_event = threading.Event()
        self._task_thread = None

    def _task_loop(self):
        try:
            while not self._stop_event.is_set():
                self.task_function()
                time.sleep(self.interval)
        except Exception as e:
            self.logger.exception("Unhandled exception in task: %s", e)
        finally:
            self._stop_event.set()
            self.logger.info("Task loop exited.")

    def start(self):
        if self._task_thread and self._task_thread.is_alive():
            self.logger.warning("Periodic task already running.")
            return
        self._stop_event.clear()
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True)
        self._task_thread.start()
        self.logger.info("Periodic task started.")

    def stop(self):
        if not self._task_thread or not self._task_thread.is_alive():
            self.logger.warning("No periodic task to stop.")
            return
        self._stop_event.set()
        self._task_thread.join()
        self.logger.info("Periodic task stopped.")


if __name__ == "__main__":
    def example_task():
        print("Task is running...")
        # Simulate a potential exception for testing purposes.
        # raise ValueError("Simulated exception in task.")


    periodic_task = PeriodicTask(example_task, interval=5)
    try:
        periodic_task.start()
        while True:
            time.sleep(1)  # Main program loop.
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e:
        print(f"Unhandled exception: {e}", file=sys.stderr)
    finally:
        periodic_task.stop()
        print("Program exited.")
