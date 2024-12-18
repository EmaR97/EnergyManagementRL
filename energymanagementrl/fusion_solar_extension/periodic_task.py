import threading
import time
import sys


class PeriodicTask:
    def __init__(self, task_function, interval=60):
        """
        Initializes the PeriodicTask object.

        :param task_function: The function to call periodically.
        :param interval: Time interval between calls (in seconds).
        """
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
            print(f"Unhandled exception in task: {e}", file=sys.stderr)
        finally:
            self._stop_event.set()  # Ensure the stop event is set to signal task completion.
            print("Task loop exited.")

    def start(self):
        if self._task_thread and self._task_thread.is_alive():
            print("Periodic task already running.")
            return
        self._stop_event.clear()
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True)
        self._task_thread.start()
        print("Periodic task started.")

    def stop(self):
        if not self._task_thread or not self._task_thread.is_alive():
            print("No periodic task to stop.")
            return
        self._stop_event.set()
        self._task_thread.join()  # Safely wait for the thread to finish.
        print("Periodic task stopped.")


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
