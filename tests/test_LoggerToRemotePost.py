import logging
import sys
import time
from typing import Optional

# Add the utility to path if needed
# sys.path.append("../.")

# Assuming the logger utility is in energymanagementrl.utility
# If not, import directly from the file
try:
    from energymanagementrl.utility.logger_to_google_app_script import LoggerToRemotePost
except ImportError:
    # Fallback for testing standalone
    from logger_to_remote_post import LoggerToRemotePost


def test_remote_logger(webhook_url: str, sheet_key: Optional[str] = None):
    """
    Test the remote POST logger with various log levels and scenarios.

    Args:
        webhook_url: Your Google Apps Script webhook URL
        sheet_key: Optional sheet key for routing logs
    """

    # Create a test logger
    test_logger = logging.getLogger("TestRemoteLogger")
    test_logger.setLevel(logging.DEBUG)

    # Add a console handler for local visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    test_logger.addHandler(console_handler)

    print("=" * 80)
    print("TESTING REMOTE POST LOGGER")
    print(f"Webhook URL: {webhook_url}")
    print(f"Sheet Key: {sheet_key or 'default'}")
    print("=" * 80)

    # Test 1: Basic logging with context manager
    print("\n[Test 1] Basic logging with context manager")
    with LoggerToRemotePost(test_logger, webhook_url, sheet_key) as log:
        test_logger.info("This is an INFO message")
        test_logger.warning("This is a WARNING message")
        test_logger.error("This is an ERROR message")
        test_logger.debug("This DEBUG message should not be sent (level INFO)")

    print("✓ Test 1 completed\n")
    time.sleep(1)

    # Test 2: Logging with different sheet keys
    print("[Test 2] Logging with different sheet keys")
    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="test_sheet_1") as log:
        test_logger.info("Message for test_sheet_1")

    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="test_sheet_2") as log:
        test_logger.warning("Message for test_sheet_2")

    print("✓ Test 2 completed\n")
    time.sleep(1)

    # Test 3: Multiple log entries in batch
    print("[Test 3] Multiple log entries in batch")
    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="batch_test") as log:
        for i in range(5):
            test_logger.info(f"Batch log entry #{i + 1} with some data: value={i * 10}")
            time.sleep(0.1)

    print("✓ Test 3 completed\n")
    time.sleep(1)

    # Test 4: Error handling (simulate)
    print("[Test 4] Logging with exception simulation")
    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="error_test") as log:
        try:
            test_logger.info("Attempting risky operation...")
            # Simulate an operation that might fail
            x = 1 / 0
        except Exception as e:
            test_logger.error(f"Caught exception: {type(e).__name__}: {str(e)}", exc_info=True)

    print("✓ Test 4 completed\n")
    time.sleep(1)

    # Test 5: Custom log formatting
    print("[Test 5] Testing custom log formatting")
    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="format_test") as log:
        test_logger.info(f"User action: login - timestamp: {time.time()}")
        test_logger.warning(f"System metric - CPU: 85%, Memory: 4.2GB, Disk: 67%")
        test_logger.error(f"API call failed: endpoint=/api/data, status=500, duration=2.3s")

    print("✓ Test 5 completed\n")

    # Test 6: Verify handler cleanup
    print("[Test 6] Testing handler cleanup")
    initial_handler_count = len(test_logger.handlers)
    print(f"Initial handlers: {initial_handler_count}")

    with LoggerToRemotePost(test_logger, webhook_url, sheet_key="cleanup_test") as log:
        test_logger.info("Inside context manager")
        handlers_inside = len(test_logger.handlers)
        print(f"Handlers inside context: {handlers_inside}")

    final_handler_count = len(test_logger.handlers)
    print(f"Final handlers: {final_handler_count}")

    if final_handler_count == initial_handler_count:
        print("✓ Handler cleanup successful")
    else:
        print("✗ Handler cleanup failed - handlers not properly removed")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("Check your Google Sheets to verify the logs were received")
    print("=" * 80)


def test_remote_logger_without_context_manager(webhook_url: str):
    """
    Alternative test using handler directly without context manager.

    Args:
        webhook_url: Your Google Apps Script webhook URL
    """
    from logger_to_remote_post import RemotePostHandler

    print("\n" + "=" * 80)
    print("TESTING WITHOUT CONTEXT MANAGER")
    print("=" * 80)

    logger = logging.getLogger("DirectHandlerTest")
    logger.setLevel(logging.DEBUG)

    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console)

    # Add remote handler
    remote_handler = RemotePostHandler(webhook_url, sheet_key="direct_test")
    remote_handler.setLevel(logging.INFO)
    logger.addHandler(remote_handler)

    print("\nSending logs...")
    logger.info("This will go to remote endpoint")
    logger.warning("This warning should appear in the sheet")
    logger.error("This error with timestamp should be recorded")
    logger.debug("This debug message should NOT appear (level INFO)")

    # Clean up
    logger.removeHandler(remote_handler)
    print("\n✓ Direct handler test completed\n")


def stress_test(webhook_url: str, num_entries: int = 20):
    """
    Stress test with many log entries.

    Args:
        webhook_url: Your Google Apps Script webhook URL
        num_entries: Number of log entries to send
    """
    print("\n" + "=" * 80)
    print(f"STRESS TEST - Sending {num_entries} log entries")
    print("=" * 80)

    logger = logging.getLogger("StressTest")
    logger.setLevel(logging.INFO)

    start_time = time.time()

    with LoggerToRemotePost(logger, webhook_url, sheet_key="stress_test") as log:
        for i in range(num_entries):
            logger.info(f"Stress test entry #{i + 1}: timestamp={time.time()}, iteration={i}")
            if (i + 1) % 10 == 0:
                print(f"  Sent {i + 1}/{num_entries} entries...")

    elapsed = time.time() - start_time
    print(f"\n✓ Stress test completed in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed / num_entries:.3f} seconds per entry\n")


if __name__ == "__main__":
    # REPLACE THIS WITH YOUR ACTUAL WEBHOOK URL
    import os

    from dotenv import load_dotenv

    load_dotenv("../notebooks/.env")


    def get_env(name):
        return os.environ.get(name)


    WEBHOOK_URL = get_env("GOOGLE_APP_SCRIPT_LOGGER_URL")

    if WEBHOOK_URL == "YOUR_GOOGLE_APPS_SCRIPT_URL_HERE":
        print("ERROR: Please set your WEBHOOK_URL before running the test")
        print("Example: WEBHOOK_URL = 'https://script.google.com/macros/s/your-script-id/exec'")
        sys.exit(1)

    # Run the tests
    test_remote_logger(WEBHOOK_URL, sheet_key="kaggle")

    # Optional: Run additional tests  # test_remote_logger_without_context_manager(WEBHOOK_URL)  # stress_test(WEBHOOK_URL, num_entries=50)
