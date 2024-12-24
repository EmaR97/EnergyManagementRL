# Scheduling Kaggle Notebook Execution with Cron and the Kaggle API ([source](https://www.kaggle.com/discussions/product-feedback/371090))

Executing Kaggle notebooks at specific times is a common user request, but Kaggle lacks a built-in scheduling feature.
Fortunately, Linux systems offer a robust task scheduler called **cron**, and Kaggle provides a public API for remote
notebook execution. This guide explains how to combine these tools for precise and automated notebook execution with
complex scheduling rules.

---

## Prerequisites

### System Requirements

- A Linux-based computer or cloud instance (
  e.g., [Oracle Cloud Always Free Services](https://www.oracle.com/cloud/free/)).
- [**cron** and **crontab**](https://man7.org/linux/man-pages/man5/crontab.5.html) (standard on most Linux
  distributions).
- System time set to your desired timezone (avoids timezone-related cron issues).

### Software Setup

1. **Install the Kaggle API:**
    - Create a Python virtual environment:
      ```bash
      python3 -m venv $HOME/kaggle
      source $HOME/kaggle/bin/activate
      ```
    - Install the Kaggle API:
      ```bash
      pip install kaggle
      ```
    - Authenticate using your Kaggle API token:
        - Download `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/account).
        - Place it in the `~/.kaggle/` directory:
          ```bash
          mkdir -p ~/.kaggle
          mv kaggle.json ~/.kaggle/
          chmod 600 ~/.kaggle/kaggle.json
          ```
    - Test the installation:
      ```bash
      kaggle datasets list
      ```
   For more details, refer to the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

---

## Automating Notebook Execution

### Step 1: Create an Execution Script

1. Create a directory for scripts:
   ```bash
   mkdir -p $HOME/kaggle_scheduler
   ```
2. Save the following script as `run_kaggle_ntbk.sh` in the directory:
   ```bash
   #! /bin/bash

   # Activate virtual environment
   source $HOME/kaggle/bin/activate

   # Log the execution time
   echo "================"
   echo "[$(date)] Starting execution of notebook: $1"

   # Change to working directory
   cd $HOME/kaggle_scheduler/

   # Pull the notebook and fix metadata
   kaggle kernels pull $1 -m
   sed -i.$(date +%F).OLD 's!code/!!g;s!datasets/!!g' kernel-metadata.json

   # Push the notebook for execution
   kaggle kernels push

   # Log completion
   echo "[$(date)] Completed execution of notebook: $1"
   kaggle kernels status $1
   echo "================"
   ```
3. Make the script executable:
   ```bash
   chmod +x $HOME/kaggle_scheduler/run_kaggle_ntbk.sh
   ```

### Step 2: Schedule the Script with Cron

1. Open the crontab editor:
   ```bash
   crontab -e
   ```
   For more information on using crontab, refer to
   the [crontab documentation](https://man7.org/linux/man-pages/man5/crontab.5.html).

2. Add cron entries to schedule your notebook execution. For example:
    - Run a notebook daily at 11:00 and 23:00 UTC:
      ```bash
      0 11,23 * * * $HOME/kaggle_scheduler/run_kaggle_ntbk.sh username/notebook-name >> $HOME/kaggle_scheduler/cron.log 2>&1
      ```
    - Run a notebook on weekdays at 13:00 UTC:
      ```bash
      0 13 * * 2-5 $HOME/kaggle_scheduler/run_kaggle_ntbk.sh username/notebook-name >> $HOME/kaggle_scheduler/cron.log 2>&1
      ```

3. Verify your changes:
   ```bash
   crontab -l
   ```

---

## Notes for Precision

- Kaggle notebooks may experience slight delays due to processing time for uploading and queuing. To align execution
  with your schedule, configure the script to run a few minutes early or include a time delay in your notebook code.

