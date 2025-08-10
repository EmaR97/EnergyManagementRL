You can use GitHub Actions to automate the notebook refresh process. Here’s how to set it up:

**1. Store your Kaggle API token securely:**  
Add your `kaggle.json` file as a GitHub Secret (e.g., `KAGGLE_JSON`).

**2. Create a workflow file:**  
Add a file like `.github/workflows/kaggle_refresh.yml` to your repo.

**3. Workflow example:**  
This workflow runs on a schedule, sets up Python, installs the Kaggle API, restores your Kaggle token, and executes the
notebook refresh commands.

```yaml
name: Refresh Kaggle Notebook

on:
  schedule:
    - cron: '0 11,23 * * *' # Runs at 11:00 and 23:00 UTC daily

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Kaggle API
        run: pip install kaggle

      - name: Restore Kaggle API token
        run: |
          mkdir -p ~/.kaggle
          echo "${{ secrets.KAGGLE_JSON }}" > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Pull and push notebook
        run: |
          kaggle kernels pull emanuelerapisarda/06-real-system-interaction -m
          # Optionally fix metadata if needed
          # sed -i 's!code/!!g;s!datasets/!!g' kernel-metadata.json
          kaggle kernels push
```

**4. Commit and push the workflow file.**  
GitHub Actions will now refresh your Kaggle notebook on the schedule you set.

**Note:**  
Replace `username/notebook-name` with your actual Kaggle notebook path.  
You can adjust the cron schedule as needed.  
All secrets must be set in your repository’s settings.