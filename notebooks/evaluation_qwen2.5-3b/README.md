## Running the Combined Evaluation

For each seed run, upload the generated prediction, generation CSV files from the `results` folder into the Colab environment.

For example, for seed 42, upload the CSV files produced by the corresponding recursive and human-control runs.

Then open and run:

```text
combined_evaluation_seed42_3b_colab.ipynb
```

Run the notebook from top to bottom to generate the combined evaluation results and analysis files.

### Option 1: Zip the Evaluation Results Manually

After the notebook finishes, navigate to the directory containing the generated evaluation files and run:

```bash
zip -r evaluation.zip .
```

This creates:

```text
evaluation.zip
```

You can then download the ZIP file manually from the Colab file browser.

### Option 2: Save the Results Directly to Google Drive

Mount Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then create a ZIP archive of the Colab output and save it directly to Drive:

```python
from pathlib import Path
import shutil

# Destination folder in Google Drive
PROJECT_DIR = Path("/content/drive/MyDrive/research")

# Make sure the destination folder exists
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# Folder containing the files to archive
SOURCE_DIR = "/content"

# Output path without the .zip extension
ZIP_PATH = PROJECT_DIR / "analysis"

shutil.make_archive(str(ZIP_PATH), "zip", SOURCE_DIR)

print(f"Saved: {ZIP_PATH}.zip")
```

The resulting archive will be saved as:

```text
/content/drive/MyDrive/research/analysis.zip
```

### Recommended Workflow

```text
Seed run
   ↓
results/
   ↓
Upload generated CSV files to Colab
   ↓
Run combined_evaluation_seed42_3b_colab.ipynb
   ↓
Generate combined evaluation outputs
   ↓
Zip results
   ↓
Download manually or save the ZIP to Google Drive
```

For other seeds, use the corresponding combined-evaluation notebook and CSV files for that seed.
