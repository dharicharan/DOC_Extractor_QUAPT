# Running the demo on Kaggle

The demo is a Gradio app. You upload a document, it classifies the document, extracts the fields, and draws boxes on each page. It runs from the Kaggle notebook in `src/doc_extractor/notebookbdb318963a.ipynb`.

## 1. Session settings (before running anything)

In the notebook's right-hand panel, open **Session options**:

- **Internet:** ON. Needed to clone the repo and download models.
- **Accelerator:** GPU T4 x2 (or a single T4). The notebook works on either.
- **Persistence:** leave the default.

If you change any of these after a run, restart the session (**Run > Restart**) so the old GPU memory is cleared.

## 2. Run the cells in order

The notebook has four code cells. Run them top to bottom.

| Cell | What it does | Time (first run) |
| --- | --- | --- |
| 1. Get code | Clones the repo the first time, or runs `git pull` after that. Prints the latest commit. | under 1 min |
| 2. Install | Installs the Python packages (`uv`, `transformers`, `gradio`, and others). | a few minutes |
| 3. Download models | Downloads the model weights. Qwen3-VL-4B is required. GLM-OCR, Qwen3-4B-Instruct and Qwen2.5-7B are optional, and the cell skips any that fail. | 10 to 20 min (about 20 GB) |
| 4. Launch | Frees port 7860, then starts the app. | the app itself loads the model at startup, 1 to 2 min |

When the app is ready, cell 4 prints a line like `Running on public URL: https://....gradio.live`. Open that link.

**Cell 1 prints the commit.** Check it matches the latest commit on GitHub. If it doesn't, the pull failed.

After any code change pushed to GitHub, you only need to re-run cells 1 and 4. Cells 2 and 3 only need re-running after a new session.

## 3. Using the app

1. Upload a file (PDF, JPG, PNG or TIFF).
2. Leave the category, type, language and engine on **Auto-Detect** / **Follow routing table** unless you want to override them.
3. Click **Classify & Extract**.
4. Watch the **Pipeline steps** panel. Each step shows its result and time:
   - split into page images
   - classify the document from page 1
   - extract each page
   - validate and build the JSON
5. Read the results:
   - **Visual Grounding:** the annotated pages, with a box around each value.
   - **Canonical JSON Output:** the full result. You can also download it.

The first classification is slower, because it's the first time the model is used. Later runs in the same session are faster.

## 4. Settings you can change

Set these with `%env` in cell 4, before `!python app.py`:

| Variable | Default | What it does |
| --- | --- | --- |
| `DOC_MAX_PAGES` | 5 | How many pages a PDF reads. Longer files are flagged. |
| `DOC_MAX_IMAGE_PIXELS` | 1300000 | Size cap for the page image sent to the extraction model. Raise it on a bigger GPU. About 4200000 gives full 200 DPI. |
| `DOC_CLASSIFY_MAX_PIXELS` | 500000 | Size of the image used for classification. |
| `DOC_DEFAULT_MODEL` | Qwen/Qwen3-VL-4B-Instruct | The main vision model. Set to `Qwen/Qwen3-VL-8B-Instruct` on a 24 GB GPU. |
| `DOC_ROUTER_MODEL` | (same as default) | A different model just for classification. |
| `DOC_DEVICE_MAP` | single | `single` puts the vision model on GPU 0. `balanced` spreads it across both GPUs. |
| `DOC_PRELOAD` | 1 | Loads the models at startup. Set to 0 to skip. |

Example:

```
%env DOC_MAX_IMAGE_PIXELS=2000000
!python app.py
```

## 5. Common problems

| Error or symptom | Likely cause | Fix |
| --- | --- | --- |
| `Temporary failure in name resolution` or model download fails | Internet is off | Turn Internet ON in session options, then restart and re-run from cell 1. |
| `No such file or directory: .../DOC_Extractor_QUAPT/src/doc_extractor` | Cell 1 didn't clone | Turn Internet on, re-run cell 1. |
| `CUDA out of memory` during extraction | Page image too big for the GPU | Lower `DOC_MAX_IMAGE_PIXELS` (for example 1000000), restart, re-run cell 4. |
| `CUDA out of memory` at the very start | Old session still holding GPU memory | Restart the session. |
| `Expecting ',' delimiter` or a JSON error on a digital document | The text model's answer was cut off | Pull the latest code; the parser now keeps the complete fields. |
| GLM-OCR fails to load | The installed `transformers` is too old for GLM-OCR | Run `!uv pip install --system -U transformers`, restart, and re-run cells 2 to 4. The app falls back to Qwen3-VL if the route fails. |
| The app shows "Running on local URL" but no public link | Gradio share link not created | Stop cell 4 and re-run it. Check that Internet is ON. |
| Port 7860 already in use | An earlier run is still going | Cell 4 frees the port. If it persists, stop the cell and re-run. |
| The steps panel stops on a classify or extract step | Still loading the model, or the step failed | Wait for the timer. If a step turns red, copy the error message. |

## 6. What to send back when something fails

1. The **Pipeline steps** table, including the failed row.
2. The **Status** text box.
3. The output of cell 1, which shows the commit.
4. The filename and whether it's a scan or a digital PDF.
