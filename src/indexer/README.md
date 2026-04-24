# Indexer

This package indexes PDF documents into a local Qdrant collection using a ColPali-family visual retriever model.

The current default model is `vidore/colqwen2-v1.0`, and the current storage backend is Qdrant local mode via `qdrant-client`.

## What this pipeline does

At a high level, the pipeline:

1. Finds candidate PDF files in the configured data directory.
2. Validates that every selected PDF is present in the domain mapping.
3. Renders each PDF page into an RGB image.
4. Encodes each rendered page into patch-level embeddings.
5. Builds one Qdrant point per page.
6. Upserts those points into a local Qdrant collection.
7. Appends a per-document status record to `artifacts/index_report.jsonl`.

## Directory structure and data flow

The package is organized as step-oriented directories so the data flow is easy to follow.

### `main.py`

`main.py` is the CLI entrypoint and orchestration layer.

It wires together validation, rendering, encoding, Qdrant writes, and index reporting.

### `load_docs/`

This step identifies which PDFs should be processed.

- `domain_mapping.py`
  - Defines the document-to-domain mapping.
  - Every selected PDF must exist in this mapping.
- `targets.py`
  - Resolves either a single requested file or all PDFs in the data directory.
  - Computes the source SHA-256 for each target document.

**Input:** files from `data/` plus the mapping table  
**Output:** `TargetDocument` records

### `validate/`

This step enforces input correctness before any expensive model work begins.

- Confirms the data directory exists.
- Confirms a requested file ends in `.pdf`.
- Confirms selected PDFs exist.
- Confirms selected PDFs are mapped to a domain.
- Can also report mapping gaps for PDFs present in `data/` but missing from the mapping.

**Input:** configured data directory, mapping, optional file selector  
**Output:** validated file list or an `InputValidationError`

### `render/`

This step converts each PDF page into a PIL image using PyMuPDF.

- `pdf_pages.py`
  - Opens a PDF.
  - Renders each page at the configured zoom.
  - Emits `RenderedPage` objects with page number, width, height, and image.

**Input:** `TargetDocument.file_path`  
**Output:** `RenderedPage` objects

### `encode/`

This step runs the ColPali-family model.

- `colpali.py`
  - Loads the configured model and processor.
  - Supports ColQwen2, ColQwen2.5, and ColPali naming patterns.
  - Encodes one rendered page into patch embeddings.
  - Infers the embedding dimension from the loaded model.

**Input:** rendered page images  
**Output:** `list[list[float]]` patch embeddings per page

### `flatten/`

This step turns each rendered page plus its embeddings into one Qdrant-ready point.

- `page_patches.py`
  - Builds a `PageInsertPoint`.
  - Uses one logical point per page.
  - Preserves metadata like document name, domain, page size, page number, source hash, timestamp, and run ID.

**Input:** `TargetDocument`, `RenderedPage`, patch embeddings  
**Output:** `PageInsertPoint`

### `insert/`

This step manages local Qdrant writes.

- `qdrant_writer.py`
  - Opens a local Qdrant database at the configured path.
  - Ensures the target collection exists.
  - Creates the collection as a multivector collection.
  - Upserts one Qdrant point per page.

**Input:** `PageInsertPoint` records  
**Output:** persisted Qdrant points

### `shared/`

This directory contains cross-cutting models, config, logging, utilities, and error types.

Notable pieces:

- `config.py`
  - Defines environment-configurable settings.
- `models.py`
  - Defines the internal data contracts that move through the pipeline.
- `errors.py`
  - Defines pipeline-specific exception types.

### `index_report.py`

This writes one JSONL record per document into the report file.

Each record includes the document name, file path, domain, page count, file hash, status, error message, and timestamp.

## End-to-end flow

A typical `index --all` run looks like this:

1. `main.py` loads `Settings`.
2. `load_docs.domain_mapping.load_domain_mapping()` builds the mapping dictionary.
3. `validate.inputs.validate_target_files()` ensures the selected PDFs are valid and mapped.
4. `load_docs.targets.resolve_target_documents()` builds `TargetDocument` objects.
5. `insert.qdrant_writer.QdrantInsertWriter.ensure_collection()` creates the Qdrant collection if needed.
6. For each document:
   1. `render.pdf_pages.PdfPageRenderer.render()` yields rendered pages.
   2. `encode.colpali.ColPaliPageEncoder.encode_page()` produces patch embeddings.
   3. `flatten.page_patches.build_page_point()` builds a page-level point.
   4. `insert.qdrant_writer.QdrantInsertWriter.upsert_points()` writes the point.
   5. `index_report.IndexReportWriter.record_success()` writes a success record.
7. If a document fails during indexing after selection, a failure record is appended to the report before the error is raised.

## Inputs you need before running

### PDF files

By default, PDFs are expected under:

```text
src/indexer/data
```

You can change this with `INDEXER_DATA_DIR`.

### Domain mapping

Every PDF that you index must appear in:

```text
src/indexer/load_docs/domain_mapping.py
```

If a PDF exists in the data directory but is missing from the mapping, validation fails.

This is intentional so unmapped inputs are treated as data-quality failures rather than silently indexed with incomplete metadata.

## Configuration and tunable parameters

All settings are defined in `src/indexer/shared/config.py` and use the `INDEXER_` environment variable prefix.

For example, `model_name` becomes `INDEXER_MODEL_NAME`.

### `INDEXER_DATA_DIR`

Default:

```text
src/indexer/data
```

What it controls:

- The directory scanned for `.pdf` files.

Performance impact:

- More files means longer validation and indexing runs.
- Putting the directory on fast local storage reduces file-read latency.

### `INDEXER_QDRANT_PATH`

Default:

```text
qdrant_data
```

What it controls:

- The local filesystem path for Qdrant persistence.

Performance impact:

- Faster disks improve collection creation and write throughput.
- Larger corpora will consume more local disk space.

### `INDEXER_COLLECTION_NAME`

Default:

```text
colpali_page_patches
```

What it controls:

- The Qdrant collection name used for page points.

Performance impact:

- No direct runtime speed effect.
- Important for isolating experiments across models or datasets.

Recommended use:

- Use a separate collection when comparing models or indexing strategies.

### `INDEXER_MODEL_NAME`

Default:

```text
vidore/colqwen2-v1.0
```

What it controls:

- Which ColPali-family checkpoint is loaded.

Performance impact:

- This is one of the biggest knobs in the system.
- Larger or more complex models usually improve retrieval quality at the cost of higher memory use and slower inference.
- Different model families can change embedding dimension, model load time, and per-page encode latency.

Recommendation:

- Keep the default ColQwen2 model unless you have a specific reason to switch.
- If you switch models, consider using a fresh Qdrant collection.

### `INDEXER_DEVICE`

Default:

```text
auto
```

What it controls:

- Which torch device to use.
- `auto` resolves in this order: `cuda`, then `mps`, then `cpu`.

Performance impact:

- `cuda` is typically fastest for throughput.
- `mps` can be significantly faster than CPU on Apple Silicon, but exact gains vary.
- `cpu` is the slowest option and may be impractical for larger corpora.

Recommendation:

- Leave this on `auto` unless you are debugging device-specific issues.

### `INDEXER_RENDER_ZOOM`

Default:

```text
2.0
```

What it controls:

- The PyMuPDF render scale applied to every page.

Performance impact:

- Higher values increase rendered image resolution.
- Higher resolution can help preserve small text or dense layouts.
- Higher resolution also increases render time, memory pressure, and model inference cost.

Typical tradeoff:

- Lower zoom: faster, lower memory, potentially less detail.
- Higher zoom: slower, higher memory, potentially better page representation for small text.

Recommendation:

- Tune this carefully if your PDFs contain very small fonts, screenshots, or dense tables.

### `INDEXER_BATCH_SIZE_PAGES`

Default:

```text
1
```

What it controls:

- This setting exists in the config model, but the current pipeline processes pages one at a time.
- In the current implementation, it is not wired into the indexing loop.

Performance impact:

- No impact today.
- If batching is added later, this would likely become a throughput-versus-memory knob.

### `INDEXER_RECREATE_COLLECTION`

Default:

```text
false
```

What it controls:

- Whether the target Qdrant collection is dropped and recreated before indexing.

Performance impact:

- Recreating the collection adds setup work.
- It removes previously indexed data, so use it intentionally.

Recommended use:

- Use this when changing the embedding model, changing the target dataset, or resetting a failed experiment.

You can also trigger this from the CLI with `--recreate-collection` on the `index` command.

### `INDEXER_LOG_LEVEL`

Default:

```text
INFO
```

What it controls:

- Logging verbosity.

Performance impact:

- Usually minor.
- Very verbose logging can add overhead in long runs.

### `INDEXER_REPORT_PATH`

Default:

```text
artifacts/index_report.jsonl
```

What it controls:

- Where per-document result records are appended.

Performance impact:

- Minimal under normal workloads.
- Useful for auditability and failure review.

## How to run

Run these commands from the repository root.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
pip install -e ".[dev]"
```

### 3. Add PDFs and verify mapping

- Put your PDFs in `src/indexer/data` or set `INDEXER_DATA_DIR`.
- Ensure every PDF filename is mapped in `src/indexer/load_docs/domain_mapping.py`.

### 4. Validate inputs before indexing

Validate all PDFs:

```bash
python -m indexer.main validate --all
```

Validate a single PDF:

```bash
python -m indexer.main validate --file your_file.pdf
```

Show PDFs that exist in `data/` but are missing from the domain mapping:

```bash
python -m indexer.main show-mapping-gaps
```

### 5. Index documents

Index all PDFs:

```bash
python -m indexer.main index --all
```

Index a single PDF:

```bash
python -m indexer.main index --file your_file.pdf
```

Recreate the collection before indexing:

```bash
python -m indexer.main index --all --recreate-collection
```

### 6. Inspect collection configuration

```bash
python -m indexer.main describe-collection
```

## Example environment overrides

```bash
export INDEXER_DEVICE=mps
export INDEXER_RENDER_ZOOM=2.5
export INDEXER_COLLECTION_NAME=colqwen_experiment_a
python -m indexer.main index --all
```

## Output artifacts

### Qdrant collection

The pipeline writes one point per page into the configured local Qdrant collection.

Each point contains:

- page-level metadata payload
- multivector patch embeddings for the rendered page

### Index report

A JSONL report is appended to:

```text
artifacts/index_report.jsonl
```

Each line represents one document outcome.

This is useful for:

- auditing what was indexed
- reviewing failures
- tracking page counts and source hashes across runs

## Dependency notes and known issues

This project intentionally pins a few critical packages in `pyproject.toml`:

- `colpali-engine==0.3.15`
- `transformers==5.5.4`
- `peft==0.18.1`
- `qdrant-client==1.17.1`

These versions are pinned because the ColPali-family model stack is sensitive to version mismatches.

### Do not casually upgrade model-stack packages independently

In particular, avoid upgrading `transformers`, `peft`, or `colpali-engine` one at a time without validating the combination.

Why this matters:

- Some newer `transformers` releases expect APIs or behaviors that are incompatible with older `peft` ranges.
- `colpali-engine` compatibility can depend on those exact versions.
- A mismatched set often fails at model or adapter load time, not at install time.

### Default model choice

The current default is a supported ColQwen checkpoint:

```text
vidore/colqwen2-v1.0
```

That default is the safest starting point for this repository.

If you switch back to an older PaliGemma/ColPali-style checkpoint, re-check dependency compatibility before assuming the current pinned set will behave the same way.

### Torch installation

`torch` can be the most platform-sensitive dependency.

If install or runtime device detection behaves unexpectedly:

- confirm that `torch` installed correctly in the active environment
- confirm the device backend you expect is available
- temporarily force `INDEXER_DEVICE=cpu` to separate model issues from accelerator issues

### PyMuPDF is required for rendering

If page rendering fails very early, make sure `pymupdf` is installed successfully.

### Qdrant local mode

This repo uses Qdrant local mode through `qdrant-client`, so you do not need to start a separate Qdrant server for the default workflow.

## Practical tuning guidance

If you want the highest-leverage performance adjustments, start here:

1. **Device selection**
   - Prefer `cuda` or `mps` over `cpu` when available.
2. **Render zoom**
   - Lower it if runs are too slow or memory-heavy.
   - Raise it if small text quality is the main problem.
3. **Model choice**
   - Keep the default model for stability.
   - Change models only when you are intentionally comparing quality or behavior.
4. **Collection isolation**
   - Use a new collection name for experiments so results stay easy to compare.

## Troubleshooting checklist

If indexing fails:

1. Run `python -m indexer.main validate --all` first.
2. Run `python -m indexer.main show-mapping-gaps` to catch unmapped PDFs.
3. Confirm the active environment has all pinned dependencies installed.
4. Confirm the selected torch device is actually available.
5. If you changed models, consider recreating the collection.
6. Review `artifacts/index_report.jsonl` for per-document failure details.
