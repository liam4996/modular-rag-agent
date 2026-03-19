"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def _run_ingestion_single(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    file_index: int = 0,
    total_files: int = 1,
    force: bool = False,
) -> bool:
    """Save a single uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # Write uploaded file to a temp location
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(f"[{file_index + 1}/{total_files}] {uploaded_file.name}: {label}")

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    success = False
    try:
        pipeline = IngestionPipeline(settings, collection=collection, force=force)
        result = pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        # Check if pipeline was successful AND generated chunks
        if result.success and result.chunk_count > 0:
            success = True
        elif result.success and result.chunk_count == 0:
            # File was skipped (already processed)
            if force:
                status_text.warning(f"[{file_index + 1}/{total_files}] {uploaded_file.name} was skipped (file content unchanged).")
            else:
                status_text.warning(f"[{file_index + 1}/{total_files}] {uploaded_file.name} was skipped (already processed). Enable 'Force' to reprocess.")
            success = True  # Still consider it success, just skipped
        else:
            status_text.error(f"[{file_index + 1}/{total_files}] {uploaded_file.name} failed: {result.error}")
    except Exception as exc:
        status_text.error(f"[{file_index + 1}/{total_files}] {uploaded_file.name} failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    
    return success


def _run_ingestion(
    uploaded_files: list,
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    force: bool = False,
) -> None:
    """Save multiple uploaded files to temp locations and run the pipeline."""
    total_files = len(uploaded_files)
    success_count = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.info(f"Processing [{idx + 1}/{total_files}]: {uploaded_file.name}...")
        if _run_ingestion_single(uploaded_file, collection, progress_bar, status_text, idx, total_files, force):
            success_count += 1
    
    # Final summary
    if success_count == total_files:
        status_text.success(f"✅ Successfully ingested {success_count}/{total_files} files into collection **{collection}**.")
    else:
        status_text.warning(f"⚠️ Ingested {success_count}/{total_files} files. Some files failed.")


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select files to ingest (支持多选)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input("Collection", value="default", key="ingest_collection")

    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个文件")
        
        # Add Force reprocess option
        col_btn, col_force = st.columns([2, 1])
        with col_btn:
            force_mode = st.checkbox("♻️ Force re-process (重新处理已上传的文件)", value=False, key="force_mode")
        with col_force:
            start_ingestion = st.button("🚀 Start Ingestion", key="btn_ingest", type="primary")
        
        if start_ingestion:
            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()
            _run_ingestion(uploaded_files, collection.strip() or "default", progress_bar, status_text, force=force_mode)

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents ingested yet.** "
            "Upload a PDF, TXT, MD, or DOCX file above and click \"Start Ingestion\" to begin."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
