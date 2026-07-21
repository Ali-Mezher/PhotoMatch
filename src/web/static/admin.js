(() => {
  const menuButton = document.querySelector("[data-menu-toggle]");
  const menu = document.querySelector("#admin-nav");
  menuButton?.addEventListener("click", () => {
    const open = menu.classList.toggle("is-open");
    menuButton.setAttribute("aria-expanded", String(open));
  });

  if (document.querySelector("[data-auto-refresh]")) {
    window.setTimeout(() => window.location.reload(), 2500);
  }

  const indexProgress = document.querySelector("[data-index-progress]");
  if (indexProgress?.dataset.active === "true") {
    const label = indexProgress.querySelector("[data-progress-label]");
    const percent = indexProgress.querySelector("[data-progress-percent]");
    const bar = indexProgress.querySelector("[data-progress-bar]");
    const refreshProgress = async () => {
      try {
        const response = await fetch(indexProgress.dataset.url, {
          headers: { "Accept": "application/json" },
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Progress unavailable");
        const result = await response.json();
        percent.textContent = `${result.percent}%`;
        bar.value = result.percent;
        label.textContent = result.status === "queued" ? "Waiting" : "Indexing";
        if (result.status === "queued" || result.status === "indexing") {
          window.setTimeout(refreshProgress, 1000);
        } else {
          label.textContent = result.status === "indexed" ? "Complete" : "Stopped";
          window.setTimeout(() => window.location.reload(), 650);
        }
      } catch (_error) {
        window.setTimeout(refreshProgress, 2000);
      }
    };
    refreshProgress();
  }

  const selectionForm = document.querySelector("[data-selection-form]");
  if (selectionForm) {
    const checkboxes = [...selectionForm.querySelectorAll('input[name="photo_ids"]')];
    const count = selectionForm.querySelector("[data-selection-count]");
    const exportButton = selectionForm.querySelector("[data-export-button]");
    const updateSelection = () => {
      const selected = checkboxes.filter((checkbox) => checkbox.checked).length;
      count.textContent = `${selected} photo${selected === 1 ? "" : "s"} selected`;
      exportButton.disabled = selected === 0;
    };
    checkboxes.forEach((checkbox) => checkbox.addEventListener("change", updateSelection));
    updateSelection();
  }

  const previewDialog = document.querySelector("[data-preview-dialog]");
  const previewImage = document.querySelector("[data-preview-image]");
  const previewTitle = document.querySelector("[data-preview-title]");
  if (previewDialog && previewImage && previewTitle) {
    document.querySelectorAll("[data-preview-url]").forEach((button) => {
      button.addEventListener("click", () => {
        previewImage.src = button.dataset.previewUrl;
        previewImage.alt = `${button.dataset.previewName} protected preview`;
        previewTitle.textContent = button.dataset.previewName;
        previewDialog.showModal();
      });
    });
    document.querySelector("[data-preview-close]")?.addEventListener("click", () => previewDialog.close());
    previewDialog.addEventListener("click", (event) => {
      if (event.target === previewDialog) previewDialog.close();
    });
    previewDialog.addEventListener("close", () => {
      previewImage.removeAttribute("src");
      previewImage.alt = "";
    });
  }

  const form = document.querySelector("[data-batch-upload]");
  if (!form || !window.fetch || !window.FormData) return;
  const input = form.querySelector("input[type=file]");
  const zone = form.querySelector(".drop-zone");
  const copy = form.querySelector("[data-upload-copy]");
  const progress = form.querySelector("[data-upload-progress]");
  const bar = form.querySelector("[data-upload-bar]");
  const status = form.querySelector("[data-upload-status]");
  const button = form.querySelector("button[type=submit]");
  const csrf = form.querySelector("input[name=_csrf_token]").value;

  input.addEventListener("change", () => {
    copy.textContent = input.files.length ? `${input.files.length} photo(s) selected` : "Multiple files are supported.";
  });
  ["dragenter", "dragover"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("is-dragging"); }));
  zone.addEventListener("drop", event => { input.files = event.dataTransfer.files; input.dispatchEvent(new Event("change")); });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const files = [...input.files];
    if (!files.length) return;
    button.disabled = true;
    progress.hidden = false;
    let imported = 0;
    let rejected = 0;
    try {
      const MAX_FILES_PER_BATCH = 10;
      const MAX_BATCH_BYTES = 200 * 1024 * 1024;
      const batches = [];
      let batch = [];
      let batchBytes = 0;
      files.forEach((file) => {
        const batchIsFull = batch.length >= MAX_FILES_PER_BATCH;
        const batchIsTooLarge = batch.length && batchBytes + file.size > MAX_BATCH_BYTES;
        if (batchIsFull || batchIsTooLarge) {
          batches.push(batch);
          batch = [];
          batchBytes = 0;
        }
        batch.push(file);
        batchBytes += file.size;
      });
      if (batch.length) batches.push(batch);
      let uploadedFiles = 0;
      for (let index = 0; index < batches.length; index += 1) {
        const body = new FormData();
        body.append("_csrf_token", csrf);
        body.append("final_batch", index === batches.length - 1 ? "1" : "0");
        batches[index].forEach(file => body.append("photos", file));
        const firstPhoto = uploadedFiles + 1;
        const lastPhoto = uploadedFiles + batches[index].length;
        status.textContent = `Uploading photos ${firstPhoto}–${lastPhoto} of ${files.length}…`;
        const response = await fetch(form.action, { method: "POST", body, headers: { "X-Requested-With": "XMLHttpRequest", "Accept": "application/json" } });
        const result = await response.json();
        imported += result.imported || 0;
        rejected += result.rejected || 0;
        uploadedFiles = lastPhoto;
        bar.style.width = `${((index + 1) / batches.length) * 100}%`;
        if (!response.ok && !result.outcomes) throw new Error(result.error || "Upload failed.");
      }
      status.textContent = `${imported} imported · ${rejected} rejected. Opening inventory…`;
      window.setTimeout(() => window.location.reload(), 700);
    } catch (error) {
      status.textContent = error.message || "Upload failed. The successfully imported files were kept.";
      button.disabled = false;
    }
  });
})();
