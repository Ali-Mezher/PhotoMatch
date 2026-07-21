(() => {
  const eventCode = document.querySelector("[data-event-code]");
  if (eventCode) {
    eventCode.addEventListener("input", () => {
      eventCode.value = eventCode.value.toUpperCase().replace(/[^0-9A-F-]/g, "");
    });
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

    checkboxes.forEach((checkbox) => {
      checkbox.addEventListener("change", updateSelection);
    });
    updateSelection();
  }

  const dialog = document.querySelector("[data-preview-dialog]");
  const dialogImage = document.querySelector("[data-preview-image]");
  const dialogTitle = document.querySelector("[data-preview-title]");
  if (dialog && dialogImage && dialogTitle) {
    document.querySelectorAll("[data-preview-url]").forEach((button) => {
      button.addEventListener("click", () => {
        dialogImage.src = button.dataset.previewUrl;
        dialogImage.alt = `${button.dataset.previewName} protected preview`;
        dialogTitle.textContent = button.dataset.previewName;
        dialog.showModal();
      });
    });

    document.querySelector("[data-preview-close]")?.addEventListener("click", () => {
      dialog.close();
    });

    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });

    dialog.addEventListener("close", () => {
      dialogImage.removeAttribute("src");
      dialogImage.alt = "";
    });
  }
})();
