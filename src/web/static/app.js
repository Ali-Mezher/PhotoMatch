(() => {
  const eventCode = document.querySelector("[data-event-code]");
  if (eventCode) {
    eventCode.addEventListener("input", () => {
      eventCode.value = eventCode.value.toUpperCase().replace(/[^0-9A-F-]/g, "");
    });
  }

  const upload = document.querySelector("#selfie");
  const fileName = document.querySelector("[data-file-name]");
  if (upload && fileName) {
    upload.addEventListener("change", () => {
      fileName.textContent = upload.files?.[0]?.name ||
        "JPG, PNG, WebP, or BMP · Maximum 12 MB";
    });
  }

  const cameraDialog = document.querySelector("[data-camera-dialog]");
  const cameraOpen = document.querySelector("[data-camera-open]");
  const cameraVideo = document.querySelector("[data-camera-video]");
  const cameraCanvas = document.querySelector("[data-camera-canvas]");
  const cameraCapture = document.querySelector("[data-camera-capture]");
  const cameraStatus = document.querySelector("[data-camera-status]");
  const cameraSupport = document.querySelector("[data-camera-support]");
  let cameraStream = null;

  const stopCamera = () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    if (cameraVideo) cameraVideo.srcObject = null;
    if (cameraCapture) cameraCapture.disabled = true;
  };

  const closeCamera = () => {
    stopCamera();
    cameraDialog?.close();
  };

  if (cameraOpen && cameraDialog && cameraVideo && cameraCanvas && cameraCapture) {
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraOpen.disabled = true;
      if (cameraSupport) {
        cameraSupport.textContent =
          "This browser cannot open the camera. Choose an existing selfie instead.";
      }
    } else {
      cameraOpen.addEventListener("click", async () => {
        cameraDialog.showModal();
        cameraStatus.textContent = "Requesting camera permission…";
        try {
          cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false,
          });
          cameraVideo.srcObject = cameraStream;
          await cameraVideo.play();
          cameraCapture.disabled = false;
          cameraStatus.textContent = "Center your face, then take the photo.";
        } catch (_error) {
          stopCamera();
          cameraStatus.textContent =
            "Camera access was unavailable. Allow permission or choose an existing selfie.";
        }
      });

      cameraCapture.addEventListener("click", () => {
        const width = cameraVideo.videoWidth;
        const height = cameraVideo.videoHeight;
        if (!width || !height) {
          cameraStatus.textContent = "The camera is not ready yet. Try again.";
          return;
        }
        cameraCanvas.width = width;
        cameraCanvas.height = height;
        cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0, width, height);
        cameraCanvas.toBlob((blob) => {
          if (!blob || !upload) {
            cameraStatus.textContent = "The photo could not be captured. Try again.";
            return;
          }
          const transfer = new DataTransfer();
          transfer.items.add(new File([blob], "camera-selfie.jpg", { type: "image/jpeg" }));
          upload.files = transfer.files;
          upload.dispatchEvent(new Event("change", { bubbles: true }));
          closeCamera();
          if (cameraSupport) cameraSupport.textContent = "Camera photo ready.";
        }, "image/jpeg", 0.92);
      });
    }

    document.querySelector("[data-camera-close]")?.addEventListener("click", closeCamera);
    document.querySelector("[data-camera-cancel]")?.addEventListener("click", closeCamera);
    cameraDialog.addEventListener("click", (event) => {
      if (event.target === cameraDialog) closeCamera();
    });
    cameraDialog.addEventListener("close", stopCamera);
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
