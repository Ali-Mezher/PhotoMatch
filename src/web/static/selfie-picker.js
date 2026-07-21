(() => {
  document.querySelectorAll("[data-selfie-picker]").forEach((picker) => {
    const upload = picker.querySelector("#selfie");
    const uploadChoice = picker.querySelector("[data-selfie-upload]");
    const fileName = picker.querySelector("[data-file-name]");
    const cameraDialog = picker.querySelector("[data-camera-dialog]");
    const cameraOpen = picker.querySelector("[data-camera-open]");
    const cameraCopy = picker.querySelector("[data-camera-copy]");
    const cameraVideo = picker.querySelector("[data-camera-video]");
    const cameraCanvas = picker.querySelector("[data-camera-canvas]");
    const cameraCapture = picker.querySelector("[data-camera-capture]");
    const cameraStatus = picker.querySelector("[data-camera-status]");
    const cameraSupport = picker.querySelector("[data-camera-support]");
    let cameraStream = null;
    let pendingSource = null;

    const setSelectedSource = (source) => {
      const uploadSelected = source === "upload";
      const cameraSelected = source === "camera";
      uploadChoice?.classList.toggle("is-selected", uploadSelected);
      cameraOpen?.classList.toggle("is-selected", cameraSelected);
      cameraOpen?.setAttribute("aria-pressed", String(cameraSelected));

      if (fileName) {
        fileName.textContent = uploadSelected
          ? upload.files[0].name
          : fileName.dataset.defaultCopy;
      }
      if (cameraCopy) {
        cameraCopy.textContent = cameraSelected
          ? "Camera photo selected"
          : cameraCopy.dataset.defaultCopy;
      }
    };

    upload?.addEventListener("change", () => {
      if (!upload.files?.length) {
        setSelectedSource(null);
        pendingSource = null;
        return;
      }
      setSelectedSource(pendingSource || "upload");
      pendingSource = null;
    });

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

    if (!cameraOpen || cameraOpen.disabled || !cameraDialog || !cameraVideo || !cameraCanvas || !cameraCapture) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      cameraOpen.disabled = true;
      if (cameraSupport) {
        cameraSupport.textContent =
          "This browser cannot open the camera. Choose an existing selfie instead.";
      }
      return;
    }

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
        pendingSource = "camera";
        upload.files = transfer.files;
        upload.dispatchEvent(new Event("change", { bubbles: true }));
        closeCamera();
        if (cameraSupport) cameraSupport.textContent = "Camera photo ready.";
      }, "image/jpeg", 0.92);
    });

    picker.querySelector("[data-camera-close]")?.addEventListener("click", closeCamera);
    picker.querySelector("[data-camera-cancel]")?.addEventListener("click", closeCamera);
    cameraDialog.addEventListener("click", (event) => {
      if (event.target === cameraDialog) closeCamera();
    });
    cameraDialog.addEventListener("close", stopCamera);
  });
})();
