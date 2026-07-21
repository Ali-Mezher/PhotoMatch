(() => {
  const RAW_LEFT = "raw_left";
  const MAX_CAPTURE_EDGE = 1280;
  const JPEG_QUALITY = 0.85;

  document.querySelectorAll("[data-liveness-picker]").forEach((picker) => {
    const form = picker.closest("form");
    const consent = form?.querySelector('input[name="consent"]');
    const launch = picker.querySelector("[data-liveness-open]");
    const label = picker.querySelector("[data-liveness-label]");
    const copy = picker.querySelector("[data-liveness-copy]");
    const support = picker.querySelector("[data-liveness-support]");
    const dialog = picker.querySelector("[data-liveness-dialog]");
    const video = picker.querySelector("[data-liveness-video]");
    const canvas = picker.querySelector("[data-liveness-canvas]");
    const capture = picker.querySelector("[data-liveness-capture]");
    const status = picker.querySelector("[data-liveness-status]");
    const arrow = picker.querySelector("[data-liveness-arrow]");
    const stepLabel = picker.querySelector("[data-liveness-step-label]");
    const steps = [...picker.querySelectorAll("[data-liveness-step]")];
    const frameInputs = [...picker.querySelectorAll("[data-liveness-frame]")];
    const submit = form?.querySelector("[data-search-submit]");
    const direction = picker.dataset.challengeDirection;
    let stream = null;
    let currentStep = 0;
    let complete = false;

    const setSubmitReady = (ready) => {
      if (!submit) return;
      submit.dataset.searchDisabled = ready ? "false" : "true";
      submit.disabled = !ready;
    };

    const stopCamera = () => {
      stream?.getTracks().forEach((track) => track.stop());
      stream = null;
      if (video) video.srcObject = null;
      if (capture) capture.disabled = true;
    };

    const clearFrames = () => {
      frameInputs.forEach((input) => { input.value = ""; });
    };

    const renderStep = () => {
      steps.forEach((step, index) => {
        step.classList.toggle("is-complete", index < currentStep);
        step.classList.toggle("is-current", index === currentStep);
      });
      stepLabel.textContent = `Step ${currentStep + 1} of 3`;
      arrow.hidden = currentStep !== 1;
      if (currentStep === 0) {
        status.textContent = "Center your face and look straight at the camera.";
        capture.textContent = "Capture Front";
      } else if (currentStep === 1) {
        status.textContent = "Turn your head toward the arrow and hold.";
        capture.textContent = "Capture Turn";
      } else {
        status.textContent = "Return to center and look straight at the camera.";
        capture.textContent = "Capture Final";
      }
    };

    const reset = (message = "Your camera stays on this device. Captured frames are not saved.") => {
      complete = false;
      currentStep = 0;
      clearFrames();
      picker.classList.remove("is-complete");
      launch?.setAttribute("aria-pressed", "false");
      if (label) label.textContent = "Start Live Selfie Check";
      if (copy) copy.textContent = "Consent first, then complete a quick 3-step camera check";
      if (support) support.textContent = message;
      setSubmitReady(false);
      renderStep();
    };

    const close = (preserve = false) => {
      stopCamera();
      if (!preserve) reset();
      if (dialog?.open) dialog.close();
    };

    const captureFrame = () => new Promise((resolve, reject) => {
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      if (!sourceWidth || !sourceHeight) {
        reject(new Error("Camera is not ready"));
        return;
      }
      const scale = Math.min(1, MAX_CAPTURE_EDGE / Math.max(sourceWidth, sourceHeight));
      canvas.width = Math.round(sourceWidth * scale);
      canvas.height = Math.round(sourceHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Capture failed"));
      }, "image/jpeg", JPEG_QUALITY);
    });

    if (!form || !consent || !launch || !dialog || !video || !canvas ||
        !capture || frameInputs.length !== 3) return;

    arrow.textContent = direction === RAW_LEFT ? "→" : "←";
    setSubmitReady(false);

    if (!navigator.mediaDevices?.getUserMedia || typeof DataTransfer === "undefined") {
      launch.disabled = true;
      support.textContent =
        "This browser cannot complete the live check. Ask staff for manual photo lookup.";
      return;
    }

    consent.addEventListener("change", () => {
      launch.disabled = !consent.checked;
      if (!consent.checked) {
        close(false);
        support.textContent = "Consent is required before the camera can open.";
      } else {
        support.textContent = "Ready for a quick 3-step camera check.";
      }
    });

    launch.addEventListener("click", async () => {
      if (!consent.checked) return;
      reset("Requesting camera permission…");
      dialog.showModal();
      status.textContent = "Requesting camera permission…";
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 960 },
          },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        capture.disabled = false;
        renderStep();
      } catch (_error) {
        stopCamera();
        status.textContent =
          "Camera access was unavailable. Allow it or ask staff for manual photo lookup.";
      }
    });

    capture.addEventListener("click", async () => {
      capture.disabled = true;
      try {
        const blob = await captureFrame();
        const transfer = new DataTransfer();
        transfer.items.add(new File([blob], `liveness-${currentStep + 1}.jpg`, {
          type: "image/jpeg",
        }));
        frameInputs[currentStep].files = transfer.files;

        if (currentStep < 2) {
          currentStep += 1;
          renderStep();
          capture.disabled = false;
          return;
        }

        complete = true;
        picker.classList.add("is-complete");
        launch.setAttribute("aria-pressed", "true");
        label.textContent = "Live Selfie Ready";
        copy.textContent = "3 camera steps captured · Select to retake";
        support.textContent = "Live selfie ready. You can now search this event.";
        setSubmitReady(consent.checked);
        close(true);
      } catch (_error) {
        status.textContent = "That frame could not be captured. Hold still and try again.";
        capture.disabled = false;
      }
    });

    const cancel = () => close(false);
    picker.querySelector("[data-liveness-close]")?.addEventListener("click", cancel);
    picker.querySelector("[data-liveness-cancel]")?.addEventListener("click", cancel);
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) cancel();
    });
    dialog.addEventListener("close", stopCamera);
    form.addEventListener("submit", (event) => {
      if (!complete) {
        event.preventDefault();
        support.textContent = "Complete the live-selfie check before searching.";
      }
    });
    window.addEventListener("pageshow", () => {
      launch.disabled = !consent.checked;
      if (!complete) setSubmitReady(false);
    });
  });
})();
