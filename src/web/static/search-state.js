(() => {
  const resetForm = (form) => {
    const submit = form.querySelector("[data-search-submit]");
    const label = form.querySelector("[data-search-submit-label]");
    const pending = form.querySelector("[data-search-pending]");
    form.classList.remove("is-searching");
    form.removeAttribute("aria-busy");
    if (submit) submit.disabled = submit.dataset.searchDisabled === "true";
    if (label) label.textContent = label.dataset.defaultLabel || label.textContent;
    if (pending) pending.hidden = true;
  };

  document.querySelectorAll("[data-search-form]").forEach((form) => {
    const submit = form.querySelector("[data-search-submit]");
    const label = form.querySelector("[data-search-submit-label]");
    const pending = form.querySelector("[data-search-pending]");
    if (!submit || !label || !pending) return;
    label.dataset.defaultLabel = label.textContent;

    form.addEventListener("submit", () => {
      if (!form.checkValidity()) return;
      form.classList.add("is-searching");
      form.setAttribute("aria-busy", "true");
      submit.disabled = true;
      label.textContent = "Searching…";
      pending.hidden = false;
    });
  });

  window.addEventListener("pageshow", () => {
    document.querySelectorAll("[data-search-form]").forEach(resetForm);
  });
})();
