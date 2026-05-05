(function () {
  "use strict";

  var API_BASE = "https://hke-backend.onrender.com";

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(message, type) {
    var box = $("contactStatus");
    if (!box) return;
    box.textContent = message || "";
    box.style.color = type === "error" ? "#b42318" : (type === "success" ? "#067647" : "#6b7280");
  }

  function cleanPhone(value) {
    return String(value || "").replace(/\D/g, "").slice(0, 10);
  }

  function setSubmitting(isSubmitting) {
    var btn = $("contactSubmitBtn");
    if (!btn) return;
    btn.disabled = !!isSubmitting;
    btn.textContent = isSubmitting ? "Sending..." : "Send";
  }

  async function postJSON(url, body) {
    var response = await window.fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    var data = {};
    try {
      data = await response.json();
    } catch (_err) {
      data = {};
    }
    if (!response.ok || data.ok === false) {
      throw new Error(data.detail || data.message || "Unable to send your enquiry right now.");
    }
    return data;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var form = $("contactForm");
    var phoneInput = $("contactPhone");
    if (!form) return;

    if (phoneInput) {
      phoneInput.addEventListener("input", function () {
        phoneInput.value = cleanPhone(phoneInput.value);
      });
    }

    form.addEventListener("submit", async function (event) {
      event.preventDefault();

      var payload = {
        name: String($("contactName").value || "").trim(),
        phone: cleanPhone($("contactPhone").value || ""),
        email: String($("contactEmail").value || "").trim(),
        message: String($("contactMessage").value || "").trim(),
        page: "contact.html",
        source: "website_contact"
      };

      if (!payload.name) {
        setStatus("Please enter your name.", "error");
        return;
      }
      if (payload.phone.length !== 10) {
        setStatus("Please enter a valid 10-digit mobile number.", "error");
        return;
      }
      if (payload.email && payload.email.indexOf("@") === -1) {
        setStatus("Please enter a valid email address.", "error");
        return;
      }
      if (!payload.message) {
        setStatus("Please enter your message.", "error");
        return;
      }

      try {
        setSubmitting(true);
        setStatus("Sending your enquiry...", "info");
        await postJSON(API_BASE + "/api/contact", payload);
        form.reset();
        setStatus("Your enquiry has been sent successfully. Our team will contact you soon.", "success");
      } catch (error) {
        setStatus(error.message || "Unable to send your enquiry right now.", "error");
      } finally {
        setSubmitting(false);
      }
    });
  });
})();
