(function (window, document) {
  "use strict";

  var API_BASE = "https://hke-backend.onrender.com";
  var SEND_OTP_API = API_BASE + "/api/auth/send-otp";
  var VERIFY_OTP_API = API_BASE + "/api/auth/verify-otp";
  var REQUEST_TIMEOUT_MS = 15000;

  var LOGGED_IN_KEY = "HKE_CUSTOMER_LOGGED_IN";
  var PHONE_KEY = "HKE_CUSTOMER_PHONE";
  var LOGIN_TIME_KEY = "HKE_CUSTOMER_LOGIN_TIME";
  var RETURN_URL_KEY = "HKE_CUSTOMER_RETURN_URL";
  var PROFILE_KEY = "HKE_CUSTOMER_PROFILE";
  var NAME_KEY = "HKE_CUSTOMER_NAME";
  var EMAIL_KEY = "HKE_CUSTOMER_EMAIL";
  var LEGACY_LOGGED_IN_KEY = "hke_logged_in";
  var LEGACY_PHONE_KEY = "hke_customer_phone";
  var LEGACY_MOBILE_KEY = "hke_customer_mobile";
  var LEGACY_USER_PHONE_KEY = "user_phone";
  var LEGACY_USER_NAME_KEY = "user_name";
  var LEGACY_USER_EMAIL_KEY = "user_email";
  var pendingAction = null;
  var authOverlayEl = null;
  var overlayStyleInjected = false;

  var PROTECTED_PAGES = {
    "ai-planner.html": true,
    "book.html": true,
    "booking-details.html": true,
    "payment-confirmation.html": true,
    "my-orders.html": true,
    "order-details.html": true,
    "pilgrimage.html": true,
    "pilgrimage-booking.html": true,
    "itinerary-result.html": true,
    "finalize.html": true,
    "kerala.html": true,
    "manali.html": true,
    "kashmir.html": true,
    "leh-ladakh.html": true
  };

  function getPageName() {
    var path = window.location.pathname || "";
    var name = path.split("/").pop();
    return name || "index.html";
  }

  function getCurrentRelativeUrl() {
    var page = getPageName();
    return page + (window.location.search || "") + (window.location.hash || "");
  }

  function normalizePhone(v) {
    var x = String(v || "").replace(/\D/g, "");
    if (x.length === 12 && x.indexOf("91") === 0) x = x.slice(2);
    if (x.length > 10) x = x.slice(-10);
    return x;
  }

  function isLoggedIn() {
    var phone =
      window.localStorage.getItem(PHONE_KEY) ||
      window.localStorage.getItem(LEGACY_PHONE_KEY) ||
      window.localStorage.getItem(LEGACY_MOBILE_KEY) ||
      window.localStorage.getItem(LEGACY_USER_PHONE_KEY);
    var state =
      window.localStorage.getItem(LOGGED_IN_KEY) === "true" ||
      window.localStorage.getItem(LEGACY_LOGGED_IN_KEY) === "true";
    return !!phone && state;
  }

  function getStoredCustomerProfile() {
    var raw = null;
    try {
      raw = window.localStorage.getItem(PROFILE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (err) {
      return {};
    }
  }

  function saveCustomerProfile(profile) {
    var existing = getStoredCustomerProfile();
    var next = Object.assign({}, existing, profile || {});
    next.phone = normalizePhone(next.phone || existing.phone || "");

    if (!next.phone) return existing || {};

    if (next.name) next.name = String(next.name).trim();
    if (next.email) next.email = String(next.email).trim();
    next.updatedAt = new Date().toISOString();

    window.localStorage.setItem(PROFILE_KEY, JSON.stringify(next));
    window.localStorage.setItem(PHONE_KEY, next.phone);
    window.localStorage.setItem(LEGACY_PHONE_KEY, next.phone);
    window.localStorage.setItem(LEGACY_MOBILE_KEY, next.phone);
    window.localStorage.setItem(LEGACY_USER_PHONE_KEY, next.phone);
    window.localStorage.setItem("HKE_OTP_VERIFIED_PHONE", next.phone);
    window.localStorage.setItem("HKE_ORDER_LOOKUP_MODE", "phone");
    window.localStorage.setItem("HKE_ORDER_LOOKUP_VALUE", next.phone);
    if (next.name) {
      window.localStorage.setItem(NAME_KEY, next.name);
      window.localStorage.setItem(LEGACY_USER_NAME_KEY, next.name);
    }
    if (next.email) {
      window.localStorage.setItem(EMAIL_KEY, next.email);
      window.localStorage.setItem(LEGACY_USER_EMAIL_KEY, next.email);
    }
    return next;
  }

  function setReturnUrl(url) {
    if (!url) return;
    window.localStorage.setItem(RETURN_URL_KEY, url);
  }

  function getStoredReturnUrl() {
    var fromQuery = "";

    try {
      fromQuery = new URLSearchParams(window.location.search).get("returnUrl") || "";
    } catch (err) {
      fromQuery = "";
    }

    return fromQuery || window.localStorage.getItem(RETURN_URL_KEY) || "";
  }

  function consumeReturnUrl() {
    var value = getStoredReturnUrl();
    window.localStorage.removeItem(RETURN_URL_KEY);
    return value;
  }

  function resolveReturnUrl(url) {
    var raw = String(url || "").trim();
    if (!raw) return "";
    if (raw.charAt(0) === "#") return getPageName() + raw;
    if (/^https?:\/\//i.test(raw)) {
      try {
        var parsed = new URL(raw);
        if (parsed.origin !== window.location.origin) return "";
        var localName = parsed.pathname.split("/").pop() || "index.html";
        return localName + (parsed.search || "") + (parsed.hash || "");
      } catch (err) {
        return "";
      }
    }
    return raw;
  }

  function saveCustomerLogin(phone, extras) {
    var normalized = normalizePhone(phone);
    if (!normalized) return "";

    window.localStorage.setItem(LOGGED_IN_KEY, "true");
    window.localStorage.setItem(LEGACY_LOGGED_IN_KEY, "true");
    window.localStorage.setItem(PHONE_KEY, normalized);
    window.localStorage.setItem(LOGIN_TIME_KEY, new Date().toISOString());
    window.localStorage.setItem("HKE_ORDER_LOOKUP_MODE", "phone");
    window.localStorage.setItem("HKE_ORDER_LOOKUP_VALUE", normalized);
    window.localStorage.setItem("HKE_OTP_VERIFIED_PHONE", normalized);
    saveCustomerProfile(Object.assign({}, extras || {}, { phone: normalized }));
    return normalized;
  }

  function clearCustomerLogin() {
    [
      LOGGED_IN_KEY,
      PHONE_KEY,
      LOGIN_TIME_KEY,
      RETURN_URL_KEY,
      PROFILE_KEY,
      NAME_KEY,
      EMAIL_KEY,
      "HKE_ORDER_LOOKUP_MODE",
      "HKE_ORDER_LOOKUP_VALUE",
      "HKE_OTP_VERIFIED_PHONE",
      LEGACY_LOGGED_IN_KEY,
      LEGACY_PHONE_KEY,
      LEGACY_MOBILE_KEY,
      LEGACY_USER_PHONE_KEY,
      LEGACY_USER_NAME_KEY,
      LEGACY_USER_EMAIL_KEY
    ].forEach(function (key) {
      window.localStorage.removeItem(key);
    });
    pendingAction = null;
    syncProtectedPageState();
  }

  function goAfterLogin(fallbackUrl) {
    if (pendingAction && typeof pendingAction.onSuccess === "function") {
      var action = pendingAction;
      pendingAction = null;
      action.onSuccess();
      return;
    }
    var target = resolveReturnUrl(consumeReturnUrl()) || resolveReturnUrl(fallbackUrl) || "my-orders.html";
    window.location.href = target;
  }

  function setPhoneFieldValue() {
    var phone = window.localStorage.getItem(PHONE_KEY);
    if (!phone) return;

    [
      "phone",
      "customerPhone",
      "otpPhone"
    ].forEach(function (id) {
      var el = document.getElementById(id);
      if (el && !String(el.value || "").trim()) {
        el.value = phone;
      }
    });

    var profile = getStoredCustomerProfile();
    [
      ["name", profile.name || window.localStorage.getItem(NAME_KEY) || window.localStorage.getItem(LEGACY_USER_NAME_KEY) || ""],
      ["customerName", profile.name || window.localStorage.getItem(NAME_KEY) || window.localStorage.getItem(LEGACY_USER_NAME_KEY) || ""],
      ["email", profile.email || window.localStorage.getItem(EMAIL_KEY) || window.localStorage.getItem(LEGACY_USER_EMAIL_KEY) || ""],
      ["customerEmail", profile.email || window.localStorage.getItem(EMAIL_KEY) || window.localStorage.getItem(LEGACY_USER_EMAIL_KEY) || ""]
    ].forEach(function (pair) {
      var el = document.getElementById(pair[0]);
      if (el && !String(el.value || "").trim() && pair[1]) {
        el.value = pair[1];
      }
    });
  }

  function ensureOverlayStyle() {
    if (overlayStyleInjected) return;
    overlayStyleInjected = true;

    var styleEl = document.createElement("style");
    styleEl.id = "hkeAuthGateStyles";
    styleEl.textContent = [
      "#hkeAuthBlocker {",
      "  position: fixed;",
      "  inset: 0;",
      "  z-index: 1050;",
      "  display: none;",
      "  align-items: center;",
      "  justify-content: center;",
      "  padding: 24px;",
      "  background: rgba(4, 8, 14, 0.58);",
      "  backdrop-filter: blur(8px);",
      "}",
      "#hkeAuthBlocker.is-open { display: flex; }",
      "#hkeAuthBlocker .hke-auth-card {",
      "  width: min(420px, 92vw);",
      "  border-radius: 24px;",
      "  padding: 24px;",
      "  background: rgba(10, 12, 16, 0.94);",
      "  border: 1px solid rgba(255,255,255,.12);",
      "  box-shadow: 0 24px 60px rgba(0,0,0,.42);",
      "  color: #EAEFF7;",
      "  text-align: center;",
      "}",
      "#hkeAuthBlocker .hke-auth-title {",
      "  margin: 0 0 10px;",
      "  font-size: 1.35rem;",
      "  font-weight: 800;",
      "}",
      "#hkeAuthBlocker .hke-auth-copy {",
      "  margin: 0;",
      "  color: rgba(234,239,247,.76);",
      "  line-height: 1.7;",
      "}",
      "#hkeAuthBlocker .hke-auth-actions {",
      "  margin-top: 16px;",
      "  display: flex;",
      "  justify-content: center;",
      "  gap: 10px;",
      "  flex-wrap: wrap;",
      "}",
      "#hkeAuthBlocker .hke-auth-login-btn {",
      "  border: none;",
      "  border-radius: 999px;",
      "  padding: 12px 20px;",
      "  font-weight: 800;",
      "  color: #111;",
      "  background: linear-gradient(135deg, #D9B25F, #F0D08A);",
      "}",
      "#hkeAuthGateModal.manual-open {",
      "  display: block;",
      "  background: rgba(4,8,14,.62);",
      "}",
      "#hkeAuthGateModal.manual-open .modal-dialog {",
      "  margin-top: min(14vh, 88px);",
      "}",
      "body.hke-auth-modal-open { overflow: hidden; }"
    ].join("");
    document.head.appendChild(styleEl);
  }

  function ensureAuthOverlay() {
    ensureOverlayStyle();
    if (authOverlayEl) return authOverlayEl;

    authOverlayEl = document.getElementById("hkeAuthBlocker");
    if (authOverlayEl) return authOverlayEl;

    authOverlayEl = document.createElement("div");
    authOverlayEl.id = "hkeAuthBlocker";
    authOverlayEl.setAttribute("aria-hidden", "true");
    authOverlayEl.innerHTML = [
      '<div class="hke-auth-card" role="dialog" aria-modal="true" aria-labelledby="hkeAuthBlockerTitle">',
      '  <div class="hke-auth-title" id="hkeAuthBlockerTitle">Please login to continue</div>',
      '  <p class="hke-auth-copy">Verify your mobile number with OTP to use booking, itinerary, payment, and customer trip features.</p>',
      '  <div class="hke-auth-actions">',
      '    <button type="button" class="hke-auth-login-btn" data-hke-auth-overlay-login="true">Login with OTP</button>',
      "  </div>",
      "</div>"
    ].join("");
    document.body.appendChild(authOverlayEl);

    var loginBtn = authOverlayEl.querySelector("[data-hke-auth-overlay-login='true']");
    if (loginBtn) {
      loginBtn.addEventListener("click", function () {
        openLoginModal(getCurrentRelativeUrl());
      });
    }

    return authOverlayEl;
  }

  function isProtectedPage(pageName) {
    return !!PROTECTED_PAGES[pageName || getPageName()];
  }

  function showAuthOverlay() {
    var overlay = ensureAuthOverlay();
    overlay.classList.add("is-open");
    overlay.setAttribute("aria-hidden", "false");
  }

  function hideAuthOverlay() {
    var overlay = ensureAuthOverlay();
    overlay.classList.remove("is-open");
    overlay.setAttribute("aria-hidden", "true");
  }

  function syncProtectedPageState() {
    if (!isProtectedPage()) return;
    if (isLoggedIn()) {
      hideAuthOverlay();
      return;
    }
    showAuthOverlay();
  }

  function getModalRefs() {
    var existingModal = document.getElementById("otpLoginModal");
    if (existingModal) {
      return {
        rootEl: existingModal,
        modalEl: existingModal,
        phoneEl: document.getElementById("otpPhone"),
        otpEl: document.getElementById("otpCode"),
        sendBtn: document.getElementById("sendOtpBtn"),
        verifyBtn: document.getElementById("verifyOtpBtn"),
        msgEl: document.getElementById("otpMsg"),
        ordersEl: existingModal.querySelector("[data-hke-open-orders], a[href='my-orders.html']")
      };
    }

    var injected = document.getElementById("hkeAuthGateModal");
    if (!injected) {
      ensureOverlayStyle();
      injected = document.createElement("div");
      injected.innerHTML = [
        '<div class="modal fade" id="hkeAuthGateModal" tabindex="-1" aria-hidden="true">',
        '  <div class="modal-dialog modal-dialog-centered">',
        '    <div class="modal-content" style="background:rgba(10,12,16,.96); border:1px solid rgba(255,255,255,.12); border-radius:22px; color:#EAEFF7;">',
        '      <div class="modal-header" style="border-bottom:1px solid rgba(255,255,255,.08);">',
        '        <div>',
        '          <h5 class="modal-title" style="font-weight:800;">Login with Mobile OTP</h5>',
        '          <div style="color:rgba(234,239,247,.72); font-size:.95rem;">Access your booking, payment, and trip tools securely.</div>',
        "        </div>",
        '        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>',
        "      </div>",
        '      <div class="modal-body">',
        '        <div class="mb-3">',
        '          <label for="hkeAuthPhone" style="font-weight:700; margin-bottom:8px;">Mobile Number</label>',
        '          <input id="hkeAuthPhone" class="form-control" type="tel" placeholder="Enter 10-digit mobile number" maxlength="10" style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#fff; border-radius:16px; padding:14px;">',
        "        </div>",
        '        <div class="d-grid gap-2">',
        '          <button id="hkeAuthSendOtpBtn" type="button" class="btn btn-warning" style="border:none; border-radius:999px; min-height:48px; font-weight:800;">Send OTP</button>',
        "        </div>",
        '        <div class="mb-3 mt-3">',
        '          <label for="hkeAuthOtpCode" style="font-weight:700; margin-bottom:8px;">OTP</label>',
        '          <input id="hkeAuthOtpCode" class="form-control" type="text" placeholder="Enter OTP" maxlength="6" style="background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); color:#fff; border-radius:16px; padding:14px;">',
        "        </div>",
        '        <div style="color:rgba(234,239,247,.72); font-size:.9rem; line-height:1.6; margin-bottom:14px;">By logging in, you agree to be contacted by Himalayan Kerala Expeditions for trip assistance, booking updates, and travel offers.</div>',
        '        <div class="d-grid gap-2">',
        '          <button id="hkeAuthVerifyOtpBtn" type="button" class="btn" style="border-radius:999px; min-height:48px; font-weight:800; border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.08); color:#EAEFF7;">Verify OTP</button>',
        '          <a href="my-orders.html" data-hke-open-orders="true" class="btn" style="border-radius:999px; min-height:48px; font-weight:800; border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.08); color:#EAEFF7; text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">Open My Booking Page</a>',
        "        </div>",
        '        <div id="hkeAuthOtpMsg" style="margin-top:14px; font-weight:700;"></div>',
        "      </div>",
        "    </div>",
        "  </div>",
        "</div>"
      ].join("");
      document.body.appendChild(injected.firstChild);
    }

    return {
      rootEl: document.getElementById("hkeAuthGateModal"),
      modalEl: document.getElementById("hkeAuthGateModal"),
      phoneEl: document.getElementById("hkeAuthPhone"),
      otpEl: document.getElementById("hkeAuthOtpCode"),
      sendBtn: document.getElementById("hkeAuthSendOtpBtn"),
      verifyBtn: document.getElementById("hkeAuthVerifyOtpBtn"),
      msgEl: document.getElementById("hkeAuthOtpMsg"),
      ordersEl: document.querySelector("[data-hke-open-orders='true']")
    };
  }

  function getStandaloneLoginRefs() {
    var phoneEl = document.getElementById("phone");
    var sendBtn = document.getElementById("sendOtpBtn");
    var verifyBtn = document.getElementById("verifyOtpBtn");
    if (!phoneEl || !sendBtn || !verifyBtn) return null;

    return {
      rootEl: document.querySelector(".card") || document.body,
      modalEl: null,
      phoneEl: phoneEl,
      otpEl: document.getElementById("otp"),
      sendBtn: sendBtn,
      verifyBtn: verifyBtn,
      msgEl: document.getElementById("msg"),
      ordersEl: document.getElementById("openOrdersBtn"),
      successRedirectUrl: "my-orders.html"
    };
  }

  function showModalElement(modalEl) {
    if (!modalEl) return;
    if (window.bootstrap && window.bootstrap.Modal) {
      window.bootstrap.Modal.getOrCreateInstance(modalEl).show();
      return;
    }
    modalEl.style.display = "block";
    modalEl.removeAttribute("aria-hidden");
    modalEl.setAttribute("aria-modal", "true");
    modalEl.classList.add("show", "manual-open");
    document.body.classList.add("hke-auth-modal-open");
  }

  function hideModalElement(modalEl) {
    if (!modalEl) return;
    if (window.bootstrap && window.bootstrap.Modal) {
      var instance = window.bootstrap.Modal.getInstance(modalEl);
      if (instance) {
        instance.hide();
        return;
      }
    }
    modalEl.classList.remove("show", "manual-open");
    modalEl.style.display = "none";
    modalEl.setAttribute("aria-hidden", "true");
    modalEl.removeAttribute("aria-modal");
    document.body.classList.remove("hke-auth-modal-open");
  }

  function setModalMsg(refs, text, isErr) {
    if (!refs.msgEl) return;
    if (refs.msgEl.classList && refs.msgEl.classList.contains("msg")) {
      refs.msgEl.className = text ? "msg " + (isErr ? "err" : "ok") : "msg";
    }
    refs.msgEl.textContent = text;
    refs.msgEl.style.color = isErr ? "#ffd2d2" : "#d8ffe7";
  }

  function setButtonsBusy(refs, disabled, labelMap) {
    [
      [refs.sendBtn, "Send OTP"],
      [refs.verifyBtn, "Verify OTP"]
    ].forEach(function (pair) {
      var button = pair[0];
      var defaultText = pair[1];
      if (!button) return;
      button.disabled = disabled;
      button.textContent = (labelMap && labelMap[button.id]) || defaultText;
    });
  }

  function getResponseText(data) {
    if (!data) return "";
    if (typeof data.detail === "string" && data.detail.trim()) return data.detail.trim();
    if (typeof data.message === "string" && data.message.trim()) return data.message.trim();
    if (typeof data.error === "string" && data.error.trim()) return data.error.trim();
    if (typeof data.raw === "string" && data.raw.trim()) return data.raw.trim();
    return "";
  }

  async function postJSON(url, body) {
    var controller = new AbortController();
    var timeoutId = window.setTimeout(function () {
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
    var res;

    try {
      res = await window.fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal
      });
    } catch (err) {
      if (err && err.name === "AbortError") {
        throw new Error("Server temporarily unavailable. Please try later.");
      }
      throw new Error("Server temporarily unavailable. Please try later.");
    } finally {
      window.clearTimeout(timeoutId);
    }

    var txt = await res.text();
    var data = null;

    try {
      data = JSON.parse(txt);
    } catch (err) {
      data = { raw: txt };
    }

    return {
      res: res,
      data: data,
      text: txt
    };
  }

  function bindModal(refs) {
    var bindRoot = refs && (refs.rootEl || refs.modalEl);
    if (!bindRoot || bindRoot.dataset.hkeAuthBound === "true") return;
    bindRoot.dataset.hkeAuthBound = "true";

    if (refs.sendBtn) {
      refs.sendBtn.addEventListener("click", async function () {
        try {
          setButtonsBusy(refs, true, {
            sendOtpBtn: "Sending OTP...",
            hkeAuthSendOtpBtn: "Sending OTP..."
          });

          var phone = normalizePhone(refs.phoneEl && refs.phoneEl.value);
          if (refs.phoneEl) refs.phoneEl.value = phone;

          if (phone.length !== 10) {
            setModalMsg(refs, "Please enter a valid 10-digit mobile number.", true);
            return;
          }

          setModalMsg(refs, "Sending OTP...", false);
          var sendResult = await postJSON(SEND_OTP_API, { phone: phone, mobile: phone });
          var sendData = sendResult.data || {};
          var sendMessage = getResponseText(sendData);
          console.log("OTP send response", sendResult.res.status, sendData);
          var sendSucceeded =
            sendResult.res.ok === true ||
            sendData.ok === true ||
            /otp sent/i.test(sendMessage);

          if (!sendSucceeded) {
            throw new Error(sendMessage || "Unable to send OTP. Please try again later.");
          }

          setModalMsg(refs, "OTP sent successfully. Please enter the OTP.", false);
          if (refs.verifyBtn) refs.verifyBtn.disabled = false;
          if (refs.otpEl && typeof refs.otpEl.focus === "function") refs.otpEl.focus();
        } catch (err) {
          setModalMsg(refs, err.message || "Unable to send OTP. Please try again later.", true);
        } finally {
          setButtonsBusy(refs, false);
        }
      });
    }

    if (refs.verifyBtn) {
      refs.verifyBtn.addEventListener("click", async function () {
        try {
          setButtonsBusy(refs, true, {
            verifyOtpBtn: "Verifying OTP...",
            hkeAuthVerifyOtpBtn: "Verifying OTP..."
          });

          var phone = normalizePhone(refs.phoneEl && refs.phoneEl.value);
          var otp = String(refs.otpEl && refs.otpEl.value || "").trim();
          if (refs.phoneEl) refs.phoneEl.value = phone;

          if (phone.length !== 10) {
            setModalMsg(refs, "Please enter a valid mobile number.", true);
            return;
          }
          if (!otp || otp.length < 4) {
            setModalMsg(refs, "Please enter a valid OTP.", true);
            return;
          }

          setModalMsg(refs, "Verifying OTP...", false);
          var verifyResult = await postJSON(VERIFY_OTP_API, { phone: phone, mobile: phone, otp: otp });
          var verifyData = verifyResult.data || {};
          var verifyMessage = getResponseText(verifyData);
          var verifySucceeded =
            (verifyResult.res.ok === true && verifyData.ok === true) ||
            verifyData.verified === true;

          if (!verifySucceeded) {
            throw new Error(verifyMessage || "OTP verification failed.");
          }

          saveCustomerLogin(phone);
          syncAuthUi();
          syncProtectedPageState();
          setModalMsg(refs, "Login successful. Continuing...", false);

          window.setTimeout(function () {
            hideModalElement(refs.modalEl);
            goAfterLogin(refs.successRedirectUrl || getCurrentRelativeUrl());
          }, 500);
        } catch (err) {
          setModalMsg(refs, err.message || "OTP verification failed.", true);
        } finally {
          setButtonsBusy(refs, false);
        }
      });
    }

    if (refs.ordersEl) {
      refs.ordersEl.addEventListener("click", function (event) {
        if (isLoggedIn()) {
          if (!refs.ordersEl.getAttribute("href")) {
            window.location.href = "my-orders.html";
            event.preventDefault();
          }
          return;
        }
        event.preventDefault();
        setReturnUrl("my-orders.html");
        setModalMsg(refs, "Please verify OTP first.", true);
      });
    }
  }

  function openLoginModal(returnUrl) {
    if (returnUrl) setReturnUrl(returnUrl);

    var refs = getModalRefs();
    bindModal(refs);
    setPhoneFieldValue();
    syncProtectedPageState();

    var savedPhone = window.localStorage.getItem(PHONE_KEY);
    if (refs.phoneEl && savedPhone && !String(refs.phoneEl.value || "").trim()) {
      refs.phoneEl.value = savedPhone;
    }

    setModalMsg(refs, "", false);
    showModalElement(refs.modalEl);
  }

  function requireLogin(opts) {
    var options = opts || {};
    var returnUrl = resolveReturnUrl(options.returnUrl) || getCurrentRelativeUrl();

    if (isLoggedIn()) {
      if (typeof options.onSuccess === "function") {
        options.onSuccess();
        return true;
      }
      return true;
    }

    pendingAction = {
      onSuccess: typeof options.onSuccess === "function" ? options.onSuccess : null,
      returnUrl: returnUrl
    };

    openLoginModal(returnUrl);
    return false;
  }

  function getElementReturnUrl(el) {
    var explicit = el.getAttribute("data-hke-return-url");
    if (explicit) return resolveReturnUrl(explicit);

    var href = el.getAttribute("href");
    if (href && href !== "#" && href.indexOf("javascript:") !== 0) {
      return resolveReturnUrl(href);
    }

    return getCurrentRelativeUrl();
  }

  function wireProtectedElements() {
    document.querySelectorAll("[data-hke-requires-login]").forEach(function (el) {
      if (el.dataset.hkeAuthClickBound === "true") return;
      el.dataset.hkeAuthClickBound = "true";

      el.addEventListener("click", function (event) {
        if (isLoggedIn()) return;

        event.preventDefault();
        event.stopPropagation();
        if (typeof event.stopImmediatePropagation === "function") {
          event.stopImmediatePropagation();
        }

        requireLogin({
          returnUrl: getElementReturnUrl(el),
          useRedirect: el.getAttribute("data-hke-login-mode") === "redirect"
        });
      }, true);
    });
  }

  function wireLoginButtons() {
    document.querySelectorAll("[data-hke-open-login]").forEach(function (el) {
      if (el.dataset.hkeOpenLoginBound === "true") return;
      el.dataset.hkeOpenLoginBound = "true";

      el.addEventListener("click", function (event) {
        event.preventDefault();
        if (isLoggedIn()) {
          clearCustomerLogin();
          syncAuthUi();
          return;
        }
        openLoginModal(getCurrentRelativeUrl());
      });
    });
  }

  function syncAuthUi() {
    var loggedIn = isLoggedIn();

    document.querySelectorAll("[data-hke-open-login]").forEach(function (el) {
      var textEl = el.querySelector(".login-text");
      if (textEl) {
        textEl.textContent = loggedIn ? "Logout" : "Login";
      } else {
        el.textContent = loggedIn ? "Logout" : "Login";
      }
      el.setAttribute("aria-label", loggedIn ? "Logout" : "Login with OTP");
      el.setAttribute("title", loggedIn ? "Logout" : "Login with OTP");
    });

    document.querySelectorAll("a[href='login.html']").forEach(function (el) {
      if (el.dataset.hkeAuthLinkBound !== "true") {
        el.dataset.hkeAuthLinkBound = "true";
        el.dataset.hkeDefaultText = el.textContent;
        el.addEventListener("click", function (event) {
          if (!isLoggedIn()) return;
          event.preventDefault();
          clearCustomerLogin();
          syncAuthUi();
          if (getPageName() !== "login.html") {
            window.location.reload();
          }
        });
      }
      el.textContent = loggedIn ? "Logout" : (el.dataset.hkeDefaultText || "Login");
    });
  }

  window.HKEAuthGate = {
    LOGGED_IN_KEY: LOGGED_IN_KEY,
    PHONE_KEY: PHONE_KEY,
    LOGIN_TIME_KEY: LOGIN_TIME_KEY,
    RETURN_URL_KEY: RETURN_URL_KEY,
    isLoggedIn: isLoggedIn,
    normalizePhone: normalizePhone,
    saveCustomerLogin: saveCustomerLogin,
    clearCustomerLogin: clearCustomerLogin,
    saveCustomerProfile: saveCustomerProfile,
    getStoredCustomerProfile: getStoredCustomerProfile,
    goAfterLogin: goAfterLogin,
    openLoginModal: openLoginModal,
    requireLogin: requireLogin,
    syncAuthUi: syncAuthUi,
    getStoredReturnUrl: getStoredReturnUrl,
    syncProtectedPageState: syncProtectedPageState
  };

  document.addEventListener("DOMContentLoaded", function () {
    wireProtectedElements();
    wireLoginButtons();
    setPhoneFieldValue();
    syncAuthUi();
    ensureAuthOverlay();
    syncProtectedPageState();

    var existingModal = document.getElementById("otpLoginModal");
    if (existingModal) {
      bindModal(getModalRefs());
    }

    var standaloneLoginRefs = getStandaloneLoginRefs();
    if (standaloneLoginRefs) {
      bindModal(standaloneLoginRefs);
    }

    if (isProtectedPage() && !isLoggedIn() && getPageName() !== "login.html") {
      window.setTimeout(function () {
        openLoginModal(getCurrentRelativeUrl());
      }, 120);
    }
  });
})(window, document);
