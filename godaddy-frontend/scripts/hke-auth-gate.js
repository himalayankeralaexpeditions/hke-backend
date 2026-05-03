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
    "finalize.html": true
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

  function redirectToLogin(returnUrl) {
    var target = resolveReturnUrl(returnUrl) || getCurrentRelativeUrl();
    setReturnUrl(target);
    window.location.href = "login.html?returnUrl=" + encodeURIComponent(target);
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
  }

  function goAfterLogin(fallbackUrl) {
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

  function getModalRefs() {
    var existingModal = document.getElementById("otpLoginModal");
    if (existingModal) {
      return {
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
      modalEl: document.getElementById("hkeAuthGateModal"),
      phoneEl: document.getElementById("hkeAuthPhone"),
      otpEl: document.getElementById("hkeAuthOtpCode"),
      sendBtn: document.getElementById("hkeAuthSendOtpBtn"),
      verifyBtn: document.getElementById("hkeAuthVerifyOtpBtn"),
      msgEl: document.getElementById("hkeAuthOtpMsg"),
      ordersEl: document.querySelector("[data-hke-open-orders='true']")
    };
  }

  function setModalMsg(refs, text, isErr) {
    if (!refs.msgEl) return;
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

    if (!res.ok) {
      var msg = (data && (data.detail || data.message)) || txt || "Request failed";
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }

    return data;
  }

  function bindModal(refs) {
    if (!refs.modalEl || refs.modalEl.dataset.hkeAuthBound === "true") return;
    refs.modalEl.dataset.hkeAuthBound = "true";

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
          await postJSON(SEND_OTP_API, { phone: phone, mobile: phone });
          setModalMsg(refs, "OTP sent successfully to your mobile number.", false);
        } catch (err) {
          setModalMsg(refs, err.message || "Unable to send OTP.", true);
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
          await postJSON(VERIFY_OTP_API, { phone: phone, mobile: phone, otp: otp });
          saveCustomerLogin(phone);
          setModalMsg(refs, "Login successful. Redirecting...", false);

          window.setTimeout(function () {
            if (window.bootstrap && window.bootstrap.Modal) {
              var instance = window.bootstrap.Modal.getOrCreateInstance(refs.modalEl);
              instance.hide();
            }
            goAfterLogin(getCurrentRelativeUrl());
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
        if (isLoggedIn()) return;
        event.preventDefault();
        setReturnUrl("my-orders.html");
        setModalMsg(refs, "Please verify OTP first.", true);
      });
    }
  }

  function openLoginModal(returnUrl) {
    if (returnUrl) setReturnUrl(returnUrl);

    if (!window.bootstrap || !window.bootstrap.Modal) {
      redirectToLogin(returnUrl || getCurrentRelativeUrl());
      return;
    }

    var refs = getModalRefs();
    bindModal(refs);
    setPhoneFieldValue();

    var savedPhone = window.localStorage.getItem(PHONE_KEY);
    if (refs.phoneEl && savedPhone && !String(refs.phoneEl.value || "").trim()) {
      refs.phoneEl.value = savedPhone;
    }

    setModalMsg(refs, "", false);
    window.bootstrap.Modal.getOrCreateInstance(refs.modalEl).show();
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

    if (options.useRedirect) {
      redirectToLogin(returnUrl);
    } else {
      openLoginModal(returnUrl);
    }
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

  if (PROTECTED_PAGES[getPageName()] && !isLoggedIn() && getPageName() !== "login.html") {
    redirectToLogin(getCurrentRelativeUrl());
    return;
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
    getStoredReturnUrl: getStoredReturnUrl
  };

  document.addEventListener("DOMContentLoaded", function () {
    wireProtectedElements();
    wireLoginButtons();
    setPhoneFieldValue();
    syncAuthUi();

    var existingModal = document.getElementById("otpLoginModal");
    if (existingModal) {
      bindModal(getModalRefs());
    }
  });
})(window, document);
