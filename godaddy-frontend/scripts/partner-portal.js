(() => {
  "use strict";

  const API_BASE = "https://hke-backend.onrender.com";
  const SESSION_KEY = "HKE_PARTNER_SESSION";
  const SERVER_DOWN_MESSAGE = "Server temporarily unavailable. Please try later.";
  const LOGIN_PAGE = "partner-login.html";
  const DASHBOARD_PAGE = "partner-dashboard.html";
  const REQUEST_TIMEOUT_MS = 15000;

  const page = document.body?.dataset?.page || "";

  const $ = (id) => document.getElementById(id);

  function setMessage(el, text = "", type = "") {
    if (!el) return;
    el.className = "msg-box";
    el.textContent = text;
    if (!text) return;
    el.classList.add("show");
    if (type) el.classList.add(type);
  }

  function normalizeMobile(value = "") {
    let digits = String(value).replace(/\D/g, "");
    if (digits.length === 12 && digits.startsWith("91")) digits = digits.slice(2);
    if (digits.length > 10) digits = digits.slice(-10);
    return digits;
  }

  function formatDate(value = "") {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric"
    });
  }

  function addTwoMonths(dateValue = "") {
    if (!dateValue) return "";
    const date = new Date(dateValue);
    if (Number.isNaN(date.getTime())) return "";
    const end = new Date(date);
    end.setMonth(end.getMonth() + 2);
    return end.toISOString().split("T")[0];
  }

  function getSession() {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  function saveSession(session) {
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  }

  function clearSession() {
    localStorage.removeItem(SESSION_KEY);
  }

  function extractSession(data, fallbackPartner = null) {
    const token =
      data?.token ||
      data?.access_token ||
      data?.session_token ||
      data?.session?.token ||
      "";

    const partner =
      data?.partner ||
      data?.user ||
      data?.data?.partner ||
      data?.session?.partner ||
      fallbackPartner;

    return token && partner ? { token, partner } : null;
  }

  function toggleButtons(buttons, disabled, loadingText = "") {
    buttons.filter(Boolean).forEach((button) => {
      if (!button.dataset.defaultText) {
        button.dataset.defaultText = button.textContent;
      }
      button.disabled = disabled;
      if (disabled && loadingText) {
        button.textContent = loadingText;
      } else if (!disabled) {
        button.textContent = button.dataset.defaultText;
      }
    });
  }

  async function requestJSON(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let response;

    try {
      response = await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new Error(SERVER_DOWN_MESSAGE);
      }
      throw new Error(SERVER_DOWN_MESSAGE);
    } finally {
      window.clearTimeout(timeoutId);
    }

    const raw = await response.text();
    let data = null;

    try {
      data = raw ? JSON.parse(raw) : null;
    } catch {
      data = { raw };
    }

    if (!response.ok) {
      if ([502, 503, 504].includes(response.status)) {
        throw new Error(SERVER_DOWN_MESSAGE);
      }

      const detail =
        data?.detail?.[0]?.msg ||
        data?.detail ||
        data?.message ||
        data?.error ||
        data?.raw ||
        `HTTP ${response.status}`;

      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    return data;
  }

  async function api(path, { method = "GET", body, auth = false } = {}) {
    const headers = { "Content-Type": "application/json" };
    const session = getSession();

    if (auth && session?.token) {
      headers.Authorization = `Bearer ${session.token}`;
    }

    return requestJSON(`${API_BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined
    });
  }

  function registerRateFieldsMarkup(type = "") {
    if (type === "Hotel") {
      return `
        <div class="field">
          <label for="hotelCategory">Hotel Category</label>
          <select id="hotelCategory" name="hotelCategory" required>
            <option value="Budget">Budget</option>
            <option value="Standard" selected>Standard</option>
            <option value="Premium">Premium</option>
            <option value="Luxury">Luxury</option>
          </select>
        </div>
        <div class="field">
          <label for="roomType">Room Type</label>
          <input id="roomType" name="roomType" type="text" placeholder="Deluxe Room" required />
        </div>
        <div class="field">
          <label for="pricePerNight">Price Per Night</label>
          <input id="pricePerNight" name="pricePerNight" type="number" min="0" step="0.01" placeholder="4500" required />
        </div>
        <div class="field">
          <label for="mealPlan">Meal Plan</label>
          <input id="mealPlan" name="mealPlan" type="text" placeholder="Breakfast Included" required />
        </div>
        <div class="field">
          <label for="totalRooms">Total Rooms</label>
          <input id="totalRooms" name="totalRooms" type="number" min="1" step="1" placeholder="12" required />
        </div>
      `;
    }

    if (type === "Driver") {
      return `
        <div class="field">
          <label for="vehicleType">Vehicle Type</label>
          <select id="vehicleType" name="vehicleType" required>
            <option value="Sedan">Sedan</option>
            <option value="SUV" selected>SUV</option>
            <option value="Crysta">Crysta</option>
            <option value="Tempo Traveller">Tempo Traveller</option>
            <option value="Urbania">Urbania</option>
          </select>
        </div>
        <div class="field">
          <label for="vehicleNumber">Vehicle Number</label>
          <input id="vehicleNumber" name="vehicleNumber" type="text" placeholder="JK01AB1234" required />
        </div>
        <div class="field">
          <label for="perDayRate">Per Day Rate</label>
          <input id="perDayRate" name="perDayRate" type="number" min="0" step="0.01" placeholder="6500" required />
        </div>
        <div class="field">
          <label for="perKmRate">Per KM Rate</label>
          <input id="perKmRate" name="perKmRate" type="number" min="0" step="0.01" placeholder="18" required />
        </div>
        <div class="field">
          <label for="driverAllowance">Driver Allowance</label>
          <input id="driverAllowance" name="driverAllowance" type="number" min="0" step="0.01" placeholder="800" required />
        </div>
        <div class="field full">
          <label for="availableRoutes">Available Routes</label>
          <input id="availableRoutes" name="availableRoutes" type="text" placeholder="Srinagar - Gulmarg, Pahalgam Circuit" required />
        </div>
      `;
    }

    if (type === "Guide") {
      return `
        <div class="field">
          <label for="language">Language</label>
          <input id="language" name="language" type="text" placeholder="English, Hindi" required />
        </div>
        <div class="field">
          <label for="guidePerDayRate">Per Day Rate</label>
          <input id="guidePerDayRate" name="guidePerDayRate" type="number" min="0" step="0.01" placeholder="3000" required />
        </div>
        <div class="field full">
          <label for="specialty">Specialty</label>
          <input id="specialty" name="specialty" type="text" placeholder="Pilgrimage, local heritage, trekking support" required />
        </div>
      `;
    }

    return "";
  }

  function buildRatePayload(form, session, editingRateId = "") {
    const fd = new FormData(form);
    const partnerType = fd.get("partnerType") || session.partner.partner_type || "";
    const payload = {
      partner_type: String(partnerType || ""),
      business_name: String(fd.get("businessName") || "").trim(),
      location: String(fd.get("location") || "").trim(),
      state: String(fd.get("state") || "").trim(),
      service_area: String(fd.get("serviceArea") || "").trim(),
      available_from: String(fd.get("availableFrom") || ""),
      available_to: String(fd.get("availableTo") || ""),
      notes: String(fd.get("notes") || "").trim(),
      available: true
    };

    if (partnerType === "Hotel") {
      Object.assign(payload, {
        hotel_category: String(fd.get("hotelCategory") || ""),
        room_type: String(fd.get("roomType") || "").trim(),
        price_per_night: Number(fd.get("pricePerNight") || 0),
        meal_plan: String(fd.get("mealPlan") || "").trim(),
        total_rooms: Number(fd.get("totalRooms") || 0)
      });
    }

    if (partnerType === "Driver") {
      Object.assign(payload, {
        vehicle_type: String(fd.get("vehicleType") || ""),
        vehicle_number: String(fd.get("vehicleNumber") || "").trim().toUpperCase(),
        per_day_rate: Number(fd.get("perDayRate") || 0),
        per_km_rate: Number(fd.get("perKmRate") || 0),
        driver_allowance: Number(fd.get("driverAllowance") || 0),
        available_routes: String(fd.get("availableRoutes") || "").trim()
      });
    }

    if (partnerType === "Guide") {
      Object.assign(payload, {
        language: String(fd.get("language") || "").trim(),
        per_day_rate: Number(fd.get("guidePerDayRate") || 0),
        specialty: String(fd.get("specialty") || "").trim()
      });
    }

    if (editingRateId) {
      payload.id = editingRateId;
    }

    return payload;
  }

  function validateRatePayload(payload) {
    if (!payload.location) return "Please enter location or city.";
    if (!payload.state) return "Please enter state.";
    if (!payload.service_area) return "Please enter service area.";
    if (!payload.available_from) return "Please select available from date.";
    if (!payload.available_to) return "Please select available to date.";

    const start = new Date(payload.available_from);
    const end = new Date(payload.available_to);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
      return "Please select valid availability dates.";
    }
    if (end < start) return "Available to date must be after available from date.";

    const maxEnd = new Date(addTwoMonths(payload.available_from));
    if (maxEnd && end > maxEnd) return "Available to date cannot exceed 2 months from start date.";

    if (payload.partner_type === "Hotel" && payload.price_per_night <= 0) {
      return "Please enter a valid price per night.";
    }
    if (payload.partner_type === "Driver" && payload.per_day_rate <= 0) {
      return "Please enter a valid per day rate.";
    }
    if (payload.partner_type === "Guide" && payload.per_day_rate <= 0) {
      return "Please enter a valid per day rate.";
    }

    return "";
  }

  function renderRateTypeFields(type = "", rate = null) {
    const wrap = $("rateTypeFields");
    if (!wrap) return;

    wrap.innerHTML = registerRateFieldsMarkup(type);

    if (!rate) return;

    const mappings = {
      hotelCategory: rate.hotel_category,
      roomType: rate.room_type,
      pricePerNight: rate.price_per_night,
      mealPlan: rate.meal_plan,
      totalRooms: rate.total_rooms,
      vehicleType: rate.vehicle_type,
      vehicleNumber: rate.vehicle_number,
      perDayRate: rate.per_day_rate,
      perKmRate: rate.per_km_rate,
      driverAllowance: rate.driver_allowance,
      availableRoutes: rate.available_routes,
      language: rate.language,
      guidePerDayRate: rate.per_day_rate,
      specialty: rate.specialty
    };

    Object.entries(mappings).forEach(([id, value]) => {
      const el = $(id);
      if (el && value !== undefined && value !== null) el.value = value;
    });
  }

  function summarizeRate(rate) {
    if (rate.partner_type === "Hotel") {
      return `${rate.hotel_category || "-"} | ${rate.room_type || "-"} | Rs ${rate.price_per_night ?? "-"}/night`;
    }
    if (rate.partner_type === "Driver") {
      return `${rate.vehicle_type || "-"} | Rs ${rate.per_day_rate ?? "-"}/day | Rs ${rate.per_km_rate ?? "-"}/km`;
    }
    if (rate.partner_type === "Guide") {
      return `${rate.language || "-"} | ${rate.specialty || "-"} | Rs ${rate.per_day_rate ?? "-"}/day`;
    }
    return rate.partner_type || "-";
  }

  function renderRates(rates, handlers) {
    const tableBody = $("ratesTableBody");
    const emptyState = $("ratesEmpty");
    if (!tableBody || !emptyState) return;

    if (!Array.isArray(rates) || !rates.length) {
      tableBody.innerHTML = "";
      emptyState.style.display = "block";
      return;
    }

    emptyState.style.display = "none";
    tableBody.innerHTML = rates.map((rate, index) => {
      const id = rate._id || rate.id || `${index}`;
      return `
        <tr data-rate-id="${id}">
          <td>
            <div class="rate-detail">
              <strong>${rate.business_name || "-"}</strong>
              <span class="rate-sub">${rate.location || "-"}, ${rate.state || "-"}</span>
              <span class="rate-sub">${rate.service_area || "-"}</span>
            </div>
          </td>
          <td>
            <div class="rate-detail">
              <strong>${rate.partner_type || "-"}</strong>
              <span class="rate-sub">${summarizeRate(rate)}</span>
            </div>
          </td>
          <td>
            <div class="rate-detail">
              <strong>${formatDate(rate.available_from)}</strong>
              <span class="rate-sub">to ${formatDate(rate.available_to)}</span>
            </div>
          </td>
          <td><span class="badge ${(rate.status || "pending").toLowerCase()}">${rate.status || "pending"}</span></td>
          <td>
            <div class="inline-actions">
              <button type="button" class="inline-btn" data-action="edit">Edit</button>
              <button type="button" class="inline-btn danger" data-action="delete">Delete</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");

    tableBody.querySelectorAll("tr").forEach((row) => {
      const rateId = row.dataset.rateId;
      row.querySelector('[data-action="edit"]')?.addEventListener("click", (event) => handlers.onEdit(rateId, event.currentTarget));
      row.querySelector('[data-action="delete"]')?.addEventListener("click", (event) => handlers.onDelete(rateId, event.currentTarget));
    });
  }

  function setupLoginPage() {
    const loginForm = $("loginForm");
    const registerForm = $("registerForm");
    const loginTab = $("loginTab");
    const registerTab = $("registerTab");
    const loginPanel = $("loginPanel");
    const registerPanel = $("registerPanel");
    const authMsg = $("authMsg");
    const registerSwitcher = $("registerSwitcher");
    const loginSwitcher = $("loginSwitcher");

    if (!loginForm || !registerForm || !loginPanel || !registerPanel) return;

    const existing = getSession();
    if (existing?.token && existing?.partner) {
      window.location.href = DASHBOARD_PAGE;
      return;
    }

    function toggle(mode) {
      const isLogin = mode === "login";
      loginPanel.style.display = isLogin ? "block" : "none";
      registerPanel.style.display = isLogin ? "none" : "block";
      loginTab?.classList.toggle("active", isLogin);
      registerTab?.classList.toggle("active", !isLogin);
      setMessage(authMsg);
    }

    loginTab?.addEventListener("click", () => toggle("login"));
    registerTab?.addEventListener("click", () => toggle("register"));
    registerSwitcher?.addEventListener("click", () => toggle("register"));
    loginSwitcher?.addEventListener("click", () => toggle("login"));

    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (loginForm.dataset.busy === "true") return;
      loginForm.dataset.busy = "true";
      toggleButtons([loginForm.querySelector('button[type="submit"]'), registerSwitcher], true, "Signing in...");
      setMessage(authMsg, "Signing in...");

      const mobile = normalizeMobile($("loginMobile")?.value || "");
      const password = String($("loginPassword")?.value || "");
      $("loginMobile").value = mobile;

      if (mobile.length !== 10) {
        setMessage(authMsg, "Please enter a valid 10-digit mobile number.", "err");
        loginForm.dataset.busy = "false";
        toggleButtons([loginForm.querySelector('button[type="submit"]'), registerSwitcher], false);
        return;
      }
      if (!password) {
        setMessage(authMsg, "Please enter password.", "err");
        loginForm.dataset.busy = "false";
        toggleButtons([loginForm.querySelector('button[type="submit"]'), registerSwitcher], false);
        return;
      }

      try {
        const data = await api("/api/partners/login", {
          method: "POST",
          body: { mobile, password }
        });

        const session = extractSession(data);
        if (!session) throw new Error("Login response did not include partner session data.");

        saveSession(session);
        setMessage(authMsg, "Login successful. Redirecting...", "ok");
        window.setTimeout(() => {
          window.location.href = DASHBOARD_PAGE;
        }, 500);
      } catch (error) {
        setMessage(authMsg, error.message || SERVER_DOWN_MESSAGE, "err");
      } finally {
        loginForm.dataset.busy = "false";
        toggleButtons([loginForm.querySelector('button[type="submit"]'), registerSwitcher], false);
      }
    });

    registerForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (registerForm.dataset.busy === "true") return;
      registerForm.dataset.busy = "true";
      toggleButtons([registerForm.querySelector('button[type="submit"]'), loginSwitcher], true, "Creating account...");
      setMessage(authMsg, "Creating partner account...");

      const payload = {
        partner_type: String($("partnerType")?.value || ""),
        business_name: String($("businessName")?.value || "").trim(),
        contact_person: String($("contactPerson")?.value || "").trim(),
        mobile: normalizeMobile($("registerMobile")?.value || ""),
        email: String($("registerEmail")?.value || "").trim(),
        password: String($("registerPassword")?.value || "")
      };

      $("registerMobile").value = payload.mobile;

      if (payload.mobile.length !== 10) {
        setMessage(authMsg, "Please enter a valid 10-digit mobile number.", "err");
        registerForm.dataset.busy = "false";
        toggleButtons([registerForm.querySelector('button[type="submit"]'), loginSwitcher], false);
        return;
      }

      try {
        const data = await api("/api/partners/register", {
          method: "POST",
          body: payload
        });

        let session = extractSession(data, {
          partner_type: payload.partner_type,
          business_name: payload.business_name,
          contact_person: payload.contact_person,
          mobile: payload.mobile,
          email: payload.email
        });

        if (!session) {
          const loginData = await api("/api/partners/login", {
            method: "POST",
            body: {
              mobile: payload.mobile,
              password: payload.password
            }
          });
          session = extractSession(loginData, {
            partner_type: payload.partner_type,
            business_name: payload.business_name,
            contact_person: payload.contact_person,
            mobile: payload.mobile,
            email: payload.email
          });
        }

        if (!session) throw new Error("Registration succeeded but automatic login was not available.");

        saveSession(session);
        setMessage(authMsg, "Registration successful. Redirecting...", "ok");
        window.setTimeout(() => {
          window.location.href = DASHBOARD_PAGE;
        }, 500);
      } catch (error) {
        setMessage(authMsg, error.message || SERVER_DOWN_MESSAGE, "err");
      } finally {
        registerForm.dataset.busy = "false";
        toggleButtons([registerForm.querySelector('button[type="submit"]'), loginSwitcher], false);
      }
    });
  }

  function setupDashboardPage() {
    const session = getSession();
    if (!session?.token || !session?.partner) {
      window.location.href = LOGIN_PAGE;
      return;
    }

    const partner = session.partner;
    const logoutBtn = $("logoutBtn");
    const rateForm = $("rateForm");
    const ratesMsg = $("ratesMsg");
    const ratesMeta = {
      partnerTypeText: $("partnerTypeText"),
      businessNameText: $("businessNameText"),
      contactPersonText: $("contactPersonText"),
      partnerMobileText: $("partnerMobileText"),
      partnerEmailText: $("partnerEmailText")
    };

    let currentEditId = "";
    let rates = [];

    function fillPartnerMeta() {
      if (ratesMeta.partnerTypeText) ratesMeta.partnerTypeText.textContent = partner.partner_type || "-";
      if (ratesMeta.businessNameText) ratesMeta.businessNameText.textContent = partner.business_name || "-";
      if (ratesMeta.contactPersonText) ratesMeta.contactPersonText.textContent = partner.contact_person || "-";
      if (ratesMeta.partnerMobileText) ratesMeta.partnerMobileText.textContent = partner.mobile || "-";
      if (ratesMeta.partnerEmailText) ratesMeta.partnerEmailText.textContent = partner.email || "-";

      const typeEl = $("partnerTypeRate");
      const businessEl = $("businessNameRate");
      if (typeEl) typeEl.value = partner.partner_type || "";
      if (businessEl) businessEl.value = partner.business_name || "";
      renderRateTypeFields(partner.partner_type || "");
    }

    function syncDateLimit() {
      const fromEl = $("availableFrom");
      const toEl = $("availableTo");
      if (!fromEl || !toEl) return;
      toEl.min = fromEl.value || "";
      toEl.max = addTwoMonths(fromEl.value || "");
      if (toEl.value && toEl.max && toEl.value > toEl.max) {
        toEl.value = toEl.max;
      }
    }

    function resetForm() {
      currentEditId = "";
      rateForm?.reset();
      if ($("partnerTypeRate")) $("partnerTypeRate").value = partner.partner_type || "";
      if ($("businessNameRate")) $("businessNameRate").value = partner.business_name || "";
      renderRateTypeFields(partner.partner_type || "");
      syncDateLimit();
      const submitText = $("saveRateText");
      if (submitText) submitText.textContent = "Save Rate";
      const cancelEditBtn = $("cancelEditBtn");
      if (cancelEditBtn) cancelEditBtn.style.display = "none";
    }

    function loadRateIntoForm(rateId) {
      const rate = rates.find((item) => String(item._id || item.id) === String(rateId));
      if (!rate || !rateForm) return;

      currentEditId = String(rate._id || rate.id);
      $("partnerTypeRate").value = rate.partner_type || partner.partner_type || "";
      $("businessNameRate").value = rate.business_name || partner.business_name || "";
      $("location").value = rate.location || "";
      $("state").value = rate.state || "";
      $("serviceArea").value = rate.service_area || "";
      $("availableFrom").value = rate.available_from || "";
      $("availableTo").value = rate.available_to || "";
      $("notes").value = rate.notes || "";

      renderRateTypeFields(rate.partner_type || partner.partner_type || "", rate);
      syncDateLimit();

      const submitText = $("saveRateText");
      if (submitText) submitText.textContent = "Update Rate";
      const cancelEditBtn = $("cancelEditBtn");
      if (cancelEditBtn) cancelEditBtn.style.display = "inline-flex";
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    async function fetchRates() {
      try {
        const data = await api("/api/partners/rates", { auth: true });
        rates = data?.rates || data?.data || data || [];

        renderRates(rates, {
          onEdit: loadRateIntoForm,
          onDelete: deleteRate
        });
      } catch (error) {
        if (/401|403/.test(error.message || "")) {
          clearSession();
          window.location.href = LOGIN_PAGE;
          return;
        }
        setMessage(ratesMsg, error.message || SERVER_DOWN_MESSAGE, "err");
      }
    }

    async function deleteRate(rateId, triggerButton) {
      const confirmed = window.confirm("Delete this rate?");
      if (!confirmed) return;

      toggleButtons([triggerButton], true, "Deleting...");
      setMessage(ratesMsg, "Deleting rate...");
      try {
        await api(`/api/partners/rates/${encodeURIComponent(rateId)}`, {
          method: "DELETE",
          auth: true
        });
        setMessage(ratesMsg, "Rate deleted successfully.", "ok");
        if (currentEditId === String(rateId)) resetForm();
        await fetchRates();
      } catch (error) {
        setMessage(ratesMsg, error.message || SERVER_DOWN_MESSAGE, "err");
      } finally {
        toggleButtons([triggerButton], false);
      }
    }

    logoutBtn?.addEventListener("click", () => {
      clearSession();
      window.location.href = LOGIN_PAGE;
    });

    $("availableFrom")?.addEventListener("change", syncDateLimit);
    $("cancelEditBtn")?.addEventListener("click", resetForm);

    rateForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (rateForm.dataset.busy === "true") return;
      rateForm.dataset.busy = "true";
      const submitBtn = rateForm.querySelector('button[type="submit"]');
      const cancelBtn = $("cancelEditBtn");
      toggleButtons([submitBtn, cancelBtn], true, currentEditId ? "Updating rate..." : "Saving rate...");
      setMessage(ratesMsg, currentEditId ? "Updating rate..." : "Saving rate...");

      const payload = buildRatePayload(rateForm, session, currentEditId);
      const validationError = validateRatePayload(payload);
      if (validationError) {
        setMessage(ratesMsg, validationError, "err");
        rateForm.dataset.busy = "false";
        toggleButtons([submitBtn, cancelBtn], false);
        return;
      }

      try {
        await api(
          currentEditId ? `/api/partners/rates/${encodeURIComponent(currentEditId)}` : "/api/partners/rates",
          {
            method: currentEditId ? "PUT" : "POST",
            body: payload,
            auth: true
          }
        );
        setMessage(ratesMsg, currentEditId ? "Rate updated successfully." : "Rate saved successfully.", "ok");
        resetForm();
        await fetchRates();
      } catch (error) {
        setMessage(ratesMsg, error.message || SERVER_DOWN_MESSAGE, "err");
      } finally {
        rateForm.dataset.busy = "false";
        toggleButtons([submitBtn, cancelBtn], false);
      }
    });

    fillPartnerMeta();
    resetForm();
    fetchRates();
  }

  if (page === "partner-login") setupLoginPage();
  if (page === "partner-dashboard") setupDashboardPage();
})();
