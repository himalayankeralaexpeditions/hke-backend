(() => {
  "use strict";

  const API_BASE = "https://hke-backend.onrender.com";
  const API_GENERATE = `${API_BASE}/api/ai/itinerary`;
  const API_WHATSAPP_LOG = `${API_BASE}/api/whatsapp/log`;
  const COMPANY_PHONE = "919797294747";
  const RESULT_PAGE = "itinerary-result.html";
  const REQUEST_TIMEOUT_MS = 20000;
  const CUSTOMER_PROFILE_KEY = "HKE_CUSTOMER_PROFILE";

  const $ = (id) => document.getElementById(id);

  const destinationEl = $("destination");
  const nameEl = $("name");
  const emailEl = $("email");
  const phoneEl = $("phone");
  const fromLocationEl = $("fromLocation");
  const endPointEl = $("endPoint");
  const startDateEl = $("startDate");
  const daysEl = $("days");
  const endDateEl = $("endDate");
  const travellersEl = $("travellers");
  const roomsEl = $("rooms");
  const budgetEl = $("budget");
  const travelTypeEl = $("travelType");
  const hotelClassEl = $("hotelClass");
  const vehicleEl = $("vehicle");
  const guideEl = $("guide");
  const needFoodEl = $("needFood");
  const foodPreferenceEl = $("foodPreference");
  const notesEl = $("notes");

  const form = $("plannerForm");
  const msgEl = $("msg");
  const outputEl = $("output");
  const itMetaEl = $("itMeta");
  const extraInfoEl = $("extraInfo");
  const itCardsEl = $("itCards");
  const postActionsEl = $("postActions");

  const generateBtn = $("generateBtn");
  const resetBtn = $("resetBtn");

  const placesValueEl = $("placesValue");
  const chipsEl = $("chips");
  const placesErrEl = $("placesErr");
  const placesSearchEl = $("placesSearch");
  const placesHintEl = $("placesHint");
  const placesListEl = $("placesList");
  const placesClearBtn = $("placesClearBtn");

  const careBtn = $("careBtn");
  const messageBtn = $("messageBtn");
  const editBtn = $("editBtn");

  const loadingOverlayEl = $("loadingOverlay");

  let selectedPlaces = [];
  let isGenerating = false;

  const PLACES_BY_STATE = {
    "Himachal Pradesh": [
      "Shimla",
      "Kufri",
      "Chail",
      "Narkanda",
      "Manali",
      "Solang Valley",
      "Rohtang Pass",
      "Atal Tunnel",
      "Kullu",
      "Kasol",
      "Manikaran",
      "Jibhi",
      "Tirthan Valley",
      "Dharamshala",
      "McLeod Ganj",
      "Dalhousie",
      "Khajjiar",
      "Spiti Valley",
      "Kalpa",
      "Kaza"
    ],
    "Uttarakhand": [
      "Nainital",
      "Bhimtal",
      "Sattal",
      "Mukteshwar",
      "Kausani",
      "Ranikhet",
      "Mussoorie",
      "Dhanaulti",
      "Dehradun",
      "Rishikesh",
      "Haridwar",
      "Auli",
      "Joshimath",
      "Chopta",
      "Kedarnath",
      "Badrinath",
      "Jim Corbett",
      "Lansdowne"
    ],
    "Kashmir": [
      "Srinagar",
      "Dal Lake",
      "Mughal Gardens",
      "Shankaracharya Temple",
      "Gulmarg",
      "Khilanmarg",
      "Pahalgam",
      "Betaab Valley",
      "Aru Valley",
      "Sonamarg",
      "Thajiwas Glacier",
      "Doodhpathri",
      "Yusmarg",
      "Patnitop",
      "Sanasar"
    ],
    "Ladakh": [
      "Leh",
      "Shanti Stupa",
      "Hall of Fame",
      "Magnetic Hill",
      "Sangam Point",
      "Khardung La",
      "Nubra Valley",
      "Hunder",
      "Diskit Monastery",
      "Pangong Lake",
      "Tso Moriri",
      "Lamayuru",
      "Hemis Monastery",
      "Thiksey Monastery",
      "Hanle"
    ],
    "Kerala": [
      "Munnar",
      "Tea Gardens",
      "Mattupetty Dam",
      "Echo Point",
      "Top Station",
      "Alleppey",
      "Backwaters",
      "Houseboat",
      "Kumarakom",
      "Kochi",
      "Fort Kochi",
      "Thekkady",
      "Periyar",
      "Wayanad",
      "Varkala",
      "Kovalam",
      "Athirappilly",
      "Vagamon"
    ],
    "Goa": [
      "North Goa Beaches",
      "Baga Beach",
      "Calangute Beach",
      "Candolim",
      "Anjuna",
      "Vagator",
      "South Goa Beaches",
      "Colva",
      "Palolem",
      "Dudhsagar Falls",
      "Old Goa Churches",
      "Panaji",
      "Fontainhas"
    ],
    "Rajasthan": [
      "Jaipur",
      "Amber Fort",
      "City Palace Jaipur",
      "Jodhpur",
      "Mehrangarh Fort",
      "Udaipur",
      "City Palace Udaipur",
      "Lake Pichola",
      "Jaisalmer",
      "Sam Sand Dunes",
      "Pushkar",
      "Ajmer",
      "Mount Abu",
      "Chittorgarh"
    ],
    "Northeast India": [
      "Gangtok",
      "Tsomgo Lake",
      "Pelling",
      "Darjeeling",
      "Shillong",
      "Cherrapunji",
      "Dawki",
      "Kaziranga",
      "Tawang",
      "Bomdila",
      "Ziro",
      "Majuli",
      "Aizawl",
      "Kohima"
    ]
  };

  function escapeHTML(str = "") {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function cleanPhone(value = "") {
    return String(value).replace(/\D/g, "").slice(0, 10);
  }

  function safeJSON(value) {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }

  function titleCase(str = "") {
    return str
      .toLowerCase()
      .split(" ")
      .map((s) => (s ? s[0].toUpperCase() + s.slice(1) : s))
      .join(" ");
  }

  function formatDateISO(date) {
    const d = new Date(date);
    if (Number.isNaN(d.getTime())) return "";
    return d.toISOString().split("T")[0];
  }

  function addDays(startDate, days) {
    if (!startDate || !days) return "";
    const d = new Date(startDate);
    d.setDate(d.getDate() + Number(days) - 1);
    return formatDateISO(d);
  }

  function updateEndDate() {
    if (!startDateEl || !daysEl || !endDateEl) return;
    endDateEl.value = addDays(startDateEl.value, daysEl.value);
  }

  function setMsg(text = "", type = "") {
    if (!msgEl) return;
    msgEl.className = "msg";
    if (type === "ok") msgEl.classList.add("ok");
    if (type === "err") msgEl.classList.add("err");
    msgEl.textContent = text;
  }

  function getStoredCustomerProfile() {
    return safeJSON(localStorage.getItem(CUSTOMER_PROFILE_KEY)) || {};
  }

  function saveCustomerProfile(data) {
    const existing = getStoredCustomerProfile();
    const next = {
      ...existing,
      ...data,
      phone: cleanPhone(data?.phone || existing.phone || ""),
      name: (data?.name || existing.name || "").trim(),
      email: (data?.email || existing.email || "").trim(),
      updatedAt: new Date().toISOString()
    };

    if (!next.phone) return existing;

    localStorage.setItem(CUSTOMER_PROFILE_KEY, JSON.stringify(next));
    localStorage.setItem("HKE_CUSTOMER_PHONE", next.phone);
    localStorage.setItem("HKE_OTP_VERIFIED_PHONE", next.phone);
    localStorage.setItem("HKE_CUSTOMER_NAME", next.name || "");
    localStorage.setItem("HKE_CUSTOMER_EMAIL", next.email || "");
    localStorage.setItem("HKE_ORDER_LOOKUP_MODE", "phone");
    localStorage.setItem("HKE_ORDER_LOOKUP_VALUE", next.phone);
    return next;
  }

  function autofillCustomerFields() {
    const profile = getStoredCustomerProfile();
    const storedPhone =
      profile.phone ||
      localStorage.getItem("HKE_CUSTOMER_PHONE") ||
      localStorage.getItem("HKE_OTP_VERIFIED_PHONE") ||
      "";
    const storedName = profile.name || localStorage.getItem("HKE_CUSTOMER_NAME") || "";
    const storedEmail = profile.email || localStorage.getItem("HKE_CUSTOMER_EMAIL") || "";

    if (nameEl && !String(nameEl.value || "").trim() && storedName) nameEl.value = storedName;
    if (emailEl && !String(emailEl.value || "").trim() && storedEmail) emailEl.value = storedEmail;
    if (phoneEl && !String(phoneEl.value || "").trim() && storedPhone) phoneEl.value = cleanPhone(storedPhone);
  }

  async function logWhatsAppEvent(payload) {
    try {
      await fetch(API_WHATSAPP_LOG, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (_error) {
      // WhatsApp should still open even if logging is unavailable.
    }
  }

  function bindWhatsAppLink(anchor, buildPayload) {
    if (!anchor || anchor.dataset.hkeWaBound === "true") return;
    anchor.dataset.hkeWaBound = "true";

    anchor.addEventListener("click", (event) => {
      const href = anchor.getAttribute("href");
      if (!href) return;
      event.preventDefault();

      const payload = typeof buildPayload === "function" ? buildPayload() : {};
      logWhatsAppEvent(payload).finally(() => {
        window.open(href, "_blank", "noopener");
      });
    });
  }

  function showLoadingOverlay() {
    if (loadingOverlayEl) loadingOverlayEl.style.display = "flex";
    if (generateBtn) {
      generateBtn.disabled = true;
      generateBtn.textContent = "Creating your itinerary with HKE AI...";
    }
    if (resetBtn) resetBtn.disabled = true;
  }

  function hideLoadingOverlay() {
    if (loadingOverlayEl) loadingOverlayEl.style.display = "none";
    if (generateBtn) {
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate Itinerary";
    }
    if (resetBtn) resetBtn.disabled = false;
  }

  function collectCheckedTravelStyles() {
    return [...document.querySelectorAll('input[name="travelStyle"]:checked')].map((el) => el.value);
  }

  function syncPlacesValue() {
    if (placesValueEl) placesValueEl.value = selectedPlaces.join(", ");
    if (placesErrEl && selectedPlaces.length > 0) placesErrEl.style.display = "none";
  }

  function renderChips() {
    if (!chipsEl) return;
    chipsEl.innerHTML = "";

    selectedPlaces.forEach((place) => {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.innerHTML = `
        <span>${escapeHTML(place)}</span>
        <button type="button" aria-label="Remove">x</button>
      `;
      chip.querySelector("button")?.addEventListener("click", () => {
        selectedPlaces = selectedPlaces.filter((p) => p !== place);
        syncPlacesValue();
        renderChips();
        rebuildPlacesList();
      });
      chipsEl.appendChild(chip);
    });
  }

  function togglePlaceSelection(place, shouldSelect) {
    if (shouldSelect) {
      if (!selectedPlaces.includes(place)) selectedPlaces.push(place);
    } else {
      selectedPlaces = selectedPlaces.filter((p) => p !== place);
    }

    syncPlacesValue();
    renderChips();
  }

  function rebuildPlacesList() {
    if (!placesListEl) return;

    const state = destinationEl?.value || "";
    const search = (placesSearchEl?.value || "").trim().toLowerCase();
    const places = PLACES_BY_STATE[state] || [];

    placesListEl.innerHTML = "";

    if (!state) {
      if (placesHintEl) placesHintEl.textContent = "Select destination first to load places.";
      placesListEl.innerHTML = `<div class="places-empty">Choose a destination to view available tourist places.</div>`;
      return;
    }

    if (placesHintEl) placesHintEl.textContent = `Showing places for: ${state}`;

    const filtered = places.filter((place) => place.toLowerCase().includes(search));

    if (!filtered.length) {
      placesListEl.innerHTML = `<div class="places-empty">No places found for your search.</div>`;
      return;
    }

    filtered.forEach((place) => {
      const row = document.createElement("label");
      row.className = "placeRow";

      const checked = selectedPlaces.includes(place);

      row.innerHTML = `
        <input type="checkbox" ${checked ? "checked" : ""} />
        <span>${escapeHTML(place)}</span>
        <span class="place-name">${checked ? "Selected" : ""}</span>
      `;

      const checkbox = row.querySelector('input[type="checkbox"]');
      const badge = row.querySelector(".place-name");

      checkbox?.addEventListener("change", (e) => {
        const isChecked = Boolean(e.target?.checked);
        togglePlaceSelection(place, isChecked);
        if (badge) badge.textContent = isChecked ? "Selected" : "";
      });

      placesListEl.appendChild(row);
    });
  }

  async function postJSON(url, body) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal
      });
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new Error("Server temporarily unavailable. Please try later.");
      }
      throw new Error("Unable to connect right now. Please try again.");
    } finally {
      window.clearTimeout(timeoutId);
    }

    const text = await res.text();
    let data;

    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }

    if (!res.ok) {
      const err =
        data?.detail?.[0]?.msg ||
        data?.detail ||
        data?.message ||
        data?.error ||
        `HTTP ${res.status}`;
      throw new Error(typeof err === "string" ? err : JSON.stringify(err));
    }

    return data;
  }

  function getPayload() {
    return {
      name: titleCase(nameEl?.value?.trim() || ""),
      email: emailEl?.value?.trim() || "",
      phone: cleanPhone(phoneEl?.value || ""),
      fromLocation: fromLocationEl?.value?.trim() || "",
      destination: destinationEl?.value || "",
      endPoint: endPointEl?.value?.trim() || "",
      startDate: startDateEl?.value || "",
      days: Number(daysEl?.value || 0),
      endDate: endDateEl?.value || "",
      travellers: Number(travellersEl?.value || 2),
      rooms: Number(roomsEl?.value || 1),
      budget: budgetEl?.value || "Standard",
      travelType: travelTypeEl?.value || "Family",
      hotelClass: hotelClassEl?.value || "Standard",
      vehicle: vehicleEl?.value || "Ertiga",
      guide: guideEl?.value || "Without Guide",
      needFood: needFoodEl?.value === "true",
      foodPreference: foodPreferenceEl?.value || "Flexible",
      travelStyle: collectCheckedTravelStyles(),
      places: [...selectedPlaces],
      notes: notesEl?.value?.trim() || ""
    };
  }

  function validatePayload(data) {
    if (!data.name) return "Please enter your name.";
    if (!data.email) return "Please enter your email.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) return "Please enter a valid email.";
    if (!data.phone || data.phone.length !== 10) return "Please enter a valid 10-digit mobile number.";
    if (!data.fromLocation) return "Please enter your starting location.";
    if (!data.destination) return "Please select destination.";
    if (!data.endPoint) return "Please enter trip end point.";
    if (!data.startDate) return "Please select start date.";
    if (!data.days || data.days < 2) return "Minimum trip duration is 2 days.";
    if (!data.places.length) return "Please select at least one tourist place.";
    return "";
  }

  function renderStructuredOutput(itinerary) {
    if (!outputEl || !itMetaEl || !extraInfoEl || !itCardsEl) return;

    outputEl.style.display = "block";

    const meta = itinerary.meta || {};
    const extra = itinerary.extraInfo || {};
    const days = itinerary.days || [];

    itMetaEl.innerHTML = `
      <span>Destination: ${escapeHTML(meta.destination || "-")}</span>
      <span>Route: ${escapeHTML(meta.route || "-")}</span>
      <span>Dates: ${escapeHTML(meta.dates || "-")}</span>
      <span>Travellers: ${escapeHTML(meta.travellers || "-")}</span>
      <span>Rooms: ${escapeHTML(meta.rooms || "-")}</span>
    `;

    extraInfoEl.innerHTML = `
      <span>Budget: ${escapeHTML(extra.budget || "-")}</span>
      <span>Travel Type: ${escapeHTML(extra.travelType || "-")}</span>
      <span>Hotel: ${escapeHTML(extra.hotel || "-")}</span>
      <span>Vehicle: ${escapeHTML(extra.vehicle || "-")}</span>
      <span>Guide: ${escapeHTML(extra.guide || "-")}</span>
      <span>Food: ${escapeHTML(extra.food || "-")}</span>
    `;

    itCardsEl.innerHTML = days.map((day) => {
      const acts = (day.activities || [])
        .map((activity) => `<li>${escapeHTML(activity)}</li>`)
        .join("");

      return `
        <div class="dayCard">
          <div class="dayTitle">
            <div class="left">Day ${escapeHTML(day.day)} - ${escapeHTML(day.title || "")}</div>
            <div class="right">${escapeHTML(day.date || "")}</div>
          </div>
          <div style="margin-bottom:8px;color:rgba(240,208,138,.85);font-weight:550;">
            ${escapeHTML(day.route || "")}
          </div>
          <ul class="dayList">${acts}</ul>
        </div>
      `;
    }).join("");

    if (postActionsEl) postActionsEl.style.display = "block";
  }

  function buildCustomerCareMessage(data) {
    return encodeURIComponent(
      `Hello HKE Customer Care, I need help with my AI trip plan for ${data.destination}.`
    );
  }

  function itineraryToText(data, itinerary) {
    const meta = itinerary?.meta || {};
    const extra = itinerary?.extraInfo || {};
    const days = itinerary?.days || [];

    const lines = [
      `${itinerary?.title || `${data.destination} Itinerary`}`,
      "",
      `Name: ${data.name}`,
      `Mobile: ${data.phone}`,
      `Destination: ${meta.destination || data.destination}`,
      `Route: ${meta.route || data.places.join(", ")}`,
      `Dates: ${meta.dates || `${data.startDate} to ${data.endDate}`}`,
      `Travellers: ${meta.travellers || data.travellers}`,
      `Rooms: ${meta.rooms || data.rooms}`,
      `Budget: ${extra.budget || data.budget}`,
      `Travel Type: ${extra.travelType || data.travelType}`,
      `Hotel: ${extra.hotel || data.hotelClass}`,
      `Vehicle: ${extra.vehicle || data.vehicle}`,
      "",
      `${itinerary?.summary || "Generated itinerary details:"}`
    ];

    days.forEach((day) => {
      lines.push("");
      lines.push(`Day ${day.day}: ${day.title || ""}`.trim());
      if (day.route) lines.push(`Route: ${day.route}`);
      (day.activities || []).forEach((activity) => lines.push(`- ${activity}`));
    });

    return lines.join("\n").trim();
  }

  function buildItineraryMessage(data, itinerary) {
    return encodeURIComponent(
      `Hello HKE Team,\n\nPlease find my generated itinerary below.\n\n${itineraryToText(data, itinerary)}`
    );
  }

  function wirePostActions(data, itinerary) {
    const careUrl = `https://wa.me/${COMPANY_PHONE}?text=${buildCustomerCareMessage(data)}`;
    const messageUrl = `https://wa.me/${COMPANY_PHONE}?text=${buildItineraryMessage(data, itinerary)}`;

    if (careBtn) careBtn.href = careUrl;
    if (messageBtn) messageBtn.href = messageUrl;
    if (editBtn) editBtn.href = `${RESULT_PAGE}#edit-itinerary`;

    const latestPlan = {
      customer: data,
      itinerary,
      createdAt: new Date().toISOString()
    };

    localStorage.setItem("hkeLatestTripPlan", JSON.stringify(latestPlan));
    sessionStorage.setItem("hke_customer_data", JSON.stringify(data));
    sessionStorage.setItem("hke_itinerary_data", JSON.stringify(itinerary));
    sessionStorage.setItem("hke_itinerary_text", itineraryToText(data, itinerary));
    saveCustomerProfile({
      phone: data.phone,
      name: data.name,
      email: data.email,
      lastDestination: data.destination
    });

    bindWhatsAppLink(careBtn, () => ({
      phone: data.phone,
      messageType: "customer_care",
      message: `AI planner customer care request for ${data.destination || "trip"}`,
      status: "open_requested"
    }));
    bindWhatsAppLink(messageBtn, () => ({
      phone: data.phone,
      messageType: "itinerary_share",
      message: itineraryToText(data, itinerary),
      status: "open_requested"
    }));
  }

  destinationEl?.addEventListener("change", () => {
    selectedPlaces = [];
    syncPlacesValue();
    renderChips();
    if (placesSearchEl) placesSearchEl.value = "";
    rebuildPlacesList();
  });

  placesSearchEl?.addEventListener("input", rebuildPlacesList);

  placesClearBtn?.addEventListener("click", () => {
    selectedPlaces = [];
    syncPlacesValue();
    renderChips();
    rebuildPlacesList();
  });

  startDateEl?.addEventListener("change", updateEndDate);
  daysEl?.addEventListener("input", updateEndDate);
  daysEl?.addEventListener("change", updateEndDate);

  phoneEl?.addEventListener("input", () => {
    phoneEl.value = cleanPhone(phoneEl.value);
  });

  if (startDateEl && !startDateEl.value) {
    startDateEl.value = formatDateISO(new Date());
  }

  autofillCustomerFields();
  updateEndDate();
  rebuildPlacesList();
  renderChips();
  syncPlacesValue();

  resetBtn?.addEventListener("click", () => {
    form.reset();
    selectedPlaces = [];

    if (startDateEl) startDateEl.value = formatDateISO(new Date());
    if (budgetEl) budgetEl.value = "Standard";
    if (hotelClassEl) hotelClassEl.value = "Standard";
    if (vehicleEl) vehicleEl.value = "Ertiga";
    if (guideEl) guideEl.value = "Without Guide";
    if (needFoodEl) needFoodEl.value = "true";
    if (placesSearchEl) placesSearchEl.value = "";

    updateEndDate();
    renderChips();
    syncPlacesValue();
    rebuildPlacesList();
    setMsg("");

    if (outputEl) outputEl.style.display = "none";
    if (postActionsEl) postActionsEl.style.display = "none";
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isGenerating) return;

    updateEndDate();

    const payload = getPayload();
    const validationError = validatePayload(payload);

    if (validationError) {
      setMsg(validationError, "err");
      if (!payload.places.length && placesErrEl) placesErrEl.style.display = "block";
      return;
    }

    if (placesErrEl) placesErrEl.style.display = "none";
    isGenerating = true;
    setMsg("Creating your itinerary with HKE AI...");
    showLoadingOverlay();
    saveCustomerProfile({
      phone: payload.phone,
      name: payload.name,
      email: payload.email,
      lastDestination: payload.destination
    });

    try {
      const resp = await postJSON(API_GENERATE, payload);
      const itinerary = resp?.itinerary || {};

      renderStructuredOutput(itinerary);
      wirePostActions(payload, itinerary);

      setMsg("Itinerary generated successfully.", "ok");

      setTimeout(() => {
        window.location.href = RESULT_PAGE;
      }, 600);
    } catch (err) {
      console.error(err);
      setMsg(err.message || "Unable to generate itinerary right now.", "err");
    } finally {
      isGenerating = false;
      hideLoadingOverlay();
    }
  });
})();
