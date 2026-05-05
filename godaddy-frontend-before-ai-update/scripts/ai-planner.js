(() => {
  "use strict";

  const API_BASE = "https://hke-backend.onrender.com";
  const API_GENERATE = `${API_BASE}/api/ai/itinerary`;
  const API_CHAT = `${API_BASE}/api/ai/chat`;
  const COMPANY_PHONE = "919797294747";

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

  const placesBtn = $("placesBtn");
  const placesValueEl = $("placesValue");
  const chipsEl = $("chips");
  const placesErrEl = $("placesErr");
  const placesModalEl = $("placesModal");
  const placesSearchEl = $("placesSearch");
  const placesHintEl = $("placesHint");
  const placesListEl = $("placesList");
  const placesClearBtn = $("placesClearBtn");
  const placesDoneBtn = $("placesDoneBtn");

  const quoteBtn = $("quoteBtn");
  const chatUsBtn = $("chatUsBtn");
  const bookNowBtn = $("bookNowBtn");

  const loadingOverlayEl = $("loadingOverlay");

  let selectedPlaces = [];

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

  function showLoadingOverlay() {
    if (loadingOverlayEl) loadingOverlayEl.style.display = "flex";
    if (generateBtn) {
      generateBtn.disabled = true;
      generateBtn.textContent = "Preparing Your Travel Plan...";
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
    return [...document.querySelectorAll('input[name="travelStyle"]:checked')].map(el => el.value);
  }

  function renderChips() {
    if (!chipsEl) return;
    chipsEl.innerHTML = "";

    selectedPlaces.forEach((place) => {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.innerHTML = `
        <span>${escapeHTML(place)}</span>
        <button type="button" aria-label="Remove">×</button>
      `;
      chip.querySelector("button")?.addEventListener("click", () => {
        selectedPlaces = selectedPlaces.filter((p) => p !== place);
        syncPlacesValue();
        renderChips();
        rebuildPlacesList();
      });
      chipsEl.appendChild(chip);
    });

    if (placesBtn) {
      placesBtn.textContent = selectedPlaces.length
        ? `Selected ${selectedPlaces.length} place(s) - click to edit`
        : "Select places (click to choose)…";
    }
  }

  function syncPlacesValue() {
    if (placesValueEl) {
      placesValueEl.value = selectedPlaces.join(", ");
    }
    if (placesErrEl && selectedPlaces.length > 0) {
      placesErrEl.style.display = "none";
    }
  }

  function rebuildPlacesList() {
    if (!placesListEl) return;

    const state = destinationEl?.value || "";
    const search = (placesSearchEl?.value || "").trim().toLowerCase();
    const places = PLACES_BY_STATE[state] || [];

    placesListEl.innerHTML = "";

    if (!state) {
      if (placesHintEl) placesHintEl.textContent = "Select destination first to load places.";
      return;
    }

    if (placesHintEl) {
      placesHintEl.textContent = `Showing places for: ${state}`;
    }

    const filtered = places.filter((p) => p.toLowerCase().includes(search));

    if (!filtered.length) {
      placesListEl.innerHTML = `<div style="padding:12px;opacity:.85;">No places found.</div>`;
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
        if (e.target.checked) {
          if (!selectedPlaces.includes(place)) selectedPlaces.push(place);
        } else {
          selectedPlaces = selectedPlaces.filter((p) => p !== place);
        }

        if (badge) badge.textContent = e.target.checked ? "Selected" : "";
        syncPlacesValue();
        renderChips();
      });

      placesListEl.appendChild(row);
    });
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

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
      <span>Destination: ${escapeHTML(meta.destination || "—")}</span>
      <span>Route: ${escapeHTML(meta.route || "—")}</span>
      <span>Dates: ${escapeHTML(meta.dates || "—")}</span>
      <span>Travellers: ${escapeHTML(meta.travellers || "—")}</span>
      <span>Rooms: ${escapeHTML(meta.rooms || "—")}</span>
    `;

    extraInfoEl.innerHTML = `
      <span>Budget: ${escapeHTML(extra.budget || "—")}</span>
      <span>Travel Type: ${escapeHTML(extra.travelType || "—")}</span>
      <span>Hotel: ${escapeHTML(extra.hotel || "—")}</span>
      <span>Vehicle: ${escapeHTML(extra.vehicle || "—")}</span>
      <span>Guide: ${escapeHTML(extra.guide || "—")}</span>
      <span>Food: ${escapeHTML(extra.food || "—")}</span>
    `;

    itCardsEl.innerHTML = days.map(day => {
      const acts = (day.activities || [])
        .map(a => `<li>${escapeHTML(a)}</li>`)
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

  function buildWhatsAppMessage(data, itinerary) {
    const title = itinerary?.title || `${data.destination} Itinerary`;
    const placesText = data.places.join(", ");

    return encodeURIComponent(
      `Hello HKE Team,\n\n` +
      `I want to continue with this AI plan.\n\n` +
      `Name: ${data.name}\n` +
      `Mobile: ${data.phone}\n` +
      `Destination: ${data.destination}\n` +
      `Places: ${placesText}\n` +
      `Dates: ${data.startDate} to ${data.endDate}\n` +
      `Travellers: ${data.travellers}\n` +
      `Rooms: ${data.rooms}\n` +
      `Plan: ${title}\n\n` +
      `Please share final quote and next steps.`
    );
  }

  function wirePostActions(data, itinerary) {
    const waUrl = `https://wa.me/${COMPANY_PHONE}?text=${buildWhatsAppMessage(data, itinerary)}`;

    if (chatUsBtn) chatUsBtn.href = waUrl;
    if (quoteBtn) quoteBtn.href = waUrl;
    if (bookNowBtn) bookNowBtn.href = "booking-details.html";

    localStorage.setItem("hkeLatestTripPlan", JSON.stringify({
      customer: data,
      itinerary,
      createdAt: new Date().toISOString()
    }));

    sessionStorage.setItem("hke_customer_data", JSON.stringify(data));
    sessionStorage.setItem("hke_itinerary_data", JSON.stringify(itinerary));
  }

  destinationEl?.addEventListener("change", () => {
    selectedPlaces = [];
    syncPlacesValue();
    renderChips();
    if (placesSearchEl) placesSearchEl.value = "";
    rebuildPlacesList();
  });

  placesBtn?.addEventListener("click", () => {
    if (!window.bootstrap || !window.bootstrap.Modal || !placesModalEl) return;
    rebuildPlacesList();
    const modal = window.bootstrap.Modal.getOrCreateInstance(placesModalEl);
    modal.show();
    setTimeout(() => placesSearchEl?.focus(), 150);
  });

  placesSearchEl?.addEventListener("input", rebuildPlacesList);

  placesClearBtn?.addEventListener("click", () => {
    selectedPlaces = [];
    syncPlacesValue();
    renderChips();
    rebuildPlacesList();
  });

  placesDoneBtn?.addEventListener("click", () => {
    syncPlacesValue();
    renderChips();
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

  updateEndDate();
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

    updateEndDate();

    const payload = getPayload();
    const validationError = validatePayload(payload);

    if (validationError) {
      setMsg(validationError, "err");
      if (!payload.places.length && placesErrEl) placesErrEl.style.display = "block";
      return;
    }

    if (placesErrEl) placesErrEl.style.display = "none";
    setMsg("Generating itinerary...");
    showLoadingOverlay();

    try {
      const resp = await postJSON(API_GENERATE, payload);
      const itinerary = resp?.itinerary || {};

      renderStructuredOutput(itinerary);
      wirePostActions(payload, itinerary);

      setMsg("Itinerary generated successfully.", "ok");

      setTimeout(() => {
        window.location.href = "itinerary-result.html";
      }, 600);
    } catch (err) {
      console.error(err);
      setMsg(`Error: ${err.message}`, "err");
    } finally {
      hideLoadingOverlay();
    }
  });
})();