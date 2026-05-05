/* =========================================================
   HKE FINALIZE — CUSTOMER VIEW (v20260211_01)
   - Reads localStorage payload + itinerary
   - Shows final price + activities
   - Internal hotel/taxi calculations are NOT shown to customer
   ========================================================= */

(() => {
  "use strict";

  const LS_KEY = "hke_ai_planner_payload";
  const LS_ITINERARY_KEY = "hke_ai_itinerary_text";

  const $ = (id) => document.getElementById(id);

  const noData = $("noData");
  const mainWrap = $("mainWrap");

  const subTitle = $("subTitle");
  const itineraryText = $("itineraryText");
  const placesLine = $("placesLine");

  const basePriceEl = $("basePrice");
  const activitiesTotalEl = $("activitiesTotal");
  const grandTotalEl = $("grandTotal");

  const activitiesList = $("activitiesList");
  const actWarn = $("actWarn");

  const copySummaryBtn = $("copySummaryBtn");
  const waQuoteBtn = $("waQuoteBtn");
  const copyMsg = $("copyMsg");

  // =========================
  // PRICING CONFIG (EDIT YOUR RATES HERE)
  // NOTE: This is an ESTIMATOR.
  // You can later connect real rates from backend/admin.
  // =========================
  const HOTEL_PER_ROOM_PER_NIGHT = {
    "Budget": 1800,
    "Standard": 2800,
    "Premium": 5200
  };

  const TAXI_PACKAGE = {
    "Sedan": 15000,
    "SUV": 19000,
    "Tempo Traveller": 28000
  };

  const GUIDE_PER_DAY = 1200;

  // Company overhead/profit (simple margin)
  const PROFIT_MARGIN = 0.18; // 18%

  // Activities list (global)
  const ACTIVITIES = [
    { id: "rafting", name: "River Rafting", price: 1500 },
    { id: "paragliding", name: "Paragliding", price: 2800 },
    { id: "gondola", name: "Gondola Ride", price: 1200 },
    { id: "shikara", name: "Shikara Ride", price: 900 },
    { id: "skiing", name: "Skiing (Equipment)", price: 2500 },
    { id: "camping", name: "Camping", price: 1500 }
  ];

  let payload = null;
  let itinerary = "";
  const qtyMap = {}; // activityId -> qty

  // =========================
  // UTILS
  // =========================
  function fmt(n) {
    const x = Math.round(Number(n || 0));
    return x.toLocaleString("en-IN");
  }

  function nightsFromDays(days) {
    const d = Math.max(1, parseInt(days || "1", 10));
    return Math.max(1, d - 1);
  }

  function safeNum(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  function calcBase() {
    // Base = (hotel rooms * nights * roomrate) + taxi + guide + overhead/profit
    const days = parseInt(payload.days || "1", 10);
    const nights = nightsFromDays(days);

    const rooms = Math.max(1, parseInt(payload.rooms || "1", 10));
    const hotelClass = payload.hotelClass || "Standard";
    const roomRate = HOTEL_PER_ROOM_PER_NIGHT[hotelClass] ?? HOTEL_PER_ROOM_PER_NIGHT.Standard;

    const vehicle = payload.vehicle || "SUV";
    const taxi = TAXI_PACKAGE[vehicle] ?? TAXI_PACKAGE.SUV;

    const guide = (payload.guide === "With Guide") ? (GUIDE_PER_DAY * days) : 0;

    const subtotal = (rooms * nights * roomRate) + taxi + guide;

    // profit/overheads
    const total = subtotal * (1 + PROFIT_MARGIN);
    return Math.round(total);
  }

  function calcActivities() {
    let total = 0;
    for (const a of ACTIVITIES) {
      const qty = Math.max(0, parseInt(qtyMap[a.id] || "0", 10));
      total += qty * a.price;
    }
    return Math.round(total);
  }

  function updateTotals() {
    const base = calcBase();
    const act = calcActivities();
    const grand = base + act;

    basePriceEl.textContent = fmt(base);
    activitiesTotalEl.textContent = fmt(act);
    grandTotalEl.textContent = fmt(grand);

    updateWhatsAppLink(base, act, grand);
  }

  function updateWhatsAppLink(base, act, grand) {
    const name = payload.name || "Customer";
    const dest = payload.destination || "";
    const days = payload.days || "";
    const sp = payload.startPoint || "";
    const ep = payload.endPoint || "";
    const travellers = payload.travellers || "";
    const hotelClass = payload.hotelClass || "";
    const vehicle = payload.vehicle || "";
    const guide = payload.guide || "";
    const places = (payload.places || []).join(", ");

    const chosenActs = ACTIVITIES
      .map(a => {
        const q = parseInt(qtyMap[a.id] || "0", 10);
        return q > 0 ? `${a.name} x${q}` : null;
      })
      .filter(Boolean)
      .join(", ") || "None";

    const text =
`HKE Package Summary ✅
Name: ${name}
Destination: ${dest}
Trip: ${sp} → ${ep}
Days: ${days}
Travellers: ${travellers}
Hotel: ${hotelClass}
Vehicle: ${vehicle}
Guide: ${guide}
Places: ${places}

Activities: ${chosenActs}

Base Package: ₹${fmt(base)}
Activities Total: ₹${fmt(act)}
Grand Total: ₹${fmt(grand)}

Note: Meals not included (available on demand).`;

    const url = `https://wa.me/919797294747?text=${encodeURIComponent(text)}`;
    waQuoteBtn.href = url;
  }

  function renderActivities() {
    activitiesList.innerHTML = "";

    for (const a of ACTIVITIES) {
      const row = document.createElement("div");
      row.className = "actCard";
      row.innerHTML = `
        <div>
          <div class="actTitle">${a.name}</div>
          <div class="actPrice">₹${fmt(a.price)} per person</div>
        </div>
        <input class="qty" type="number" min="0" value="${qtyMap[a.id] ?? 0}" />
      `;

      const input = row.querySelector("input");
      input.addEventListener("input", () => {
        const v = input.value;
        const n = Number(v);
        if (!Number.isFinite(n) || n < 0) {
          actWarn.style.display = "block";
          return;
        }
        actWarn.style.display = "none";
        qtyMap[a.id] = String(Math.floor(n));
        updateTotals();
      });

      activitiesList.appendChild(row);
    }
  }

  function buildSubtitle() {
    const dest = payload.destination || "";
    const days = payload.days || "";
    const startDate = payload.startDate || "";
    const travellers = payload.travellers || "";
    return `${dest} • ${days} days • Start ${startDate} • ${travellers} travellers`;
  }

  function copySummary() {
    const base = safeNum(basePriceEl.textContent.replaceAll(",", ""));
    const act = safeNum(activitiesTotalEl.textContent.replaceAll(",", ""));
    const grand = safeNum(grandTotalEl.textContent.replaceAll(",", ""));

    const chosenActs = ACTIVITIES
      .map(a => {
        const q = parseInt(qtyMap[a.id] || "0", 10);
        return q > 0 ? `- ${a.name} x${q} (₹${fmt(a.price)} pp)` : null;
      })
      .filter(Boolean)
      .join("\n") || "- None";

    const text =
`HKE Package Summary
Destination: ${payload.destination}
Trip: ${payload.startPoint} → ${payload.endPoint}
Days: ${payload.days}
Travellers: ${payload.travellers}
Rooms: ${payload.rooms}
Hotel: ${payload.hotelClass}
Vehicle: ${payload.vehicle}
Guide: ${payload.guide}
Places: ${(payload.places || []).join(", ")}

Activities:
${chosenActs}

Base Package: ₹${fmt(base)}
Activities Total: ₹${fmt(act)}
Grand Total: ₹${fmt(grand)}

Note: Meals not included (available on demand).`;

    navigator.clipboard.writeText(text)
      .then(() => (copyMsg.textContent = "✅ Copied! You can paste it anywhere."))
      .catch(() => (copyMsg.textContent = "❌ Copy failed. Please copy manually."));
  }

  // =========================
  // INIT
  // =========================
  function init() {
    try {
      payload = JSON.parse(localStorage.getItem(LS_KEY) || "null");
      itinerary = localStorage.getItem(LS_ITINERARY_KEY) || "";
    } catch (e) {
      payload = null;
      itinerary = "";
    }

    if (!payload || !itinerary || itinerary.trim().length < 10) {
      noData.style.display = "block";
      mainWrap.style.display = "none";
      return;
    }

    noData.style.display = "none";
    mainWrap.style.display = "block";

    subTitle.textContent = buildSubtitle();
    itineraryText.textContent = itinerary;

    const places = (payload.places || []).join(", ");
    placesLine.textContent = places || "—";

    // default quantities = 0
    for (const a of ACTIVITIES) qtyMap[a.id] = "0";

    renderActivities();
    updateTotals();

    copySummaryBtn.addEventListener("click", copySummary);
  }

  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", init)
    : init();

})();
