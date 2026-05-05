/* ============================================================
   HKE Booking Page - scripts/book.js
   - Razorpay payment
   - Saves success data
   - Redirects to booking-success.html
   ============================================================ */

(() => {
  "use strict";

  // =========================
  // CONFIG
  // =========================
  const LS_BOOKING_KEY = "HKE_BOOKING_DRAFT";
  const LS_SUCCESS_KEY = "HKE_BOOKING_SUCCESS";

  const API_BASE = "https://hke-backend.onrender.com";
  const API_PAYMENT_CONFIG = `${API_BASE}/api/payment/config`;
  const API_CREATE_ORDER = `${API_BASE}/api/payment/create-order`;
  const API_VERIFY_PAYMENT = `${API_BASE}/api/payment/verify`;

  const WHATSAPP_COMPANY = "919797294747";

  const GST_RATE = 0.05;
  const ADV_RATE = 0.20;

  // =========================
  // HELPERS
  // =========================
  function byId(id) {
    return document.getElementById(id);
  }

  function safe(value) {
    return String(value ?? "").trim();
  }

  function rupee(n) {
    return `₹ ${Number(n || 0).toLocaleString("en-IN")}`;
  }

  function encodeWA(text) {
    return `https://wa.me/${WHATSAPP_COMPANY}?text=${encodeURIComponent(text)}`;
  }

  function randomRef() {
    return "HKE-" + Date.now().toString(36).toUpperCase();
  }

  function toast(msg) {
    let box = document.getElementById("hkeToast");
    if (!box) {
      box = document.createElement("div");
      box.id = "hkeToast";
      box.style.cssText = `
        position: fixed;
        left: 50%;
        bottom: 24px;
        transform: translateX(-50%);
        background: rgba(0,0,0,.82);
        color: #fff;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,.12);
        z-index: 99999;
        font: 700 14px Inter, system-ui, sans-serif;
        backdrop-filter: blur(10px);
        max-width: 92vw;
        text-align: center;
      `;
      document.body.appendChild(box);
    }

    box.textContent = msg;
    box.style.opacity = "1";

    setTimeout(() => {
      box.style.opacity = "0";
    }, 2400);
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const txt = await res.text();
    let data = null;

    try {
      data = JSON.parse(txt);
    } catch {
      data = { raw: txt };
    }

    if (!res.ok) {
      const msg =
        data?.detail ||
        data?.message ||
        data?.error ||
        txt ||
        `Request failed (${res.status})`;

      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }

    return data;
  }

  async function getJSON(url) {
    const res = await fetch(url);
    const txt = await res.text();
    let data = null;

    try {
      data = JSON.parse(txt);
    } catch {
      data = { raw: txt };
    }

    if (!res.ok) {
      const msg =
        data?.detail ||
        data?.message ||
        data?.error ||
        txt ||
        `Request failed (${res.status})`;

      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }

    return data;
  }

  function estimateBase(d) {
    const days = Number(d.days || 5);
    const hotel = String(d.hotelClass || "Standard");
    const vehicle = String(d.vehicle || "SUV");

    let perDay = 5500;

    if (hotel === "Budget") perDay = 4500;
    if (hotel === "Standard") perDay = 5500;
    if (hotel === "Deluxe") perDay = 7000;
    if (hotel === "Premium") perDay = 8500;
    if (hotel === "Luxury") perDay = 11500;

    if (vehicle === "Sedan") perDay -= 300;
    if (vehicle === "Innova") perDay += 600;
    if (vehicle === "Tempo Traveller") perDay += 1200;

    const placesCount = Array.isArray(d.placesArr) ? d.placesArr.length : 0;
    const factor = Math.min(1.18, 1 + (placesCount * 0.02));

    return Math.max(Math.round(perDay * days * factor), 18000);
  }

  function formatDueDate(dateStr) {
    if (!dateStr) return "";
    const d = new Date(dateStr + "T00:00:00");
    if (isNaN(d.getTime())) return "";
    d.setDate(d.getDate() - 1);

    return d.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric"
    });
  }

  function saveSuccessData(payload) {
    localStorage.setItem(LS_SUCCESS_KEY, JSON.stringify(payload));
  }

  // =========================
  // ITINERARY RENDER
  // =========================
  function renderAccordionFromJson(itJson) {
    const acc = byId("itAccordion");
    if (!acc) return;

    const days = Array.isArray(itJson?.day_wise) ? itJson.day_wise : [];
    acc.innerHTML = "";

    days.forEach((dayObj, idx) => {
      const id = "d" + idx;

      const title = `Day ${safe(dayObj.day)} — ${safe(dayObj.from)} → ${safe(dayObj.to)}${safe(dayObj.drive_time) ? ` | ${safe(dayObj.drive_time)}` : ""}${safe(dayObj.start_time) ? ` | Start ${safe(dayObj.start_time)}` : ""}`;

      const plan = Array.isArray(dayObj.plan) ? dayObj.plan : [];
      const meals = Array.isArray(dayObj.meals_breaks) ? dayObj.meals_breaks : [];

      const html = `
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button ${idx ? "collapsed" : ""}" type="button"
              data-bs-toggle="collapse" data-bs-target="#${id}">
              ${title}
            </button>
          </h2>
          <div id="${id}" class="accordion-collapse collapse ${idx ? "" : "show"}">
            <div class="accordion-body">
              <ul style="margin:0; padding-left:18px;">
                ${plan.map(x => `<li>${safe(x)}</li>`).join("")}
              </ul>

              ${meals.length ? `
                <div style="margin-top:10px; font-weight:700;">Meals / breaks</div>
                <ul style="margin:6px 0 0; padding-left:18px;">
                  ${meals.map(x => `<li>${safe(x)}</li>`).join("")}
                </ul>
              ` : ""}

              ${dayObj.night_stay ? `
                <div style="margin-top:10px; font-weight:700;">Night stay:</div>
                <div>${safe(dayObj.night_stay)}</div>
              ` : ""}
            </div>
          </div>
        </div>
      `;
      acc.insertAdjacentHTML("beforeend", html);
    });
  }

  function renderAccordionFromText(text) {
    const acc = byId("itAccordion");
    if (!acc) return;

    const lines = String(text || "")
      .replace(/\r/g, "")
      .split("\n")
      .map(l => l.trim())
      .filter(Boolean);

    const dayStarts = [];

    for (let i = 0; i < lines.length; i++) {
      if (/^day\s*\d+/i.test(lines[i])) dayStarts.push(i);
    }

    const blocks = [];
    if (!dayStarts.length) {
      blocks.push({
        title: "Itinerary",
        bullets: lines.slice(0, 50)
      });
    } else {
      for (let i = 0; i < dayStarts.length; i++) {
        const s = dayStarts[i];
        const e = i + 1 < dayStarts.length ? dayStarts[i + 1] : lines.length;
        blocks.push({
          title: lines[s],
          bullets: lines.slice(s + 1, e).slice(0, 30)
        });
      }
    }

    acc.innerHTML = "";

    blocks.forEach((b, idx) => {
      const id = "t" + idx;

      const html = `
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button ${idx ? "collapsed" : ""}" type="button"
              data-bs-toggle="collapse" data-bs-target="#${id}">
              ${safe(b.title)}
            </button>
          </h2>
          <div id="${id}" class="accordion-collapse collapse ${idx ? "" : "show"}">
            <div class="accordion-body">
              <ul style="margin:0; padding-left:18px;">
                ${(b.bullets || []).map(x => `<li>${safe(x.replace(/^[-•✅]+/g, "").trim())}</li>`).join("")}
              </ul>
            </div>
          </div>
        </div>
      `;

      acc.insertAdjacentHTML("beforeend", html);
    });
  }

  // =========================
  // PAYMENT
  // =========================
  async function loadRazorpayKey() {
    const data = await getJSON(API_PAYMENT_CONFIG);
    return data?.razorpayKeyId || "";
  }

  async function startPayment(d, bookingRef, base, gst, total, advance) {
    toast("Opening secure payment...");

    const keyFromConfig = await loadRazorpayKey();

    const remaining = total - advance;
    const dueDate = formatDueDate(d.startDate);

    const orderResp = await postJSON(API_CREATE_ORDER, {
      amount: advance * 100,
      currency: "INR",
      receipt: bookingRef,
      notes: {
        booking_ref: bookingRef,
        customer_name: safe(d.name),
        customer_phone: safe(d.phone),
        customer_email: safe(d.email),
        destination: safe(d.destination),
        start_date: safe(d.startDate),
        end_date: safe(d.endDate),
        days: safe(d.days),
        travellers: safe(d.travellers),
        rooms: safe(d.rooms),
        hotel_class: safe(d.hotelClass),
        vehicle: safe(d.vehicle),
        guide: safe(d.guide),
        places: Array.isArray(d.placesArr) ? d.placesArr.join(", ") : "",
        itinerary: ""
      }
    });

    const order = orderResp?.order;
    const razorpayKey = orderResp?.key || keyFromConfig;

    if (!order?.id || !razorpayKey) {
      throw new Error("Payment configuration is incomplete.");
    }

    const options = {
      key: razorpayKey,
      amount: order.amount,
      currency: order.currency,
      name: "Himalayan Kerala Expeditions",
      description: "Book your slot (20% advance)",
      order_id: order.id,
      prefill: {
        name: safe(d.name),
        email: safe(d.email),
        contact: safe(d.phone)
      },
      notes: {
        booking_ref: bookingRef,
        destination: safe(d.destination)
      },
      theme: {
        color: "#D9B25F"
      },
      handler: async function (response) {
        try {
          const vr = await postJSON(API_VERIFY_PAYMENT, {
            razorpay_order_id: response.razorpay_order_id,
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_signature: response.razorpay_signature
          });

          if (vr?.ok) {
            saveSuccessData({
              bookingRef: vr.booking_ref || bookingRef,
              bookingStatus: vr.booking_status || "received",
              paymentStatus: vr.payment_status || "paid",
              customer: {
                name: safe(d.name),
                phone: safe(d.phone),
                email: safe(d.email)
              },
              trip: {
                destination: safe(d.destination),
                startPoint: safe(d.startPoint),
                endPoint: safe(d.endPoint),
                startDate: safe(d.startDate),
                endDate: safe(d.endDate),
                days: safe(d.days),
                travellers: safe(d.travellers),
                rooms: safe(d.rooms),
                hotelClass: safe(d.hotelClass),
                vehicle: safe(d.vehicle),
                guide: safe(d.guide),
                places: Array.isArray(d.placesArr) ? d.placesArr : []
              },
              payment: {
                razorpayOrderId: response.razorpay_order_id,
                razorpayPaymentId: response.razorpay_payment_id,
                baseAmount: base,
                gstAmount: gst,
                totalAmount: total,
                advancePaid: advance,
                remainingBalance: remaining,
                dueDate: dueDate
              },
              createdAt: new Date().toISOString()
            });

            window.location.href = "booking-success.html";
          } else {
            alert("Payment received but verification failed. Please contact support.");
          }
        } catch (err) {
          console.error(err);
          alert("Payment received. Verification is pending. Please contact support if needed.");
        }
      },
      modal: {
        ondismiss: function () {
          toast("Payment popup closed.");
        }
      }
    };

    const rzp = new Razorpay(options);

    rzp.on("payment.failed", function () {
      toast("Payment failed. Please try again.");
    });

    rzp.open();
  }

  // =========================
  // INIT
  // =========================
  function init() {
    const raw = localStorage.getItem(LS_BOOKING_KEY);
    const noData = byId("noData");
    const content = byId("content");

    if (!raw) {
      if (noData) noData.style.display = "block";
      if (content) content.style.display = "none";
      return;
    }

    let draft = null;
    try {
      draft = JSON.parse(raw);
    } catch {
      draft = null;
    }

    if (!draft?.data) {
      if (noData) noData.style.display = "block";
      if (content) content.style.display = "none";
      return;
    }

    const d = draft.data;
    const bookingRef = draft.bookingRef || randomRef();

    draft.bookingRef = bookingRef;
    localStorage.setItem(LS_BOOKING_KEY, JSON.stringify(draft));

    const waText = `Hi HKE, I need help with booking. Booking Ref: ${bookingRef}`;
    const chatLink = encodeWA(waText);

    if (byId("topChat")) byId("topChat").href = chatLink;
    if (byId("chatBtn")) byId("chatBtn").href = chatLink;

    if (noData) noData.style.display = "none";
    if (content) content.style.display = "block";

    const pills = byId("pills");
    if (pills) {
      pills.innerHTML = `
        <span class="pill">📍 ${safe(d.destination)}</span>
        <span class="pill">🗓️ ${safe(d.startDate)} → ${safe(d.endDate)} (${safe(d.days)} days)</span>
        <span class="pill">👥 ${safe(d.travellers || 2)} Travellers</span>
        <span class="pill">🏨 ${safe(d.hotelClass)}</span>
        <span class="pill">🚗 ${safe(d.vehicle)}</span>
        <span class="pill">Ref: ${bookingRef}</span>
      `;
    }

    const cust = byId("cust");
    if (cust) {
      cust.innerHTML = `
        <div><b>${safe(d.name)}</b></div>
        <div>${safe(d.phone)} • ${safe(d.email)}</div>
      `;
    }

    const route = byId("route");
    if (route) {
      route.innerHTML = `
        <div><b>${safe(d.startPoint)} → ${safe(d.endPoint)}</b></div>
        <div>Rooms: ${safe(d.rooms || 1)} • Places: ${Array.isArray(d.placesArr) ? d.placesArr.join(", ") : ""}</div>
      `;
    }

    if (draft.itineraryJson?.day_wise) {
      renderAccordionFromJson(draft.itineraryJson);
    } else {
      renderAccordionFromText(draft.itineraryText || "");
    }

    const base = draft.priceBase && draft.priceBase > 0
      ? Number(draft.priceBase)
      : estimateBase(d);

    const gst = Math.round(base * GST_RATE);
    const total = base + gst;
    const advance = Math.round(total * ADV_RATE);

    const priceBox = byId("priceBox");
    if (priceBox) {
      priceBox.innerHTML = `
        <div class="rowLine"><div>Base package (estimate)</div><div>${rupee(base)}</div></div>
        <div class="rowLine"><div>GST</div><div>${rupee(gst)}</div></div>
        <div class="rowLine"><div><b>Total (estimate)</b></div><div><b>${rupee(total)}</b></div></div>
        <div class="rowLine"><div>Advance (20%)</div><div>${rupee(advance)}</div></div>
      `;
    }

    const payBtn = byId("payBtn");
    if (payBtn) {
      payBtn.addEventListener("click", async () => {
        try {
          await startPayment(d, bookingRef, base, gst, total, advance);
        } catch (e) {
          console.error(e);
          alert(e.message || "Unable to start payment.");
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();