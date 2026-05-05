(function () {
  "use strict";

  var API_BASE = "https://hke-backend.onrender.com";
  var BOOKING_CONTEXT_KEY = "HKE_BOOKING_CONTEXT";
  var CONFIRMATION_KEY = "hkeLastConfirmedBooking";
  var PAYMENT_SUCCESS_KEY = "hkeLastPaymentSuccess";
  var FALLBACK_TRIP_KEY = "hkeLatestTripPlan";
  var PACKAGE_DEFAULTS = {
    kashmir: { packageName: "Kashmir Delight 6D/5N", destination: "Kashmir", pricePerTraveller: 16999, tripDays: 6 },
    manali: { packageName: "Manali Mountain Escape 6D/5N", destination: "Manali", pricePerTraveller: 15999, tripDays: 6 },
    kerala: { packageName: "Kerala Backwater Bliss 6D/5N", destination: "Kerala", pricePerTraveller: 18499, tripDays: 6 },
    himachal: { packageName: "Himachal Grand Circuit 7D/6N", destination: "Himachal Pradesh", pricePerTraveller: 19999, tripDays: 7 },
    leh: { packageName: "Leh Ladakh Explorer", destination: "Leh Ladakh", pricePerTraveller: 22999, tripDays: 6 }
  };

  var els = {};
  var razorpayConfig = null;
  var currentBookingRef = "";

  function $(id) {
    return document.getElementById(id);
  }

  function safeJSONParse(raw) {
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (_err) {
      return null;
    }
  }

  function readContext() {
    return safeJSONParse(window.localStorage.getItem(BOOKING_CONTEXT_KEY)) || {};
  }

  function writeContext(data) {
    window.localStorage.setItem(BOOKING_CONTEXT_KEY, JSON.stringify(data));
  }

  function formatINR(value) {
    return "Rs " + Math.round(Number(value) || 0).toLocaleString("en-IN");
  }

  function normalizePhone(value) {
    return String(value || "").replace(/\D/g, "").slice(-10);
  }

  function addDays(dateString, days) {
    if (!dateString) return "";
    var date = new Date(dateString + "T00:00:00");
    if (Number.isNaN(date.getTime())) return "";
    date.setDate(date.getDate() + Number(days || 0));
    return date.toISOString().slice(0, 10);
  }

  function subtractDays(dateString, days) {
    return addDays(dateString, Number(days || 0) * -1);
  }

  function buildBookingRef() {
    return "HKE-" + Date.now();
  }

  function getQueryValue(params, keys) {
    for (var i = 0; i < keys.length; i += 1) {
      var value = params.get(keys[i]);
      if (value) return value;
    }
    return "";
  }

  function getInitialData() {
    var params = new URLSearchParams(window.location.search || "");
    var slug = getQueryValue(params, ["slug", "packageKey", "packageSlug"]).toLowerCase();
    var defaults = PACKAGE_DEFAULTS[slug] || {};
    var storedTrip = safeJSONParse(window.localStorage.getItem(FALLBACK_TRIP_KEY)) || {};
    var customerTrip = storedTrip.customer || {};
    var itinerary = storedTrip.itinerary || {};
    var storedProfile = window.HKEAuthGate && window.HKEAuthGate.getStoredCustomerProfile
      ? window.HKEAuthGate.getStoredCustomerProfile()
      : {};
    var context = readContext();
    var queryTravellers = Number(getQueryValue(params, ["travellers"]) || 0);
    var queryTotalAmount = Number(getQueryValue(params, ["totalAmount"]) || 0);
    var contextTravellers = Number(context.travellers || customerTrip.travellers || 0);
    var packageName = getQueryValue(params, ["package", "packageName"]) || context.packageName || itinerary.title || customerTrip.packageName || defaults.packageName || "";
    var destination = getQueryValue(params, ["destination"]) || context.destination || customerTrip.destination || defaults.destination || packageName;
    var pricePerTraveller = Number(getQueryValue(params, ["price", "pricePerTraveller"]) || context.pricePerTraveller || defaults.pricePerTraveller || 0);
    var tripDays = Number(getQueryValue(params, ["days", "tripDays"]) || context.tripDays || customerTrip.days || defaults.tripDays || 6);
    var travellers = Number(getQueryValue(params, ["travellers"]) || context.travellers || customerTrip.travellers || 2);
    var rooms = Number(getQueryValue(params, ["rooms"]) || context.rooms || customerTrip.rooms || Math.max(1, Math.ceil(travellers / 2)));
    var startDate = getQueryValue(params, ["startDate"]) || context.startDate || customerTrip.startDate || "";
    var endDate = getQueryValue(params, ["endDate"]) || context.endDate || customerTrip.endDate || addDays(startDate, Math.max(tripDays - 1, 0));
    var originPage = getQueryValue(params, ["origin"]) || context.originPage || document.referrer || "direct";
    var totalAmount = Number(getQueryValue(params, ["totalAmount"]) || context.totalAmount || 0);
    var advanceAmount = Number(getQueryValue(params, ["advanceAmount"]) || context.advanceAmount || 0);
    var remainingAmount = Number(getQueryValue(params, ["remainingAmount"]) || context.remainingAmount || 0);

    if ((!pricePerTraveller || pricePerTraveller <= 0) && totalAmount > 0) {
      var divisor = travellers || queryTravellers || contextTravellers || 1;
      pricePerTraveller = Math.round(totalAmount / Math.max(1, divisor));
    }

    return {
      packageName: packageName,
      destination: destination,
      pricePerTraveller: pricePerTraveller,
      customerName: getQueryValue(params, ["customerName", "name"]) || context.customerName || customerTrip.name || storedProfile.name || "",
      phone: getQueryValue(params, ["phone"]) || context.phone || customerTrip.phone || storedProfile.phone || "",
      email: getQueryValue(params, ["email"]) || context.email || customerTrip.email || storedProfile.email || "",
      startDate: startDate,
      endDate: endDate,
      travellers: travellers,
      rooms: rooms,
      tripDays: tripDays,
      originPage: originPage,
      totalAmount: totalAmount,
      advanceAmount: advanceAmount,
      remainingAmount: remainingAmount
    };
  }

  function getStoredTripPlan() {
    return safeJSONParse(window.localStorage.getItem(FALLBACK_TRIP_KEY)) || {};
  }

  function getSelectedPaymentOption() {
    var checked = document.querySelector('input[name="paymentOption"]:checked');
    return checked ? checked.value : "advance";
  }

  function getFormValues() {
    return {
      packageName: String(els.packageName.value || "").trim(),
      destination: String(els.destination.value || "").trim(),
      pricePerTraveller: Number(els.pricePerTraveller.value || 0),
      customerName: String(els.customerName.value || "").trim(),
      phone: normalizePhone(els.phone.value),
      email: String(els.email.value || "").trim(),
      startDate: String(els.startDate.value || "").trim(),
      endDate: String(els.endDate.value || "").trim(),
      travellers: Number(els.travellers.value || 0),
      rooms: Number(els.rooms.value || 0),
      tripDays: Number(els.tripDays.value || 0),
      originPage: String(els.originPage.value || "").trim(),
      paymentOption: getSelectedPaymentOption()
    };
  }

  function getComputed(values) {
    var totalAmount = Math.max(0, Math.round((Number(values.pricePerTraveller) || 0) * (Number(values.travellers) || 0)));
    var advanceAmount = Math.round(totalAmount * 0.20);
    var remainingAmount = Math.max(0, totalAmount - advanceAmount);
    var payableNow = values.paymentOption === "full" ? totalAmount : advanceAmount;
    var dueDate = subtractDays(values.startDate, 7);
    return {
      totalAmount: totalAmount,
      advanceAmount: advanceAmount,
      remainingAmount: remainingAmount,
      payableNow: payableNow,
      dueDate: dueDate
    };
  }

  function setStatus(message, type) {
    var box = els.bookingStatus;
    box.textContent = message || "";
    box.className = "status-box";
    if (!message) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
    box.classList.add(type || "info");
  }

  function validate(values) {
    if (!values.packageName || !values.destination) return "Package and destination are required.";
    if (!values.customerName) return "Customer name is required.";
    if (values.phone.length !== 10) return "A verified 10-digit phone number is required.";
    if (!values.email || values.email.indexOf("@") === -1) return "A valid email is required.";
    if (!values.startDate) return "Start date is required.";
    if (values.travellers < 1) return "At least one traveller is required.";
    if (values.rooms < 1) return "At least one room is required.";
    if (values.pricePerTraveller <= 0) return "Package price must be greater than zero.";
    return "";
  }

  function updatePaymentCards(option) {
    Array.prototype.forEach.call(document.querySelectorAll(".pay-option"), function (node) {
      node.classList.toggle("active", node.getAttribute("data-payment-option") === option);
    });
  }

  function syncComputedUi() {
    var values = getFormValues();
    if (!values.endDate && values.startDate && values.tripDays > 0) {
      values.endDate = addDays(values.startDate, Math.max(values.tripDays - 1, 0));
      els.endDate.value = values.endDate;
    }
    var computed = getComputed(values);
    if (!currentBookingRef) currentBookingRef = buildBookingRef();

    $("advanceAmountLabel").textContent = formatINR(computed.advanceAmount);
    $("fullAmountLabel").textContent = formatINR(computed.totalAmount);
    $("totalAmountValue").textContent = formatINR(computed.totalAmount);
    $("advanceAmountValue").textContent = formatINR(computed.advanceAmount);
    $("remainingAmountValue").textContent = formatINR(computed.remainingAmount);
    $("payableNowValue").textContent = formatINR(computed.payableNow);
    $("packageTotalValue").textContent = formatINR(computed.totalAmount);
    $("balanceDueDateValue").textContent = computed.dueDate || "-";
    $("bookingRefPreview").textContent = currentBookingRef;
    $("snapshotPackage").textContent = values.packageName || "-";
    $("snapshotDestination").textContent = values.destination || "-";
    $("snapshotPrice").textContent = formatINR(values.pricePerTraveller);
    $("snapshotMode").textContent = values.paymentOption === "full" ? "Full payment" : "20% advance";
    $("heroTitle").textContent = values.packageName ? "Book " + values.packageName + " with HKE." : "Secure your HKE package in one flow.";

    writeContext({
      packageName: values.packageName,
      destination: values.destination,
      pricePerTraveller: values.pricePerTraveller,
      totalAmount: computed.totalAmount,
      advanceAmount: computed.advanceAmount,
      remainingAmount: computed.remainingAmount,
      customerName: values.customerName,
      phone: values.phone,
      email: values.email,
      startDate: values.startDate,
      endDate: values.endDate,
      travellers: values.travellers,
      rooms: values.rooms,
      tripDays: values.tripDays,
      originPage: values.originPage
    });
    updatePaymentCards(values.paymentOption);
    return { values: values, computed: computed };
  }

  async function fetchJSON(url, options) {
    var response = await window.fetch(url, options);
    var data = {};
    try {
      data = await response.json();
    } catch (_err) {
      data = {};
    }
    if (!response.ok || data.ok === false) {
      throw new Error(data.detail || data.message || "Request failed");
    }
    return data;
  }

  async function ensureRazorpayConfig() {
    if (razorpayConfig) return razorpayConfig;
    razorpayConfig = await fetchJSON(API_BASE + "/api/payment/config");
    return razorpayConfig;
  }

  function disablePayButton(disabled) {
    els.proceedBtn.disabled = !!disabled;
    els.proceedBtn.style.opacity = disabled ? "0.65" : "1";
    els.proceedBtn.style.cursor = disabled ? "not-allowed" : "pointer";
  }

  async function startPayment() {
    var synced = syncComputedUi();
    var values = synced.values;
    var computed = synced.computed;
    var validationError = validate(values);
    if (validationError) {
      setStatus(validationError, "error");
      return;
    }

    if (!window.HKEAuthGate || !window.HKEAuthGate.isLoggedIn || !window.HKEAuthGate.isLoggedIn()) {
      setStatus("OTP login is required before booking. Complete login and continue.", "error");
      if (window.HKEAuthGate && window.HKEAuthGate.requireLogin) {
        window.HKEAuthGate.requireLogin("book.html" + window.location.search);
      }
      return;
    }

    try {
      disablePayButton(true);
      setStatus("Checking payment configuration...", "info");
      var config = await ensureRazorpayConfig();
      if (!config.razorpay_enabled || !config.razorpay_key_id) {
        throw new Error("Online payment is not available right now because Razorpay is not configured.");
      }
      if (!window.Razorpay) {
        throw new Error("Razorpay checkout failed to load on this page.");
      }
      if (!currentBookingRef) currentBookingRef = buildBookingRef();

      setStatus("Creating your Razorpay order...", "info");
      var order = await fetchJSON(API_BASE + "/api/payment/create-order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: computed.payableNow,
          currency: "INR",
          receipt: currentBookingRef,
          name: values.customerName,
          email: values.email,
          phone: values.phone,
          trip_name: values.packageName,
          payment_type: values.paymentOption,
          notes: {
            booking_ref: currentBookingRef,
            destination: values.destination,
            start_date: values.startDate,
            end_date: values.endDate,
            travellers: values.travellers,
            rooms: values.rooms,
            total_amount: computed.totalAmount,
            paid_amount: computed.payableNow,
            remaining_amount: Math.max(0, computed.totalAmount - computed.payableNow),
            full_payment_deadline: computed.dueDate,
            origin_page: values.originPage
          }
        })
      });

      var options = {
        key: config.razorpay_key_id,
        amount: Math.round(computed.payableNow * 100),
        currency: "INR",
        name: "Himalayan Kerala Expeditions",
        description: values.paymentOption === "full" ? "Full package payment" : "20% advance package payment",
        order_id: order.order_id,
        prefill: {
          name: values.customerName,
          email: values.email,
          contact: values.phone
        },
        theme: { color: "#d9b25f" },
        modal: {
          ondismiss: function () {
            disablePayButton(false);
            setStatus("Payment popup closed before completion.", "info");
          }
        },
        handler: async function (response) {
          try {
            setStatus("Verifying payment with backend...", "info");
            var verified = await fetchJSON(API_BASE + "/api/payment/verify", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature
              })
            });

            var storedTrip = getStoredTripPlan();
            var plannerCustomer = storedTrip.customer || {};
            var plannerItinerary = storedTrip.itinerary || {};
            var confirmationPayload = {
              customer: {
                bookingRef: verified.booking_ref || currentBookingRef,
                name: values.customerName,
                phone: values.phone,
                email: values.email,
                destination: values.destination,
                packageName: values.packageName,
                fromLocation: plannerCustomer.fromLocation || plannerCustomer.startPoint || values.destination,
                endPoint: plannerCustomer.endPoint || values.destination,
                startDate: values.startDate,
                endDate: values.endDate,
                travellers: values.travellers,
                rooms: values.rooms,
                days: values.tripDays,
                budget: plannerCustomer.budget || "Standard",
                travelType: plannerCustomer.travelType || "Family",
                hotelClass: plannerCustomer.hotelClass || "Standard",
                vehicle: plannerCustomer.vehicle || "SUV",
                guide: plannerCustomer.guide || "Without Guide",
                needFood: !!plannerCustomer.needFood,
                foodPreference: plannerCustomer.foodPreference || "Flexible",
                travelStyle: plannerCustomer.travelStyle || [],
                places: plannerCustomer.places || [],
                notes: plannerCustomer.notes || ""
              },
              itinerary: (plannerItinerary && Object.keys(plannerItinerary).length)
                ? plannerItinerary
                : {
                    title: values.packageName,
                    source: values.originPage
                  },
              pricing: {
                finalFare: computed.totalAmount,
                advanceFare: computed.advanceAmount,
                balanceFare: Math.max(0, computed.totalAmount - computed.advanceAmount)
              },
              payment: {
                bookingRef: verified.booking_ref || currentBookingRef,
                paymentType: values.paymentOption,
                paidAmount: computed.payableNow,
                remainingAmount: Math.max(0, computed.totalAmount - computed.payableNow),
                paymentStatus: values.paymentOption === "full" ? "paid" : "partially_paid",
                fullPaymentDeadline: computed.dueDate,
                nextScheduleText: "Remaining balance must be paid 7 days before travel.",
                razorpayOrderId: response.razorpay_order_id,
                razorpayPaymentId: response.razorpay_payment_id,
                paidAt: new Date().toISOString()
              }
            };

            var saved = await fetchJSON(API_BASE + "/api/payment/save-confirmation", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(confirmationPayload)
            });
            var finalBookingRef = saved.bookingRef || verified.booking_ref || currentBookingRef;
            var stored = {
              bookingRef: finalBookingRef,
              packageName: values.packageName,
              destination: values.destination,
              startDate: values.startDate,
              endDate: values.endDate,
              travellers: values.travellers,
              rooms: values.rooms,
              totalAmount: computed.totalAmount,
              paidAmount: computed.payableNow,
              remainingAmount: Math.max(0, computed.totalAmount - computed.payableNow),
              paymentStatus: values.paymentOption === "full" ? "paid" : "partially_paid",
              bookingStatus: "confirmed",
              razorpayOrderId: response.razorpay_order_id,
              razorpayPaymentId: response.razorpay_payment_id,
              customerName: values.customerName,
              customerPhone: values.phone,
              customerEmail: values.email,
              fullPaymentDeadline: computed.dueDate,
              latestItinerary: confirmationPayload.itinerary,
              sheetWarning: saved.googleSheet && saved.googleSheet.ok === false ? saved.googleSheet.message : ""
            };
            window.localStorage.setItem(CONFIRMATION_KEY, JSON.stringify(stored));
            window.localStorage.setItem(PAYMENT_SUCCESS_KEY, JSON.stringify(stored));
            writeContext(Object.assign({}, readContext(), stored));
            window.location.href = "payment-confirmation.html?bookingRef=" + encodeURIComponent(finalBookingRef);
          } catch (error) {
            disablePayButton(false);
            setStatus(error.message || "Payment succeeded but booking confirmation failed.", "error");
          }
        }
      };

      var razorpay = new window.Razorpay(options);
      razorpay.open();
    } catch (error) {
      disablePayButton(false);
      setStatus(error.message || "Unable to start payment right now.", "error");
    }
  }

  function applyInitialData() {
    var initial = getInitialData();
    els.packageName.value = initial.packageName || "";
    els.destination.value = initial.destination || "";
    els.pricePerTraveller.value = initial.pricePerTraveller || "";
    els.customerName.value = initial.customerName || "";
    els.phone.value = initial.phone || "";
    els.email.value = initial.email || "";
    els.startDate.value = initial.startDate || "";
    els.tripDays.value = initial.tripDays || 6;
    els.endDate.value = initial.endDate || addDays(initial.startDate, Math.max((initial.tripDays || 6) - 1, 0));
    els.travellers.value = initial.travellers || 2;
    els.rooms.value = initial.rooms || 1;
    els.originPage.value = initial.originPage || "direct";
    syncComputedUi();
  }

  function bindEvents() {
    [
      els.packageName,
      els.destination,
      els.pricePerTraveller,
      els.customerName,
      els.phone,
      els.email,
      els.startDate,
      els.travellers,
      els.rooms,
      els.tripDays
    ].forEach(function (node) {
      node.addEventListener("input", syncComputedUi);
      node.addEventListener("change", syncComputedUi);
    });
    Array.prototype.forEach.call(document.querySelectorAll('input[name="paymentOption"]'), function (node) {
      node.addEventListener("change", syncComputedUi);
    });
    els.bookingForm.addEventListener("submit", function (event) {
      event.preventDefault();
      startPayment();
    });
  }

  async function initConfigState() {
    try {
      var config = await ensureRazorpayConfig();
      if (!config.razorpay_enabled || !config.razorpay_key_id) {
        disablePayButton(true);
        setStatus("Online payment is not available right now because Razorpay is not configured.", "error");
        return;
      }
      setStatus("Booking form is ready. Choose your payment option and continue.", "success");
    } catch (error) {
      disablePayButton(true);
      setStatus(error.message || "Unable to load payment configuration.", "error");
    }
  }

  function init() {
    els = {
      bookingForm: $("bookingForm"),
      packageName: $("packageName"),
      destination: $("destination"),
      pricePerTraveller: $("pricePerTraveller"),
      customerName: $("customerName"),
      phone: $("phone"),
      email: $("email"),
      startDate: $("startDate"),
      endDate: $("endDate"),
      travellers: $("travellers"),
      rooms: $("rooms"),
      tripDays: $("tripDays"),
      originPage: $("originPage"),
      bookingStatus: $("bookingStatus"),
      proceedBtn: $("proceedBtn")
    };
    applyInitialData();
    bindEvents();
    initConfigState();
    if (window.HKEAuthGate && window.HKEAuthGate.isLoggedIn && !window.HKEAuthGate.isLoggedIn()) {
      setStatus("OTP login is required before booking. Please complete login to continue.", "info");
      if (window.HKEAuthGate.requireLogin) {
        window.HKEAuthGate.requireLogin("book.html" + window.location.search);
      }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
